"""Werewolf tournament scheduler + runner (T27).

Turns a population of model labels into a deterministic, seeded pairwise cross-play
sweep and aggregates the outcomes into a TrueSkill leaderboard. Two layers, both
game-agnostic of how a single game is actually played (the runner is injected):

- `schedule_tournament` is the deterministic scheduler. It plays every model pair
  INCLUDING self-pairs (A-vs-A, a self-play diagnostic), `games_per_pair` games
  each, side-swapped so each model of a distinct pair plays the wolves in exactly
  half that pair's games (controls the known wolf-win bias). Every per-game seed and
  roster derives from one master `GameRNG(seed)`, drawn in schedule order, so the
  whole schedule replays byte-identically (invariant #4): a recorded game
  reconstructs its board from its own seed.

- `run_tournament` drives each `Matchup` through an injected `GameRunner`, reassembles
  the per-game results in schedule order even when run concurrently (a thread pool's
  `map` preserves input order), then rates and rolls them up. Self-play games carry no
  cross-model signal so `rate_games` skips them, but the metric rollups still see
  every game.

This module is a game library — it MUST NOT import the CLI layer.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import itertools
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from social_deduction_bench.engine import GameRNG
from social_deduction_bench.games.werewolf.assignment import assign_default_roles
from social_deduction_bench.games.werewolf.metrics import (
    AggregateMetrics,
    DeceiverDetectorReport,
    GameMetrics,
    aggregate_metrics,
    deceiver_detector_split,
    to_game_results,
)
from social_deduction_bench.games.werewolf.roles import Faction, faction_of
from social_deduction_bench.rating.trueskill import Leaderboard, rate_games

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Matchup:
    """One scheduled game: its identity, seed, dealt roster, and per-seat models.

    `roster` is the seeded `(name, role)` deal from `seed`, and `seat_models` maps
    each seat to its model label (the werewolf model on the wolf seats, the villager
    model elsewhere). Both are stored so a recorded game replays the exact board
    (invariant #4) and so a fake runner can attribute outcomes without re-dealing.
    """

    game_id: str
    seed: int
    roster: tuple[tuple[str, str], ...]
    seat_models: tuple[tuple[str, str], ...]
    werewolf_model: str
    villager_model: str


@dataclass(frozen=True, slots=True)
class TournamentResult:
    """The completed sweep: rated matchups, their metrics, the board, rollups, failures.

    `matchups` and `games` are aligned and in schedule order — they cover only the
    games that actually produced a result. `failed` lists the matchups whose runner
    raised (empty unless `run_tournament(..., skip_failures=True)`).
    """

    matchups: tuple[Matchup, ...]
    games: tuple[GameMetrics, ...]
    leaderboard: Leaderboard
    aggregate: AggregateMetrics
    split: DeceiverDetectorReport
    failed: tuple[Matchup, ...] = ()


class GameRunner(Protocol):
    """Plays one matchup and returns its metrics — the seam between schedule and play."""

    def __call__(self, matchup: Matchup, /) -> GameMetrics: ...


def _san(s: str) -> str:
    """Sanitize a model label for use in a game-id path segment (labels may contain '/')."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def schedule_tournament(
    models: Sequence[str],
    *,
    games_per_pair: int = 10,
    seed: int,
    names: Sequence[str],
) -> tuple[Matchup, ...]:
    """Build the deterministic pairwise cross-play schedule (invariant #4).

    Plays every pair over `combinations_with_replacement` of the sorted unique
    models — so self-pairs are included on purpose (self-play diagnostic). Each
    distinct pair side-swaps which model is the werewolves across its games to
    control the wolf-win bias; a self-pair keeps one model on both sides. Every
    per-game seed is drawn from one master `GameRNG(seed)` in iteration order, so
    the whole schedule reproduces from `seed`, and each game's roster is the seeded
    deal from its own seed.
    """
    if not models:
        raise ValueError("schedule_tournament needs at least one model")
    if games_per_pair < 1:
        raise ValueError(f"games_per_pair must be >= 1, got {games_per_pair}")

    unique = sorted(set(models))
    master = GameRNG(seed)
    matchups: list[Matchup] = []
    idx = 0
    for lo, hi in itertools.combinations_with_replacement(unique, 2):
        for k in range(games_per_pair):
            if lo == hi:
                werewolf_model = villager_model = lo
            elif k % 2 == 0:
                werewolf_model, villager_model = lo, hi
            else:
                werewolf_model, villager_model = hi, lo

            game_seed = master.randrange(2**31)
            roster = assign_default_roles(names, GameRNG(game_seed))
            seat_models = tuple(
                (name, werewolf_model if faction_of(role) is Faction.WEREWOLVES else villager_model)
                for name, role in roster
            )
            game_id = f"g{idx:04d}-{_san(werewolf_model)}-vs-{_san(villager_model)}"
            matchups.append(
                Matchup(
                    game_id=game_id,
                    seed=game_seed,
                    roster=roster,
                    seat_models=seat_models,
                    werewolf_model=werewolf_model,
                    villager_model=villager_model,
                )
            )
            idx += 1
    return tuple(matchups)


def run_tournament(
    matchups: Sequence[Matchup],
    runner: GameRunner,
    *,
    max_concurrency: int = 1,
    skip_failures: bool = False,
) -> TournamentResult:
    """Run every matchup through `runner`, then rate and roll up the results.

    Results are reassembled in schedule order regardless of completion order, so a
    concurrent run is byte-identical to a sequential one (invariant #4 for the
    aggregation step): futures are gathered in submission order. A thread pool (not
    asyncio) is used because the production runner calls `asyncio.run` internally, so
    each game needs its own thread/event loop.

    With `skip_failures` (for an unattended paid sweep), a game whose runner raises is
    logged and dropped — the rest still rate, and the dropped matchups land in
    `TournamentResult.failed`. Without it (the default), a failure propagates so a bug
    fails loud instead of silently shrinking the sweep.
    """
    completed: list[tuple[Matchup, GameMetrics]] = []
    failed: list[Matchup] = []

    def _record(matchup: Matchup, produce: Callable[[], GameMetrics]) -> None:
        try:
            completed.append((matchup, produce()))
        except Exception:
            if not skip_failures:
                raise
            logger.warning("tournament game %s failed, skipping", matchup.game_id, exc_info=True)
            failed.append(matchup)

    if not matchups:
        pass
    elif max_concurrency <= 1:
        for m in matchups:
            _record(m, lambda m=m: runner(m))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_concurrency, len(matchups))) as ex:
            futures = [(m, ex.submit(runner, m)) for m in matchups]
            for m, future in futures:  # submission order == schedule order
                _record(m, future.result)

    games = tuple(metrics for _, metrics in completed)
    leaderboard = rate_games(to_game_results(list(games)))
    aggregate = aggregate_metrics(games)
    split = deceiver_detector_split(games)
    return TournamentResult(
        matchups=tuple(m for m, _ in completed),
        games=games,
        leaderboard=leaderboard,
        aggregate=aggregate,
        split=split,
        failed=tuple(failed),
    )


def tournament_summary_dict(result: TournamentResult) -> dict[str, object]:
    """Build a JSON-serializable summary of the sweep (no timestamps — byte-stable).

    Returned as JSON-canonical types (tuples in the `dataclasses.asdict` rollups
    become lists) by round-tripping through `json`, so the dict equals the loaded
    summary file — the file writer is just `json.dumps` of this dict, and a caller
    can serialize it however they like.
    """
    summary = {
        "n_games": result.leaderboard.n_games,
        "n_skipped": result.leaderboard.n_skipped,
        "n_failed": len(result.failed),
        "failed": [m.game_id for m in result.failed],
        "leaderboard": [dataclasses.asdict(r) for r in result.leaderboard.ratings],
        "aggregate": dataclasses.asdict(result.aggregate),
        "split": dataclasses.asdict(result.split),
        "games": [
            {
                "game_id": m.game_id,
                "seed": m.seed,
                "werewolf_model": m.werewolf_model,
                "villager_model": m.villager_model,
                "winner": g.winner,
            }
            for m, g in zip(result.matchups, result.games, strict=True)
        ],
    }
    return json.loads(json.dumps(summary))


def write_tournament_summary(result: TournamentResult, path: Path) -> None:
    """Write the summary to `path` as deterministic JSON (sorted keys, no wall-clock)."""
    path.write_text(json.dumps(tournament_summary_dict(result), indent=2, sort_keys=True), encoding="utf-8")
