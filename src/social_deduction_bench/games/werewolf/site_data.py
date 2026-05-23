"""Static-site data layer (T28).

`build_site_data` re-aggregates a tournament run's per-game artifacts into the
deterministic `data.json` payload the static results site renders. It reuses the
existing metric/rating functions (`extract_run_dir`, `aggregate_metrics`,
`deceiver_detector_split`, `to_game_results`, `rate_games`) so no analysis logic is
duplicated, and adds two werewolf-shaped aggregations the site needs:

- `head_to_head` — who-beat-whom across DISTINCT models (the winning faction's model
  beat the other's). Self-pairs and mixed/unknown-faction games are unattributable
  and excluded, reusing `_unique_faction_model` (the single source of truth for the
  unknown-model sentinel).
- `self_play_stats` — a model against an identical copy of itself (TrueSkill-excluded
  but revealing of its deceiver/detector balance).

The payload is deterministic (invariant #4): every list is sorted, and no wall-clock,
`git_sha`, or absolute path is embedded, so the same set of completed game dirs yields
a byte-identical `data.json`. A torn or in-progress game dir (no `GAME_OVER`) is logged
and skipped rather than crashing the build, because the run directory is written live
by a parallel sweep; zero usable games fails loud.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_deduction_bench.games.werewolf.metrics import (
    _UNKNOWN_MODEL,
    AggregateMetrics,
    GameMetrics,
    _unique_faction_model,
    aggregate_metrics,
    deceiver_detector_split,
    extract_run_dir,
    to_game_results,
)
from social_deduction_bench.games.werewolf.roles import Faction
from social_deduction_bench.rating.trueskill import rate_games

logger = logging.getLogger(__name__)

_WOLF = Faction.WEREWOLVES.value
_VILLAGE = Faction.VILLAGERS.value


@dataclass(frozen=True, slots=True)
class HeadToHeadCell:
    """One model pair's record: `row` beat `col` `row_wins` times and vice versa."""

    row: str
    col: str
    row_wins: int
    col_wins: int
    games: int


@dataclass(frozen=True, slots=True)
class HeadToHeadMatrix:
    """The who-beat-whom grid: axis `models` plus sparse `cells` (only games > 0)."""

    models: tuple[str, ...]
    cells: tuple[HeadToHeadCell, ...]


@dataclass(frozen=True, slots=True)
class SelfPlayStat:
    """One model's record in games where it staffs both factions (self-play)."""

    model: str
    games: int
    wolf_win_rate: float
    exile_accuracy: float | None  # None when no exile resolved
    mean_rounds: float


def head_to_head(games: Sequence[GameMetrics]) -> HeadToHeadMatrix:
    """Build the who-beat-whom matrix over distinct-model cross-play games.

    For each game the wolf-faction and village-faction models are the unique model
    staffing each side (`_unique_faction_model`); a game is skipped when either side
    is unattributable (mixed / unknown) or when both sides are the same model
    (self-pair). The winning faction's model "beat" the other. Records accumulate
    into an unordered model-pair cell (`row` < `col`); only pairs with games are
    emitted, and `models` lists every cross-pair participant (empty for a
    single-model run).
    """
    pair_wins: dict[tuple[str, str], list[int]] = {}
    for game in games:
        wolf = _unique_faction_model(game.seats, _WOLF)
        village = _unique_faction_model(game.seats, _VILLAGE)
        if wolf is None or village is None or wolf == village:
            continue
        if game.winner == _WOLF:
            winner_model = wolf
        elif game.winner == _VILLAGE:
            winner_model = village
        else:
            continue
        a, b = sorted((wolf, village))
        rec = pair_wins.setdefault((a, b), [0, 0])
        rec[0 if winner_model == a else 1] += 1

    cells = tuple(
        HeadToHeadCell(row=a, col=b, row_wins=rec[0], col_wins=rec[1], games=rec[0] + rec[1])
        for (a, b), rec in sorted(pair_wins.items())
    )
    models = tuple(sorted({m for cell in cells for m in (cell.row, cell.col)}))
    return HeadToHeadMatrix(models=models, cells=cells)


def self_play_stats(games: Sequence[GameMetrics]) -> tuple[SelfPlayStat, ...]:
    """Per-model record over self-play games (one model staffing both factions).

    Exile accuracy POOLS across decisions (Σcorrect / Σtotal, None when no exile was
    resolved), matching `deceiver_detector_split`; `mean_rounds` averages game length.
    Output is sorted by model name.
    """
    games_n: Counter[str] = Counter()
    wolf_wins: Counter[str] = Counter()
    exiles_total: Counter[str] = Counter()
    exiles_correct: Counter[str] = Counter()
    rounds_sum: Counter[str] = Counter()

    for game in games:
        wolf = _unique_faction_model(game.seats, _WOLF)
        village = _unique_faction_model(game.seats, _VILLAGE)
        if wolf is None or village is None or wolf != village:
            continue
        model = wolf
        games_n[model] += 1
        if game.winner == _WOLF:
            wolf_wins[model] += 1
        exiles_total[model] += game.exiles_total
        exiles_correct[model] += game.exiles_correct
        rounds_sum[model] += game.rounds

    return tuple(
        SelfPlayStat(
            model=model,
            games=games_n[model],
            wolf_win_rate=wolf_wins[model] / games_n[model],
            exile_accuracy=(exiles_correct[model] / exiles_total[model] if exiles_total[model] else None),
            mean_rounds=rounds_sum[model] / games_n[model],
        )
        for model in sorted(games_n)
    )


def _cost_efficiency(games: Sequence[GameMetrics], aggregate: AggregateMetrics) -> dict[str, object]:
    seat_costs = [g.total_cost_usd for g in games if g.total_cost_usd is not None]
    totals = {
        "prompt": sum(g.total_prompt_tokens for g in games),
        "completion": sum(g.total_completion_tokens for g in games),
        "lm_calls": sum(g.total_lm_calls for g in games),
        "tool_calls": sum(g.total_tool_calls for g in games),
        "illegal_moves": sum(g.total_illegal_moves for g in games),
        "cost_usd": sum(seat_costs) if seat_costs else None,
    }
    models = [
        {
            "model": m.model,
            "games": m.games,
            "prompt": m.prompt_tokens,
            "completion": m.completion_tokens,
            "illegal_moves": m.illegal_moves,
            "tool_calls": m.tool_calls,
            "win_rate": m.win_rate,
        }
        for m in aggregate.models
        if m.model != _UNKNOWN_MODEL
    ]
    return {
        "totals": totals,
        "mean_game_length": aggregate.mean_game_length,
        "mean_illegal_move_rate": aggregate.mean_illegal_move_rate,
        "models": models,
    }


def _load_games(run_dirs: Sequence[Path]) -> list[GameMetrics]:
    games: list[GameMetrics] = []
    for run_dir in run_dirs:
        for game_dir in sorted(p for p in run_dir.glob("g*") if p.is_dir()):
            try:
                games.append(extract_run_dir(game_dir))
            except (OSError, ValueError, KeyError) as e:
                # A torn / in-progress dir (no GAME_OVER, half-written file) is expected
                # while a sweep writes live: skip it, don't abort the whole site.
                logger.warning("site: skipping unreadable game dir %s: %s", game_dir, e)
    return games


def build_site_data(run_dirs: Sequence[Path]) -> dict[str, Any]:
    """Re-aggregate the per-game artifacts under `run_dirs` into the `data.json` payload.

    Reads every `g*/` game dir, skipping unreadable / in-progress ones, then assembles
    a deterministic dict (sorted lists, no wall-clock / git_sha / absolute paths). Fails
    loud when no game dir is usable. The return is the JSON document boundary, so it is
    typed `dict[str, Any]` rather than a domain model.
    """
    games = _load_games(run_dirs)
    if not games:
        raise ValueError(f"no valid game directories found under {[str(d) for d in run_dirs]}")

    leaderboard = rate_games(to_game_results(games))
    aggregate = aggregate_metrics(games)
    split = deceiver_detector_split(games)
    h2h = head_to_head(games)
    selfplay = self_play_stats(games)
    seat_models = sorted({s.model for g in games for s in g.seats if s.model != _UNKNOWN_MODEL})

    return {
        "meta": {
            "n_games": len(games),
            "n_rated": leaderboard.n_games,
            "n_skipped": leaderboard.n_skipped,
            "models": seat_models,
            "run_dirs": sorted({run_dir.name for run_dir in run_dirs}),
        },
        "leaderboard": [
            {
                "model": r.model,
                "skill": r.skill,
                "mu": r.mu,
                "sigma": r.sigma,
                "games": r.games,
                "wins": r.wins,
                "losses": r.losses,
            }
            for r in leaderboard.ratings
        ],
        "deceiver_detector": {
            "wolf_win_rate": split.wolf_win_rate,
            "exile_accuracy": split.exile_accuracy,
            "total_exiles": split.total_exiles,
            "total_correct_exiles": split.total_correct_exiles,
            "models": [
                {
                    "model": f.model,
                    "wolf_games": f.wolf_games,
                    "wolf_wins": f.wolf_wins,
                    "wolf_win_rate": f.wolf_win_rate,
                    "village_exiles": f.village_exiles,
                    "village_correct_exiles": f.village_correct_exiles,
                    "exile_accuracy": f.exile_accuracy,
                }
                for f in split.models
            ],
        },
        "head_to_head": {
            "models": list(h2h.models),
            "cells": [
                {"row": c.row, "col": c.col, "row_wins": c.row_wins, "col_wins": c.col_wins, "games": c.games}
                for c in h2h.cells
            ],
        },
        "cost_efficiency": _cost_efficiency(games, aggregate),
        "self_play": [
            {
                "model": s.model,
                "games": s.games,
                "wolf_win_rate": s.wolf_win_rate,
                "exile_accuracy": s.exile_accuracy,
                "mean_rounds": s.mean_rounds,
            }
            for s in selfplay
        ],
        "games": [
            {
                "game_id": g.game_id,
                "seed": g.seed,
                "winner": g.winner,
                "rounds": g.rounds,
                "wolf_model": _unique_faction_model(g.seats, _WOLF),
                "village_model": _unique_faction_model(g.seats, _VILLAGE),
            }
            for g in sorted(games, key=lambda g: g.game_id)
        ],
    }
