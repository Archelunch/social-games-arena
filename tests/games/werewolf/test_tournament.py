"""Tests for the Werewolf tournament runner (T27).

The tournament runner turns a population of models into many seeded cross-play
games and aggregates the outcomes into a TrueSkill leaderboard. Two layers:

- `schedule_tournament` is the deterministic scheduler — pairwise over every model
  pair (INCLUDING self-pairs A-vs-A, a self-play diagnostic), `games_per_pair`
  games each, side-swapped so each model plays wolves in half a distinct pair's
  games (controls the known wolf-win bias). Per-game seed + roster derive from the
  master seed, so the whole schedule replays byte-identically (invariant #4).
- `run_tournament` drives each matchup through an injected `GameRunner` (a real
  game in production, a fake here), reassembles results in schedule order even when
  run concurrently, and aggregates via `to_game_results` → `rate_games` plus the
  metric rollups.

Self-play note: an A-vs-A game has the same model on both factions, so `rate_games`
skips it (no cross-model signal → `Leaderboard.n_skipped`); it still feeds the
metric rollups. These tests pin that split so it is not mistaken for a bug.
"""

from __future__ import annotations

import json
from itertools import combinations_with_replacement

import pytest

from social_deduction_bench.cli import _DEFAULT_NAMES
from social_deduction_bench.engine import GameRNG
from social_deduction_bench.games.werewolf.assignment import assign_default_roles
from social_deduction_bench.games.werewolf.metrics import GameMetrics, SeatMetrics
from social_deduction_bench.games.werewolf.roles import Faction, Role, faction_of
from social_deduction_bench.games.werewolf.tournament import (
    Matchup,
    TournamentResult,
    run_tournament,
    schedule_tournament,
    tournament_summary_dict,
    write_tournament_summary,
)

_WOLVES = Faction.WEREWOLVES.value
_VILLAGERS = Faction.VILLAGERS.value


# --- fake game runner ----------------------------------------------------


def _metrics_for(matchup: Matchup, *, winner: str) -> GameMetrics:
    """Build a GameMetrics for `matchup` with the given winning faction.

    Token/tool fields are zeroed: these tests exercise scheduling, rating, and
    aggregation wiring, not telemetry extraction (covered by test_metrics.py).
    """
    seat_model = dict(matchup.seat_models)
    seats = tuple(
        SeatMetrics(
            name=name,
            role=role,
            model=seat_model[name],
            faction=faction_of(role).value,
            won=faction_of(role).value == winner,
            survived=faction_of(role).value == winner,
            prompt_tokens=0,
            completion_tokens=0,
            lm_calls=0,
            tool_calls=0,
            illegal_moves=0,
            cost_usd=None,
        )
        for name, role in matchup.roster
    )
    return GameMetrics(
        game_id=matchup.game_id,
        seed=matchup.seed,
        winner=winner,
        rounds=1,
        n_players=len(matchup.roster),
        seats=seats,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_lm_calls=0,
        total_tool_calls=0,
        total_illegal_moves=0,
        illegal_move_rate=0.0,
        total_cost_usd=None,
        tool_usage=(),
        exiles_total=0,
        exiles_correct=0,
        exile_accuracy=None,
    )


def _champion_runner(champion: str):
    """A runner where `champion` always wins, whichever faction it is seated on."""

    def run(matchup: Matchup) -> GameMetrics:
        winner = _WOLVES if matchup.werewolf_model == champion else _VILLAGERS
        return _metrics_for(matchup, winner=winner)

    return run


# --- scheduling: coverage ------------------------------------------------


def test_schedule_includes_self_pairs_and_every_distinct_pair() -> None:
    # The user wants A-vs-A games (self-play diagnostic) AND every cross pair. The
    # schedule is combinations-with-replacement over the sorted model set, so its
    # size is exactly that count times games_per_pair.
    matchups = schedule_tournament(["A", "B", "C"], games_per_pair=2, seed=1, names=_DEFAULT_NAMES)
    n_pairs = len(list(combinations_with_replacement(["A", "B", "C"], 2)))  # 6
    assert len(matchups) == n_pairs * 2

    self_pairs = {(m.werewolf_model, m.villager_model) for m in matchups if m.werewolf_model == m.villager_model}
    assert self_pairs == {("A", "A"), ("B", "B"), ("C", "C")}

    cross = [m for m in matchups if m.werewolf_model != m.villager_model]
    distinct = {frozenset((m.werewolf_model, m.villager_model)) for m in cross}
    assert distinct == {frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"B", "C"})}


def test_distinct_pair_side_swaps_and_self_pair_keeps_one_model() -> None:
    # A distinct pair must put each model on wolves in exactly half its games — the
    # bias control. A self-pair has the same model on both sides every game.
    matchups = schedule_tournament(["A", "B"], games_per_pair=4, seed=7, names=_DEFAULT_NAMES)

    ab = [m for m in matchups if {m.werewolf_model, m.villager_model} == {"A", "B"}]
    assert len(ab) == 4
    assert sum(1 for m in ab if m.werewolf_model == "A") == 2
    assert sum(1 for m in ab if m.werewolf_model == "B") == 2

    aa = [m for m in matchups if m.werewolf_model == "A" and m.villager_model == "A"]
    assert len(aa) == 4
    assert all(m.villager_model == "A" for m in aa)


def test_single_model_schedules_pure_self_play() -> None:
    # One model is a valid (degenerate) population: only the self-pair, all games
    # A-vs-A. Pure self-play, which rate_games will later skip.
    matchups = schedule_tournament(["solo"], games_per_pair=3, seed=1, names=_DEFAULT_NAMES)
    assert len(matchups) == 3
    assert all(m.werewolf_model == "solo" and m.villager_model == "solo" for m in matchups)


# --- scheduling: determinism & replay (invariant #4) ---------------------


def test_schedule_is_deterministic() -> None:
    a = schedule_tournament(["A", "B", "C"], games_per_pair=3, seed=99, names=_DEFAULT_NAMES)
    b = schedule_tournament(["A", "B", "C"], games_per_pair=3, seed=99, names=_DEFAULT_NAMES)
    assert a == b


def test_each_matchup_roster_replays_from_its_seed() -> None:
    # The stored roster must equal the seeded deal from the matchup's own seed, so a
    # recorded game reconstructs the same board (invariant #4). And the seat-model
    # map must put the werewolf model on exactly the werewolf seats.
    matchups = schedule_tournament(["A", "B"], games_per_pair=2, seed=5, names=_DEFAULT_NAMES)
    for m in matchups:
        assert m.roster == assign_default_roles(_DEFAULT_NAMES, GameRNG(m.seed))
        seat_model = dict(m.seat_models)
        for name, role in m.roster:
            expected = m.werewolf_model if role == Role.WEREWOLF.value else m.villager_model
            assert seat_model[name] == expected


def test_schedule_seeds_differ_across_games() -> None:
    # Each scheduled game gets a distinct seed drawn from the master RNG, so the
    # games are different boards, not the same game replayed N times.
    matchups = schedule_tournament(["A", "B", "C"], games_per_pair=3, seed=3, names=_DEFAULT_NAMES)
    seeds = [m.seed for m in matchups]
    assert len(set(seeds)) == len(seeds)


# --- scheduling: fail loud -----------------------------------------------


def test_schedule_fails_loud_on_no_models() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        schedule_tournament([], games_per_pair=2, seed=1, names=_DEFAULT_NAMES)


def test_schedule_fails_loud_on_nonpositive_games_per_pair() -> None:
    with pytest.raises(ValueError, match="games_per_pair"):
        schedule_tournament(["A", "B"], games_per_pair=0, seed=1, names=_DEFAULT_NAMES)


# --- run_tournament: aggregation -----------------------------------------


def test_run_tournament_rates_all_games_and_champion_tops_board() -> None:
    matchups = schedule_tournament(["A", "B", "C"], games_per_pair=2, seed=1, names=_DEFAULT_NAMES)
    result = run_tournament(matchups, _champion_runner("A"))

    assert len(result.games) == len(matchups)
    assert {r.model for r in result.leaderboard.ratings} == {"A", "B", "C"}
    assert result.leaderboard.ratings[0].model == "A"  # the model that always won


def test_self_play_games_are_rate_skipped_but_kept_in_metrics() -> None:
    # 3 self-pairs (AA/BB/CC) at games_per_pair=2 → 6 self-play games. rate_games
    # skips them (overlap, no cross-model signal) but the metric rollups still see
    # them, so a model's self-play shows up in aggregate/split.
    matchups = schedule_tournament(["A", "B", "C"], games_per_pair=2, seed=1, names=_DEFAULT_NAMES)
    result = run_tournament(matchups, _champion_runner("A"))

    assert result.leaderboard.n_skipped == 3 * 2
    assert result.leaderboard.n_games == len(matchups) - 3 * 2
    assert result.aggregate.n_games == len(matchups)  # aggregate counts all games
    assert {m.model for m in result.aggregate.models} == {"A", "B", "C"}
    assert result.split.n_games == len(matchups)


def test_run_tournament_calls_runner_once_per_matchup_in_order() -> None:
    matchups = schedule_tournament(["A", "B"], games_per_pair=2, seed=1, names=_DEFAULT_NAMES)
    seen: list[str] = []

    def run(matchup: Matchup) -> GameMetrics:
        seen.append(matchup.game_id)
        return _metrics_for(matchup, winner=_WOLVES)

    result = run_tournament(matchups, run)
    assert len(seen) == len(matchups)
    assert [g.game_id for g in result.games] == [m.game_id for m in matchups]


def test_run_tournament_is_deterministic_under_concurrency() -> None:
    # Invariant #4 for the aggregation step: running games in parallel must not
    # change the leaderboard. Results are reassembled in schedule order regardless
    # of completion order, so concurrency 4 equals concurrency 1.
    matchups = schedule_tournament(["A", "B", "C"], games_per_pair=2, seed=1, names=_DEFAULT_NAMES)
    sequential = run_tournament(matchups, _champion_runner("A"), max_concurrency=1)
    parallel = run_tournament(matchups, _champion_runner("A"), max_concurrency=4)
    assert sequential == parallel


def test_run_tournament_over_empty_schedule_is_safe() -> None:
    result = run_tournament((), _champion_runner("A"))
    assert isinstance(result, TournamentResult)
    assert result.games == ()
    assert result.leaderboard.n_games == 0
    assert result.leaderboard.ratings == ()


# --- summary persistence -------------------------------------------------


def test_write_tournament_summary_is_deterministic_and_complete(tmp_path) -> None:
    matchups = schedule_tournament(["A", "B", "C"], games_per_pair=2, seed=1, names=_DEFAULT_NAMES)
    result = run_tournament(matchups, _champion_runner("A"))

    path = tmp_path / "summary.json"
    write_tournament_summary(result, path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["n_games"] == result.leaderboard.n_games
    assert data["n_skipped"] == result.leaderboard.n_skipped
    assert {row["model"] for row in data["leaderboard"]} == {"A", "B", "C"}
    assert "aggregate" in data
    assert "split" in data
    assert len(data["games"]) == len(matchups)
    for row in data["games"]:
        assert {"game_id", "seed", "winner", "werewolf_model", "villager_model"} <= set(row)

    # Same result → byte-identical summary (no wall-clock in the summary).
    again = tmp_path / "again.json"
    write_tournament_summary(result, again)
    assert path.read_text(encoding="utf-8") == again.read_text(encoding="utf-8")


def test_tournament_summary_dict_matches_written_file(tmp_path) -> None:
    # The dict builder and the file writer must agree (the writer is just JSON of
    # the dict), so a caller can serialize the summary however they like.
    matchups = schedule_tournament(["A", "B"], games_per_pair=2, seed=2, names=_DEFAULT_NAMES)
    result = run_tournament(matchups, _champion_runner("A"))
    path = tmp_path / "summary.json"
    write_tournament_summary(result, path)
    assert json.loads(path.read_text(encoding="utf-8")) == tournament_summary_dict(result)
