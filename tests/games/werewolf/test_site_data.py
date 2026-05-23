"""Tests for the static-site data layer (T28).

`site_data.build_site_data` re-aggregates a tournament run's per-game artifacts into
the deterministic `data.json` payload the static site renders. Two pure helpers,
`head_to_head` (who-beat-whom across distinct models) and `self_play_stats` (a model
against an identical copy of itself), are the only NEW aggregations; the rest reuse
the existing metric/rating functions.

The benchmark invariants drive the contract:
- The payload must be deterministic (invariant #4): same set of completed game dirs
  -> byte-identical JSON. So it must never embed wall-clock / git_sha / absolute paths.
- A torn or in-progress game dir (no `GAME_OVER`) must be skipped, not crash the build,
  because the run directory is written live by a parallel sweep.
- Faction attribution reuses `_unique_faction_model`, so a mixed-faction or
  unknown-model faction is unattributable and excluded (never a bogus row).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from social_deduction_bench.games.werewolf.metrics import GameMetrics, SeatMetrics
from social_deduction_bench.games.werewolf.roles import Role, faction_of
from social_deduction_bench.games.werewolf.site_data import (
    HeadToHeadCell,
    HeadToHeadMatrix,
    SelfPlayStat,
    build_site_data,
    head_to_head,
    self_play_stats,
)

# `write_game_dir` / `two_game_run` are auto-injected fixtures from tests/conftest.py.
GameDirWriter = Callable[..., Path]

_WOLVES = "werewolves"
_VILLAGERS = "villagers"


# --- in-memory GameMetrics factory (pure-helper tests, no disk) ----------


def _seat(name: str, role: str, model: str, winner: str) -> SeatMetrics:
    faction = faction_of(role).value
    return SeatMetrics(
        name=name,
        role=role,
        model=model,
        faction=faction,
        won=faction == winner,
        survived=True,
        prompt_tokens=10,
        completion_tokens=5,
        lm_calls=1,
        tool_calls=2,
        illegal_moves=0,
        cost_usd=None,
    )


def _game(
    game_id: str,
    seats_spec: list[tuple[str, str, str]],
    winner: str,
    *,
    rounds: int = 2,
    exiles_total: int = 0,
    exiles_correct: int = 0,
    seed: int = 1,
) -> GameMetrics:
    seats = tuple(_seat(n, r, m, winner) for (n, r, m) in seats_spec)
    return GameMetrics(
        game_id=game_id,
        seed=seed,
        winner=winner,
        rounds=rounds,
        n_players=len(seats),
        seats=seats,
        total_prompt_tokens=sum(s.prompt_tokens for s in seats),
        total_completion_tokens=sum(s.completion_tokens for s in seats),
        total_lm_calls=sum(s.lm_calls for s in seats),
        total_tool_calls=sum(s.tool_calls for s in seats),
        total_illegal_moves=0,
        illegal_move_rate=0.0,
        total_cost_usd=None,
        tool_usage=(),
        exiles_total=exiles_total,
        exiles_correct=exiles_correct,
        exile_accuracy=(exiles_correct / exiles_total if exiles_total else None),
    )


def _cross_game(game_id: str, wolf_model: str, village_model: str, winner: str, **kw: object) -> GameMetrics:
    """A standard 5-seat game: 2 wolves of one model vs seer/doctor/villager of another."""
    spec = [
        ("Alice", Role.WEREWOLF.value, wolf_model),
        ("Bob", Role.WEREWOLF.value, wolf_model),
        ("Carol", Role.SEER.value, village_model),
        ("Dave", Role.DOCTOR.value, village_model),
        ("Eve", Role.VILLAGER.value, village_model),
    ]
    return _game(game_id, spec, winner, **kw)  # type: ignore[arg-type]


# --- head_to_head --------------------------------------------------------


def test_head_to_head_three_game_fixture_directional() -> None:
    """Direction matters: the winning faction's model beat the other's, summed per pair."""
    games = [
        _cross_game("g1", "A", "B", _WOLVES),  # A (wolves) beat B
        _cross_game("g2", "B", "A", _WOLVES),  # B (wolves) beat A
        _cross_game("g3", "A", "B", _VILLAGERS),  # B (village) beat A
    ]
    matrix = head_to_head(games)
    assert isinstance(matrix, HeadToHeadMatrix)
    assert matrix.models == ("A", "B")
    assert matrix.cells == (HeadToHeadCell(row="A", col="B", row_wins=1, col_wins=2, games=3),)


def test_head_to_head_excludes_self_pair_and_mixed_faction() -> None:
    """Self-pairs (same model both factions) and mixed-faction games are unattributable."""
    mixed = _game(
        "g_mixed",
        [
            ("Alice", Role.WEREWOLF.value, "A"),
            ("Bob", Role.WEREWOLF.value, "B"),  # two models on the wolf faction
            ("Carol", Role.SEER.value, "C"),
            ("Dave", Role.DOCTOR.value, "C"),
            ("Eve", Role.VILLAGER.value, "C"),
        ],
        _WOLVES,
    )
    games = [
        _cross_game("g_self", "A", "A", _WOLVES),  # self-pair -> excluded
        mixed,  # mixed wolf faction -> excluded
        _cross_game("g_ok", "A", "C", _VILLAGERS),  # C (village) beat A
    ]
    matrix = head_to_head(games)
    assert matrix.models == ("A", "C")
    assert matrix.cells == (HeadToHeadCell(row="A", col="C", row_wins=0, col_wins=1, games=1),)


def test_head_to_head_single_model_empty_cells() -> None:
    """A run with only self-play games yields no cross-pair cells (and no axis models)."""
    games = [_cross_game("g1", "A", "A", _WOLVES), _cross_game("g2", "A", "A", _VILLAGERS)]
    matrix = head_to_head(games)
    assert matrix.models == ()
    assert matrix.cells == ()


# --- self_play_stats -----------------------------------------------------


def test_self_play_stats_counts_rates_and_mean_rounds() -> None:
    """Self-play pools exile decisions and averages rounds; cross-play games are ignored."""
    games = [
        _cross_game("g1", "A", "A", _WOLVES, rounds=2, exiles_total=2, exiles_correct=1),
        _cross_game("g2", "A", "A", _VILLAGERS, rounds=4, exiles_total=2, exiles_correct=2),
        _cross_game("g3", "A", "B", _WOLVES, rounds=9, exiles_total=5, exiles_correct=0),  # not self-play
    ]
    stats = self_play_stats(games)
    assert stats == (SelfPlayStat(model="A", games=2, wolf_win_rate=0.5, exile_accuracy=0.75, mean_rounds=3.0),)


def test_self_play_stats_zero_exiles_yields_none_not_zero() -> None:
    """No resolved exiles is unknown detection (None), never a 0.0 accuracy."""
    games = [_cross_game("g1", "A", "A", _WOLVES, exiles_total=0, exiles_correct=0)]
    (stat,) = self_play_stats(games)
    assert stat.exile_accuracy is None


# --- build_site_data: on-disk read path + determinism --------------------
# The on-disk game-dir writer + `two_game_run` live in tests/conftest.py.


def test_build_site_data_deterministic_byte_identical(two_game_run: Path) -> None:
    """Same set of game dirs -> identical payload, including list element order (invariant #4).

    Serialized WITHOUT `sort_keys`: dict-key order (fixed build order) AND list-element
    order (every list is `sorted()`) must both be stable, so this catches a list whose
    order leaks from set iteration, not just a key-order difference.
    """
    first = json.dumps(build_site_data([two_game_run]))
    second = json.dumps(build_site_data([two_game_run]))
    assert first == second


def test_build_site_data_independent_of_run_dir_order(tmp_path: Path, write_game_dir: GameDirWriter) -> None:
    """The payload is the same regardless of --run argument order (canonical game_id sort).

    Online TrueSkill is order-sensitive, so without a canonical ordering a multi-dir build
    could rank differently per CLI invocation. Two run dirs are built in both orders.
    """
    run_a = tmp_path / "runA"
    run_b = tmp_path / "runB"
    write_game_dir(run_a, "g0000-A-vs-B", wolf_model="A", village_model="B", winner=_WOLVES)
    write_game_dir(run_b, "g0001-B-vs-A", wolf_model="B", village_model="A", winner=_VILLAGERS)
    forward = json.dumps(build_site_data([run_a, run_b]))
    reversed_ = json.dumps(build_site_data([run_b, run_a]))
    assert forward == reversed_


def test_data_json_omits_created_at_and_git_sha(two_game_run: Path) -> None:
    """The public payload must never embed run bookkeeping (would break determinism + leak)."""
    blob = json.dumps(build_site_data([two_game_run]))
    assert "created_at" not in blob
    assert "git_sha" not in blob
    assert "deadbeefcafe" not in blob
    assert "2026-05-23T00:00:00" not in blob


def test_meta_run_dirs_are_basenames_not_abspaths(two_game_run: Path) -> None:
    """run_dirs are basenames so the payload is machine-independent and path-leak-free."""
    data = build_site_data([two_game_run])
    assert data["meta"]["run_dirs"] == ["run2"]
    blob = json.dumps(data)
    assert "/Users" not in blob
    assert str(two_game_run) not in blob


def test_build_site_data_shape_and_leaderboard(two_game_run: Path) -> None:
    """The payload carries every panel's block and a rated, skill-sorted leaderboard."""
    data = build_site_data([two_game_run])
    assert set(data) >= {
        "meta",
        "leaderboard",
        "deceiver_detector",
        "head_to_head",
        "cost_efficiency",
        "self_play",
        "games",
    }
    assert data["meta"]["n_games"] == 2
    assert sorted(data["meta"]["models"]) == ["A", "B"]
    board_models = {row["model"] for row in data["leaderboard"]}
    assert board_models == {"A", "B"}
    skills = [row["skill"] for row in data["leaderboard"]]
    assert skills == sorted(skills, reverse=True)


def test_cost_none_serializes_as_json_null(two_game_run: Path) -> None:
    """Cost is unknown (no LM call records) -> JSON null, never coerced to 0."""
    data = build_site_data([two_game_run])
    assert data["cost_efficiency"]["totals"]["cost_usd"] is None
    assert '"cost_usd": null' in json.dumps(data["cost_efficiency"]["totals"], indent=0)


def test_build_skips_torn_dir_but_uses_valid(tmp_path: Path, write_game_dir: GameDirWriter) -> None:
    """An in-progress dir with no GAME_OVER is skipped; the build proceeds on valid ones."""
    run = tmp_path / "run2"
    write_game_dir(run, "g0000-A-vs-B", wolf_model="A", village_model="B", winner=_WOLVES)
    write_game_dir(run, "g0001-torn", wolf_model="A", village_model="B", winner=_WOLVES, with_game_over=False)
    data = build_site_data([run])
    assert data["meta"]["n_games"] == 1


def test_build_fails_loud_on_zero_valid_dirs(tmp_path: Path, write_game_dir: GameDirWriter) -> None:
    """Zero usable games is a caller error, not an empty site (invariant: fail loud)."""
    run = tmp_path / "run2"
    write_game_dir(run, "g0000-torn", wolf_model="A", village_model="B", winner=_WOLVES, with_game_over=False)
    with pytest.raises(ValueError, match="no valid game"):
        build_site_data([run])
