"""Tests for the game-agnostic TrueSkill rating core (T26).

`rate_games` turns a sequence of `GameResult`s (each: model-name teams + the
winning team index) into a per-model `Leaderboard`. Werewolf is a team game with
asymmetric roles, so TrueSkill updates *individual* model ratings from *team*
outcomes (WEREWOLF_DESIGN.md §10). This module is game-agnostic — it imports only
`trueskill`, never the engine or any game — so ONUW / Secret Hitler reuse it.

Determinism (invariant #4): `trueskill.rate` is pure analytic math with no RNG, so
the only thing that affects the result is the order of games; identical input →
identical `Leaderboard`. These tests pin update direction, win/loss accounting, the
leaderboard sort (skill desc, then model name), self-play/degenerate skipping,
intra-team duplicate collapse, fail-loud construction, and determinism.
"""

from __future__ import annotations

import pytest
import trueskill

from social_deduction_bench.rating.trueskill import GameResult, Leaderboard, rate_games

_DEFAULT_SIGMA = 25.0 / 3.0  # TrueSkill default starting sigma (8.3333...)


# --- core update direction -----------------------------------------------


def test_winner_rating_rises_and_loser_rating_drops() -> None:
    # A beats B once. The winner's mean must rise above the 25.0 start and the
    # loser's fall below it; both grow more certain (sigma shrinks). This pins the
    # fundamental TrueSkill update direction the leaderboard depends on.
    board = rate_games([GameResult(teams=(("A",), ("B",)), winner=0)])
    by_model = {m.model: m for m in board.ratings}
    a, b = by_model["A"], by_model["B"]
    assert a.mu > 25.0 > b.mu
    assert a.skill > b.skill
    assert a.sigma < _DEFAULT_SIGMA
    assert b.sigma < _DEFAULT_SIGMA
    assert board.n_games == 1
    assert board.n_skipped == 0


def test_games_wins_losses_accumulate_per_model() -> None:
    # A beats B three times. Counts are per contributing game, attributed by the
    # winning-team membership; losses are games - wins.
    results = [GameResult(teams=(("A",), ("B",)), winner=0) for _ in range(3)]
    board = rate_games(results)
    by_model = {m.model: m for m in board.ratings}
    assert (by_model["A"].games, by_model["A"].wins, by_model["A"].losses) == (3, 3, 0)
    assert (by_model["B"].games, by_model["B"].wins, by_model["B"].losses) == (3, 0, 3)
    assert board.n_games == 3
    assert board.n_skipped == 0


# --- leaderboard ordering ------------------------------------------------


def test_leaderboard_sorted_by_skill_desc_then_model_name() -> None:
    # Two independent matchups against fresh opponents: A beats B, C beats D. A and
    # C have identical histories (won vs a fresh rating) → identical skill; likewise
    # B and D (lost vs a fresh rating). The sort is (skill desc, then model name), so
    # among the tied winners A precedes C and among the tied losers B precedes D.
    board = rate_games(
        [
            GameResult(teams=(("A",), ("B",)), winner=0),
            GameResult(teams=(("C",), ("D",)), winner=0),
        ]
    )
    assert [m.model for m in board.ratings] == ["A", "C", "B", "D"]
    by_model = {m.model: m for m in board.ratings}
    assert by_model["A"].skill == pytest.approx(by_model["C"].skill)
    assert by_model["B"].skill == pytest.approx(by_model["D"].skill)
    assert by_model["A"].skill > by_model["B"].skill


# --- determinism (invariant #4) ------------------------------------------


def test_rate_games_is_deterministic() -> None:
    # Same sequence twice → byte-identical leaderboard (frozen dataclasses compare by
    # value, mu/sigma included). TrueSkill has no RNG; the result is a pure function
    # of the input order.
    results = [
        GameResult(teams=(("A",), ("B",)), winner=0),
        GameResult(teams=(("B",), ("A",)), winner=0),
        GameResult(teams=(("A",), ("B",)), winner=1),
    ]
    assert rate_games(results) == rate_games(results)


# --- degenerate / self-play games skipped --------------------------------


def test_pure_self_play_game_is_skipped_not_rated() -> None:
    # A model facing itself yields no cross-model signal (it can't be both rating
    # groups). Skip it: counted in n_skipped, contributes nothing to ratings.
    board = rate_games([GameResult(teams=(("A",), ("A",)), winner=0)])
    assert board.n_skipped == 1
    assert board.n_games == 0
    assert board.ratings == ()


def test_overlap_game_skipped_while_clean_game_is_rated() -> None:
    # A model on more than one team in the same game (overlap) is skipped; a clean
    # cross-model game in the same batch still rates normally.
    board = rate_games(
        [
            GameResult(teams=(("A",), ("A",)), winner=0),  # degenerate
            GameResult(teams=(("A",), ("B",)), winner=0),  # clean
        ]
    )
    assert board.n_skipped == 1
    assert board.n_games == 1
    by_model = {m.model: m for m in board.ratings}
    assert by_model["A"].games == 1  # only the clean game
    assert by_model["B"].games == 1


def test_duplicate_model_within_a_team_is_collapsed_to_one_entity() -> None:
    # A model occupying two seats on the same team is one rated entity, updated once
    # (not twice). Guards against double-counting a faction's shared model.
    board = rate_games([GameResult(teams=(("A", "A"), ("B",)), winner=0)])
    by_model = {m.model: m for m in board.ratings}
    assert by_model["A"].games == 1
    assert by_model["A"].wins == 1
    assert by_model["A"].mu > 25.0
    assert by_model["B"].games == 1


# --- fail-loud construction & empty input --------------------------------


def test_game_result_construction_is_fail_loud() -> None:
    # A malformed result is a caller bug, not a skippable degeneracy: fail loud at
    # construction (CLAUDE.md rule 11) rather than silently mis-rating.
    with pytest.raises(ValueError, match="at least 2 teams"):
        GameResult(teams=(("A",),), winner=0)  # fewer than 2 teams
    with pytest.raises(ValueError, match="at least one model"):
        GameResult(teams=(("A",), ()), winner=0)  # an empty team
    with pytest.raises(ValueError, match="out of range"):
        GameResult(teams=(("A",), ("B",)), winner=2)  # winner index out of range
    with pytest.raises(ValueError, match="out of range"):
        GameResult(teams=(("A",), ("B",)), winner=-1)  # negative winner index


def test_rate_games_over_no_results_is_safe() -> None:
    board = rate_games([])
    assert board == Leaderboard(n_games=0, n_skipped=0, ratings=())


# --- game-agnostic generality: >2 teams, ordering, env override ----------


def test_three_team_game_ranks_winner_above_equal_losers() -> None:
    # The core is game-agnostic and accepts >2 teams (GameResult allows >=2). A
    # 3-team game ranks the winner first and every loser equal (rank 1). Werewolf is
    # 2-team, but ONUW / Secret Hitler may not be — this pins the advertised generality.
    board = rate_games([GameResult(teams=(("A",), ("B",), ("C",)), winner=1)])
    by_model = {m.model: m for m in board.ratings}
    assert by_model["B"].wins == 1
    assert by_model["B"].mu > 25.0
    assert (by_model["A"].wins, by_model["A"].losses) == (0, 1)
    assert (by_model["C"].wins, by_model["C"].losses) == (0, 1)
    # Both losers share a fresh history and the same losing rank → identical rating.
    assert by_model["A"].mu == pytest.approx(by_model["C"].mu)
    assert by_model["B"].skill > by_model["A"].skill
    assert board.n_games == 1


def test_game_order_changes_result_so_games_are_not_re_sorted() -> None:
    # TrueSkill is sequential/online: "A beats B then B beats C" differs from the
    # reverse, because B's rating when it meets C depends on whether it already lost
    # to A. The caller owns ordering, so rate_games must NOT re-sort the batch — a
    # reordered batch must therefore yield a different leaderboard. Guards against an
    # accidental internal sort that the same-order determinism test cannot catch.
    forward = rate_games([GameResult((("A",), ("B",)), 0), GameResult((("B",), ("C",)), 0)])
    reordered = rate_games([GameResult((("B",), ("C",)), 0), GameResult((("A",), ("B",)), 0)])
    assert forward != reordered


def test_custom_env_overrides_trueskill_defaults() -> None:
    # The env param lets a caller supply a non-default TrueSkill environment; the
    # ratings must derive from it (here a mu=100 start), not the built-in mu=25.
    env = trueskill.TrueSkill(mu=100.0, draw_probability=0.0)
    board = rate_games([GameResult((("A",), ("B",)), 0)], env=env)
    by_model = {m.model: m for m in board.ratings}
    assert by_model["A"].mu > 100.0  # winner rises above the custom start, not 25.0
    assert by_model["B"].mu < 100.0
