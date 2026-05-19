"""Tests for seeded Werewolf role assignment (T10).

These encode benchmark invariant #4: role dealing is deterministic — the same
seed yields the same name->role mapping, so a recorded game replays identically
from turn one. The golden-literal test pins the exact seed-42 assignment so a
future RNG-algorithm or shuffle-order change fails loud rather than silently
forking every recorded transcript (mirrors `tests/engine/test_rng.py`). The
count-mismatch test guards a silent-truncation bug that would drop a werewolf
and make the game unwinnable.
"""

import random

import pytest

from social_deduction_bench.engine import GameRNG, GameState
from social_deduction_bench.games.werewolf.assignment import assign_default_roles, assign_roles
from social_deduction_bench.games.werewolf.config import default_role_multiset

NAMES = ["Alice", "Bob", "Cara", "Dan", "Eve", "Finn", "Gwen"]


def test_same_seed_yields_identical_assignment() -> None:
    """Two `GameRNG` streams with the same seed deal the identical mapping.

    This *is* invariant #4 for role dealing — without it a replayed game would
    diverge from its recorded transcript before the first night.
    """
    a = assign_roles(NAMES, default_role_multiset(), GameRNG(99))
    b = assign_roles(NAMES, default_role_multiset(), GameRNG(99))

    assert a == b


def test_different_seeds_diverge() -> None:
    """Different seeds produce different assignments — the seed is not a no-op.

    Seeds 42 and 123 are known to shuffle the 7-role multiset differently. If
    assignment ignored the seed, cross-play variety (and the leaderboard's
    statistical spread) would collapse.
    """
    a = assign_roles(NAMES, default_role_multiset(), GameRNG(42))
    b = assign_roles(NAMES, default_role_multiset(), GameRNG(123))

    assert a != b


def test_seed_42_golden_assignment() -> None:
    """The exact seed-42 assignment is pinned as a golden literal.

    A change to the RNG algorithm or the shuffle order would silently break
    replay of every recorded game. Pinning the literal turns that into a loud
    test failure (same guard as `test_rng.py`'s golden sequence).
    """
    assigned = assign_roles(NAMES, default_role_multiset(), GameRNG(42))

    assert assigned == (
        ("Alice", "werewolf"),
        ("Bob", "doctor"),
        ("Cara", "villager"),
        ("Dan", "seer"),
        ("Eve", "villager"),
        ("Finn", "werewolf"),
        ("Gwen", "villager"),
    )


def test_dealt_roles_are_exactly_the_role_multiset() -> None:
    """Every declared role is placed exactly once — none invented or dropped.

    The dealt roles, sorted, must equal the input multiset sorted. A dealt set
    missing a werewolf or with a phantom seer would corrupt win conditions.
    """
    multiset = default_role_multiset()
    assigned = assign_roles(NAMES, multiset, GameRNG(7))

    assert sorted(role for _, role in assigned) == sorted(multiset)


def test_every_name_appears_exactly_once() -> None:
    """Each input player gets exactly one seat — no name dropped or duplicated.

    A duplicated name would misroute private events (invariant #2); a dropped
    name would leave a seat unfilled.
    """
    assigned = assign_roles(NAMES, default_role_multiset(), GameRNG(7))

    assert sorted(name for name, _ in assigned) == sorted(NAMES)


def test_count_mismatch_fewer_names_than_roles_is_rejected() -> None:
    """Fewer names than roles fails loud instead of silently truncating.

    `zip` would otherwise drop the extra roles — silently undealing a role
    (e.g. a werewolf), which makes the game unwinnable. This must raise.
    """
    with pytest.raises(ValueError, match="player"):
        assign_roles(NAMES[:5], default_role_multiset(), GameRNG(1))


def test_count_mismatch_more_names_than_roles_is_rejected() -> None:
    """More names than roles fails loud too — the mismatch is symmetric.

    `zip` would otherwise leave the trailing players unseated; both directions
    of a count mismatch must raise, not silently produce a short roster.
    """
    with pytest.raises(ValueError, match="player"):
        assign_roles(NAMES, default_role_multiset()[:5], GameRNG(1))


def test_unknown_role_string_is_rejected() -> None:
    """A role string that is not a real `Role` fails loud at deal time.

    A typo'd role would otherwise be dealt as a phantom role and only surface
    far later as a corrupt win-condition count (T13). Catch it here.
    """
    bad_multiset = ("werewolf", "werewolf", "seer", "doctor", "villager", "villager", "vampire")
    with pytest.raises(ValueError, match="unknown role"):
        assign_roles(NAMES, bad_multiset, GameRNG(1))


def test_assignment_does_not_touch_global_random() -> None:
    """Assignment draws only from the passed `GameRNG`, never global `random`.

    Mirrors `test_rng.py`: leaking into global `random` state would make the
    result depend on unrelated code and break determinism (invariant #4).
    """
    random.seed(12345)
    before = random.random()
    random.seed(12345)
    assign_roles(NAMES, default_role_multiset(), GameRNG(42))
    after = random.random()

    assert before == after


def test_input_name_list_is_not_mutated() -> None:
    """The caller's name list is left untouched.

    Assignment is a pure derivation; mutating the caller's input would be an
    invisible side effect that could corrupt a later call.
    """
    names = list(NAMES)
    assign_roles(names, default_role_multiset(), GameRNG(42))

    assert names == NAMES


def test_assignment_output_constructs_a_game_state() -> None:
    """The `(name, role)` output is exactly the shape `GameState.initial` accepts.

    Pins the contract with the engine (T03): assignment feeds straight into the
    game state with no adaptation, and the unique input names pass
    `GameState.initial`'s duplicate-name check.
    """
    assigned = assign_roles(NAMES, default_role_multiset(), GameRNG(42))
    state = GameState.initial(assigned)

    assert len(state.players) == len(NAMES)


def test_assign_default_roles_matches_explicit_default_multiset() -> None:
    """`assign_default_roles` is `assign_roles` with the default 7-player multiset.

    The convenience wrapper must not diverge from the explicit call — same
    seed, same result.
    """
    convenience = assign_default_roles(NAMES, GameRNG(42))
    explicit = assign_roles(NAMES, default_role_multiset(), GameRNG(42))

    assert convenience == explicit
