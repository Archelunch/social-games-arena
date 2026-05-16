"""Tests for the seeded engine RNG (T02).

These encode benchmark invariant #4: everything stochastic derives from the
engine seed, so the same seed yields an identical, replayable game. They also
guard the CLAUDE.md determinism rule that no game logic may touch global
`random` state. If `GameRNG` ever leaks into or out of global RNG state, or
stops being seed-deterministic, these fail loudly.
"""

import random

import pytest

from social_deduction_bench.engine import GameRNG


def test_game_rng_same_seed_produces_identical_sequence() -> None:
    """Same seed -> identical draws. This is the replay guarantee itself."""
    deck = list(range(20))
    rng_a = GameRNG(seed=42)
    rng_b = GameRNG(seed=42)

    assert rng_a.shuffle(deck) == rng_b.shuffle(deck)
    assert rng_a.choice(deck) == rng_b.choice(deck)
    assert rng_a.sample(deck, 5) == rng_b.sample(deck, 5)


def test_game_rng_different_seed_diverges() -> None:
    """A different seed must change outcomes, or the seed is a no-op."""
    deck = list(range(20))
    shuffled_1 = GameRNG(seed=1).shuffle(deck)
    shuffled_2 = GameRNG(seed=2).shuffle(deck)

    assert shuffled_1 != shuffled_2
    # Both must still be genuine reorderings of the deck — guards against a
    # degenerate "returns the input unchanged" shuffle that diverges only
    # because the inputs differ.
    assert sorted(shuffled_1) == deck
    assert sorted(shuffled_2) == deck


def test_game_rng_shuffle_does_not_mutate_input() -> None:
    """shuffle returns a new list; engine state stays immutable."""
    deck = list(range(20))
    original = deck.copy()

    result = GameRNG(seed=7).shuffle(deck)

    assert deck == original
    assert result is not deck
    assert sorted(result) == original


def test_game_rng_does_not_touch_global_random_state() -> None:
    """GameRNG must never advance the global `random` stream.

    Game logic that read or mutated global RNG state would be nondeterministic
    across processes and unreplayable — the CLAUDE.md determinism rule.
    """
    random.seed(123)
    state_before = random.getstate()
    try:
        rng = GameRNG(seed=999)
        rng.shuffle(list(range(20)))
        rng.choice(list(range(20)))
        rng.sample(list(range(20)), 5)

        assert random.getstate() == state_before
    finally:
        random.setstate(state_before)


def test_game_rng_instances_are_independent() -> None:
    """Two RNGs own separate streams; one's draws never consume another's.

    Interleaving calls on a second instance must not shift the first
    instance's sequence — otherwise game subsystems sharing the engine seed
    would corrupt each other's determinism.
    """
    deck = list(range(20))

    rng_a = GameRNG(seed=7)
    rng_b = GameRNG(seed=7)
    first = rng_a.shuffle(deck)
    rng_b.shuffle(deck)  # interleaved draw on the other instance
    second = rng_a.shuffle(deck)

    reference = GameRNG(seed=7)
    assert reference.shuffle(deck) == first
    assert reference.shuffle(deck) == second


def test_game_rng_exposes_seed() -> None:
    """The seed is recoverable for the event log and replay tooling."""
    assert GameRNG(seed=2026).seed == 2026


def test_game_rng_seed_is_read_only() -> None:
    """The seed must not be reassignable.

    A mutated seed would silently desync replay from the recorded event
    stream — the rebound seed no longer describes the stream in use.
    """
    rng = GameRNG(seed=2026)
    with pytest.raises(AttributeError):
        rng.seed = 99  # type: ignore  # assigning to a read-only property is the point


def test_game_rng_matches_golden_sequence_for_known_seed() -> None:
    """Pin the exact draws for seed 42 against golden literals.

    The same-seed test only proves two instances agree with each other; it
    would not catch a change in the underlying draw algorithm. A replay
    recorded today must still reproduce on a future interpreter, so the
    concrete sequence is frozen here.
    """
    rng = GameRNG(seed=42)

    assert rng.shuffle(list(range(10))) == [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
    assert rng.choice(list(range(10))) == 9
    assert rng.sample(list(range(10)), 3) == [6, 0, 8]
