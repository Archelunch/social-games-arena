"""Tests for the determinism harness (T08).

These encode benchmark invariants #4 (everything seeded -> deterministic ->
replayable) and #5 (every game is an append-only event stream). The harness is
the test utility that actively *enforces* those invariants: it runs a seeded
producer twice and asserts the two transcripts are byte-for-byte identical.

The comparator deliberately does not rely on `EventStream ==`. `EventLog`
defines no `__eq__`, so a frozen `EventStream`'s generated equality compares its
`log` field by identity — two streams with identical content are never `==`.
`assert_streams_identical` compares `header` and `log.events` instead. The
positive comparator test pins that gap with an explicit `a != b` assertion, so
it fails loudly if that assumption ever changes.
"""

import itertools

import pytest

from social_deduction_bench.engine import (
    EventLog,
    EventStream,
    GameRNG,
    Phase,
    StreamHeader,
    assert_deterministic,
    assert_streams_identical,
)

# A canonical roster reused across the harness tests: (name, role) pairs.
ROSTER = (("Ada", "villager"), ("Bo", "werewolf"), ("Cy", "seer"))


def _stream_from_values(values: list[int], *, seed: int = 42, game_id: str = "harness") -> EventStream:
    """Build a transcript with one NIGHT `draw` event per value, `seq` 0, 1, 2, ..."""
    log = EventLog()
    for round_, value in enumerate(values, start=1):
        log.append(round=round_, phase=Phase.NIGHT, type="draw", payload={"value": value})
    header = StreamHeader(seed=seed, game_id=game_id, players=ROSTER)
    return EventStream(header=header, log=log)


def _seeded_producer(seed: int) -> EventStream:
    """A deterministic producer: every event payload derives from `GameRNG(seed)`."""
    rng = GameRNG(seed)
    deck = list(range(100))
    values = [rng.choice(deck) for _ in range(3)]
    return _stream_from_values(values, seed=seed)


def test_assert_deterministic_passes_for_seeded_producer() -> None:
    """A producer whose every value comes from the engine seed runs clean.

    This is the harness's core promise: same seed -> identical event stream.
    If `assert_deterministic` rejected a genuinely deterministic producer, the
    harness would be useless as a gate for T14 / T28.
    """
    assert_deterministic(_seeded_producer, 42)


def test_assert_deterministic_detects_nondeterministic_producer() -> None:
    """A producer that ignores its seed must make the harness raise.

    The critical negative test: a harness that cannot fail when determinism
    breaks (CLAUDE.md rule #8) would silently certify an unreplayable game.
    The counter is scoped to this test so no shared module state leaks across
    the suite — the producer ignores `seed` and emits a fresh value each call,
    so run 2 diverges from run 1.
    """
    counter = itertools.count()

    def nondeterministic_producer(seed: int) -> EventStream:
        return _stream_from_values([next(counter)], seed=seed)

    with pytest.raises(AssertionError):
        assert_deterministic(nondeterministic_producer, 42)


def test_assert_deterministic_passes_producer_the_given_seed() -> None:
    """The harness calls the producer with exactly the given seed, twice.

    If it passed a wrong or varying seed, "same seed -> identical stream" would
    not be what was actually verified.
    """
    seen: list[int] = []

    def recording_producer(seed: int) -> EventStream:
        seen.append(seed)
        return _seeded_producer(seed)

    assert_deterministic(recording_producer, 2026)

    assert seen == [2026, 2026]


def test_assert_streams_identical_passes_for_independently_built_equal_streams() -> None:
    """Two streams with identical content but separate `EventLog`s compare equal.

    `EventLog` has no `__eq__`, so raw `EventStream ==` is identity-only on the
    `log` field — `a != b` here even though the content matches. The comparator
    must look through that and still pass. This pins exactly why a dedicated
    comparator exists rather than `assert actual == expected`.
    """
    a = _seeded_producer(42)
    b = _seeded_producer(42)

    assert a != b  # raw equality is identity-only on `log`; see the docstring

    assert_streams_identical(a, b)


def test_assert_streams_identical_detects_differing_event_payload() -> None:
    """A single divergent event payload is caught, and the message names its seq."""
    actual = _stream_from_values([1, 2, 3])
    expected = _stream_from_values([1, 9, 3])

    with pytest.raises(AssertionError, match="seq 1"):
        assert_streams_identical(actual, expected)


def test_assert_streams_identical_detects_differing_header() -> None:
    """A different header (here, a different seed) is a divergence.

    The seed is what replay reconstructs the RNG from; two streams under
    different seeds are not the same game and must not compare identical.
    """
    actual = _stream_from_values([1, 2], seed=1)
    expected = _stream_from_values([1, 2], seed=2)

    with pytest.raises(AssertionError, match="headers differ"):
        assert_streams_identical(actual, expected)


def test_assert_streams_identical_detects_differing_event_count() -> None:
    """A stream with an extra event must not compare identical to a shorter one.

    A truncated or padded transcript is a determinism break even if every
    shared event matches.
    """
    actual = _stream_from_values([1, 2, 3])
    expected = _stream_from_values([1, 2])

    with pytest.raises(AssertionError, match="event-count mismatch"):
        assert_streams_identical(actual, expected)


def test_assert_streams_identical_detects_recipient_divergence() -> None:
    """Same payload and type but different `recipients` is a divergence.

    Routing (invariant #2) is part of the transcript: a public broadcast and a
    private event with identical payload are not the same recorded event.
    """
    header = StreamHeader(seed=42, game_id="harness", players=ROSTER)

    public_log = EventLog()
    public_log.append(round=1, phase=Phase.NIGHT, type="inspect", payload={"x": 1})
    public_stream = EventStream(header=header, log=public_log)

    private_log = EventLog()
    private_log.append(round=1, phase=Phase.NIGHT, type="inspect", payload={"x": 1}, recipients=("Cy",))
    private_stream = EventStream(header=header, log=private_log)

    with pytest.raises(AssertionError, match="seq 0"):
        assert_streams_identical(public_stream, private_stream)


def test_assert_streams_identical_pinpoints_first_divergent_seq() -> None:
    """When events 0 and 1 match and event 2 differs, the message cites seq 2.

    A harness that only said "streams differ" would be far harder to debug; the
    comparator must report the *first* divergent position, not seq 0.
    """
    actual = _stream_from_values([1, 2, 3, 4])
    expected = _stream_from_values([1, 2, 9, 4])

    with pytest.raises(AssertionError, match="seq 2"):
        assert_streams_identical(actual, expected)
