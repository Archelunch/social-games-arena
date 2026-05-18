"""Tests for read-side observation routing (T05).

These encode benchmark invariant #2: agents never read hidden state. The
engine emits public events to everyone and private events only to their named
recipients; `observations_for` is the single filter that enforces it. If that
filter is ever inverted, dropped, or leaks a private payload to a non-recipient,
these tests fail loudly — that is a Critical benchmark-integrity failure, not a
style nit.

They also pin invariants #4 (routing is deterministic) and #5 (routing is a
pure read — it never mutates the append-only log).
"""

from collections.abc import Iterator

import pytest

from social_deduction_bench.engine import (
    Event,
    EventLog,
    Phase,
    observations_for,
    public_events,
)


def _mixed_log() -> EventLog:
    """A small night/day log mixing public broadcasts with private events.

    seq 0 public · seq 1 private to Cara · seq 2 private to the Bob+Dan pack ·
    seq 3 public — so a filtered observer sees a non-contiguous seq sequence.
    """
    log = EventLog()
    log.append(round=1, phase=Phase.NIGHT, type="phase_change", payload={"phase": "night"})
    log.append(
        round=1,
        phase=Phase.NIGHT,
        type="seer_inspect",
        payload={"target": "Bob", "is_werewolf": True},
        recipients=("Cara",),
    )
    log.append(
        round=1,
        phase=Phase.NIGHT,
        type="werewolf_chat",
        payload={"text": "kill Cara"},
        recipients=("Bob", "Dan"),
    )
    log.append(round=1, phase=Phase.DAY, type="speak", payload={"text": "I suspect Bob"})
    return log


# --- Public broadcast routing ---------------------------------------------


def test_observer_sees_all_public_events() -> None:
    """Every empty-`recipients` event reaches every player — the routing baseline.

    If public events were filtered, agents would be blind to the shared game.
    """
    log = _mixed_log()
    for player in ("Alice", "Bob", "Cara", "Dan"):
        seen = observations_for(log, player)
        assert log.events[0] in seen  # phase_change
        assert log.events[3] in seen  # speak


# --- Private delivery to recipients ---------------------------------------


def test_observer_sees_private_events_addressed_to_them() -> None:
    """A private event must be *delivered* to its named recipient.

    The seer must see its own inspection result, not just have it hidden from
    everyone else — routing delivers as well as redacts.
    """
    log = _mixed_log()
    seen = observations_for(log, "Cara")
    assert log.events[1] in seen  # seer_inspect addressed to Cara


def test_non_recipient_observations_contain_zero_private_events() -> None:
    """The core hidden-state-leak test (invariant #2).

    A non-recipient's observation stream must contain no event whose non-empty
    `recipients` omit them — the seer result must be unreachable to Bob. A
    non-recipient reading a private payload is a Critical integrity failure;
    this test fails if the filter is ever inverted or dropped.
    """
    log = _mixed_log()
    seen = observations_for(log, "Bob")
    assert seen  # guard: a vacuous loop below would hide a total filter failure
    for event in seen:
        assert event.is_public or "Bob" in event.recipients
    assert log.events[1] not in seen  # the seer_inspect never reaches Bob


def test_pack_chat_routes_to_every_werewolf() -> None:
    """A multi-recipient event reaches all recipients and excludes everyone else.

    The werewolf pack shares one private channel; routing must deliver pack
    chat to both members — a single-recipient assumption would break it.
    """
    log = _mixed_log()
    pack_chat = log.events[2]
    assert pack_chat in observations_for(log, "Bob")
    assert pack_chat in observations_for(log, "Dan")
    assert pack_chat not in observations_for(log, "Cara")
    assert pack_chat not in observations_for(log, "Alice")


# --- Order & seq preservation ---------------------------------------------


def test_observations_preserve_original_seq() -> None:
    """Filtered output keeps each event's original, now non-contiguous, `seq`.

    `seq` is the engine's stable global id that replay (T28) and determinism
    checks (T08) correlate against; renumbering would silently misalign an
    agent's memory with the transcript.
    """
    seen = observations_for(_mixed_log(), "Alice")
    assert [e.seq for e in seen] == [0, 3]


def test_observations_preserve_relative_order() -> None:
    """Routed events stay in log order — *when* things were said matters.

    Reordering an agent's observations would corrupt its reasoning and replay.
    """
    seen = observations_for(_mixed_log(), "Cara")
    assert [e.seq for e in seen] == sorted(e.seq for e in seen)
    assert [e.seq for e in seen] == [0, 1, 3]


# --- Purity & determinism --------------------------------------------------


def test_observations_for_does_not_mutate_source_log() -> None:
    """Routing is a pure read — the append-only log is untouched (invariant #5)."""
    log = _mixed_log()
    before = log.events
    observations_for(log, "Cara")
    assert log.events == before
    assert len(log) == 4


def test_observations_for_is_deterministic() -> None:
    """Two independent equal logs route to equal observation streams (invariant #4).

    Routing must add no nondeterminism (no set-iteration order, no wall-clock)
    and hold no cross-call state, so T08 can compare two runs of the same
    seeded game. Using two separately built logs catches a stateful filter
    that a same-object repeat call would miss.
    """
    first = observations_for(_mixed_log(), "Cara")
    second = observations_for(_mixed_log(), "Cara")
    assert first == second
    assert [e.seq for e in first] == [0, 1, 3]


# --- Validation & edge cases ----------------------------------------------


def test_observations_for_rejects_blank_player() -> None:
    """A blank player name fails loud rather than degrading to public-only.

    `observations_for(events, "")` returning public events would silently hide
    a caller bug — a blank name can never be a valid recipient.
    """
    with pytest.raises(ValueError, match="player"):
        observations_for(_mixed_log(), "")


def test_observations_for_unknown_player_yields_public_only() -> None:
    """A non-blank unknown name observes only public events.

    The game-agnostic router has no roster; a name matching no `recipients`
    correctly sees nothing private — that is the safe result, not an error.
    """
    seen = observations_for(_mixed_log(), "Eve")
    assert [e.seq for e in seen] == [0, 3]
    assert all(e.is_public for e in seen)


def test_observations_for_matches_recipient_names_exactly() -> None:
    """Recipient matching is exact — case and whitespace are significant.

    A near-miss name (`"cara"`, `" Cara"`) must NOT receive Cara's private
    event; routing keys on the literal recipient string, so a sloppy lookup
    that normalized case or trimmed space would leak hidden state.
    """
    log = _mixed_log()
    seer_inspect = log.events[1]  # recipients=("Cara",)
    assert seer_inspect not in observations_for(log, "cara")
    assert seer_inspect not in observations_for(log, " Cara")
    assert seer_inspect in observations_for(log, "Cara")


def test_observations_for_accepts_a_bare_iterable() -> None:
    """Routing accepts any `Iterable[Event]`, not just an `EventLog`.

    Callers route live logs, tuple snapshots, and generators without
    converting — pins the input contract.
    """
    events = tuple(_mixed_log().events)

    def _gen() -> Iterator[Event]:
        yield from events

    assert observations_for(_gen(), "Cara") == observations_for(events, "Cara")


def test_observations_for_handles_empty_input() -> None:
    """An empty event stream routes to an empty tuple — a game may end early."""
    assert observations_for((), "Alice") == ()


def test_public_events_returns_only_broadcasts() -> None:
    """`public_events` is the spectator view — exactly the public broadcasts.

    A replay/spectator viewer (T28) must never be shown a private event; the
    public events keep their original order and `seq`.
    """
    seen = public_events(_mixed_log())
    assert [e.seq for e in seen] == [0, 3]
    assert all(e.is_public for e in seen)
