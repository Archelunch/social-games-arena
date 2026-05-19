"""Tests for Werewolf event-type constants and the `EventDraft` spec (T11).

The event-type constants name every event the Werewolf rules emit. The private
ones must exactly equal `config.PRIVATE_EVENT_TYPES` — that frozenset is what
the engine's private-event guard enforces, and a drift between the emitter
constants and the declared set would silently leave a private channel
unguarded (invariant #2). `EventDraft` is the transient event spec resolution
functions return; the game loop turns each draft into a logged `Event`.
"""

from dataclasses import FrozenInstanceError

import pytest

from social_deduction_bench.games.werewolf.config import PRIVATE_EVENT_TYPES
from social_deduction_bench.games.werewolf.events import (
    DOCTOR_PROTECT,
    KILL_RESOLVED,
    SEER_INSPECT,
    WEREWOLF_CHAT,
    EventDraft,
)


def test_private_event_constants_equal_the_declared_private_set() -> None:
    """The three private-event constants are exactly `PRIVATE_EVENT_TYPES`.

    This binds the emitter constants to the set the guard enforces. If a new
    private channel is added as a constant but not to `PRIVATE_EVENT_TYPES`,
    the guard would not cover it and a leak could ship — this test fails first.
    """
    assert {SEER_INSPECT, DOCTOR_PROTECT, WEREWOLF_CHAT} == PRIVATE_EVENT_TYPES


def test_kill_resolved_is_a_public_type() -> None:
    """`KILL_RESOLVED` is not a declared-private type — a death is public.

    Who died at night is common knowledge announced to everyone; if it landed
    in `PRIVATE_EVENT_TYPES` the guard would wrongly demand recipients on a
    broadcast.
    """
    assert KILL_RESOLVED not in PRIVATE_EVENT_TYPES


def test_event_draft_defaults_to_a_public_empty_event() -> None:
    """An `EventDraft` with only a type defaults to empty payload and recipients.

    The common case is a simple public event; empty `recipients` makes it a
    broadcast (invariant #2's marker).
    """
    draft = EventDraft(type=KILL_RESOLVED)

    assert draft.payload == {}
    assert draft.recipients == ()


def test_event_draft_is_frozen() -> None:
    """`EventDraft` is immutable — a spec cannot be edited after creation.

    A draft is handed from a resolver to the loop; a mutable draft could be
    altered in transit, desyncing the logged event from the resolved outcome.
    """
    draft = EventDraft(type=KILL_RESOLVED)

    with pytest.raises(FrozenInstanceError):
        draft.type = SEER_INSPECT  # type: ignore[misc]
