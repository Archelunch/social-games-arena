"""Tests for the private-event guard (T11 enforcement of invariant #2).

`assert_recipients_present` is the shared engine check that makes invariant #2
("agents never read hidden state") enforceable. The engine treats `Event.type`
as an opaque string, so it cannot, on its own, tell an intentional broadcast
from a private event that forgot its recipients. Each game declares its private
event types; this guard rejects any declared-private type emitted with empty
`recipients` — the leak would otherwise broadcast a seer result or pack-chat
message to every agent. The check is game-agnostic: the declared set is a
parameter, so Werewolf, ONUW, and Secret Hitler all reuse one function.
"""

import pytest

from social_deduction_bench.engine import assert_recipients_present

PRIVATE = frozenset({"seer_inspect", "werewolf_chat", "doctor_protect"})


def test_declared_private_type_with_no_recipients_is_rejected() -> None:
    """A declared-private type emitted with empty recipients fails loud.

    This is the core leak prevention: a `seer_inspect` event with no recipients
    would be a public broadcast, exposing the seer's hidden result to every
    agent and breaking invariant #2.
    """
    with pytest.raises(ValueError, match="private"):
        assert_recipients_present("seer_inspect", (), PRIVATE)


def test_declared_private_type_with_recipients_passes() -> None:
    """A correctly-routed private event is accepted — the guard does not over-block.

    `seer_inspect` addressed to the seer is exactly how a private event should
    look; the guard must let it through.
    """
    assert_recipients_present("seer_inspect", ("Seer",), PRIVATE)


def test_public_type_with_no_recipients_passes() -> None:
    """A genuine broadcast (a type not in the declared private set) is accepted.

    `kill_resolved` is public — empty recipients is correct for it. The guard
    must distinguish intent via the declared set, not flag every empty-recipient
    event, or it would block all legitimate broadcasts.
    """
    assert_recipients_present("kill_resolved", (), PRIVATE)


def test_empty_private_set_blocks_nothing() -> None:
    """With no declared private types, every event passes.

    A game (or a phase) that declares no private types must not be broken by
    the guard — the check is purely opt-in via the declared set.
    """
    assert_recipients_present("seer_inspect", (), frozenset())
