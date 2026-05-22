"""Tests for `ScriptedDecisions` — the `DecisionSource` backed by a fixed script.

`ScriptedDecisions` is the test- and smoke-run-time wiring that feeds
pre-decided night/day actions, optional werewolf chat, optional bids, and
optional speeches into `run_game`. T29 extends it with three new staging
fields (chats, bids, speeches) so a scripted game can exercise the
dialogue / bidding paths without an LLM. The legacy two-field construction
(`nights=`, `days=` only) must keep working — every M2 test relies on it.
"""

import pytest

from social_deduction_bench.engine import GameState, Phase
from social_deduction_bench.games.werewolf.day import DayActions
from social_deduction_bench.games.werewolf.events import (
    ACCUSATION,
    BID,
    DEFENSE,
    SPEECH,
    WEREWOLF_CHAT,
)
from social_deduction_bench.games.werewolf.night import NightActions
from social_deduction_bench.games.werewolf.scripted import ScriptedDecisions

ROSTER = (
    ("Wolf1", "werewolf"),
    ("Wolf2", "werewolf"),
    ("Seer", "seer"),
    ("Doc", "doctor"),
    ("Vil1", "villager"),
    ("Vil2", "villager"),
    ("Vil3", "villager"),
)


def _night_state() -> GameState:
    return GameState.initial(ROSTER)


def _day_state() -> GameState:
    return GameState.initial(ROSTER).with_phase(Phase.DAY)


def test_legacy_two_field_construction_still_works() -> None:
    """Constructing with only `nights=` and `days=` is back-compat.

    Every M2 test uses this shape; T29 adds optional fields and must not
    break the legacy seat.
    """
    night = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})
    day = DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})
    decisions = ScriptedDecisions(nights=[night], days=[day])

    assert decisions.night_actions(_night_state()) is night
    assert decisions.day_actions(_day_state()) is day
    assert decisions.drain_drafts() == ()


def test_bids_default_is_empty_when_not_scripted() -> None:
    """`bids()` returns an empty mapping when no day_bids are scripted.

    The default lets legacy scripts run without supplying bid data. The loop
    treats an empty bid map as "no bidders" — `resolve_discussion` clamps to
    zero speakers.
    """
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
    )

    assert decisions.bids(_day_state()) == {}


def test_next_speech_returns_empty_when_not_scripted() -> None:
    """`next_speech()` returns "" when no day_speeches are scripted for the speaker."""
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
    )

    assert decisions.next_speech(_day_state(), "Wolf1") == ""


def test_night_chats_drain_as_werewolf_chat_drafts() -> None:
    """Per-round `night_chats` flush as `WEREWOLF_CHAT` drafts on the chat sub-phase.

    Each `(speaker, message)` becomes one `WEREWOLF_CHAT` draft whose
    `recipients` is the sorted living werewolf pack. The drafts flush once
    and the buffer empties; a second `drain_drafts()` returns `()`. The
    chats are staged by `night_chat` (the first half of the two-phase
    night), not `night_actions`.
    """
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        night_chats=[(("Wolf1", "hunt the seer"), ("Wolf2", "agreed"))],
    )
    state = _night_state()

    # The driver calls night_chat first; chats are staged on that call.
    decisions.night_chat(state)
    drafts = decisions.drain_drafts()

    assert len(drafts) == 2
    assert all(d.type == WEREWOLF_CHAT for d in drafts)
    assert drafts[0].payload == {"speaker": "Wolf1", "message": "hunt the seer"}
    assert drafts[1].payload == {"speaker": "Wolf2", "message": "agreed"}
    assert drafts[0].recipients == ("Wolf1", "Wolf2")
    assert drafts[1].recipients == ("Wolf1", "Wolf2")

    # Buffer empties — a second drain returns nothing.
    assert decisions.drain_drafts() == ()


def test_night_actions_no_longer_stages_chats() -> None:
    """`night_actions` stages no `WEREWOLF_CHAT` drafts — that is `night_chat`'s job.

    The two-phase night splits chat from the kill vote so wolves can observe
    each other before voting. If `night_actions` still staged chats, they
    would land in the wrong sub-phase and never be observed in time.
    """
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        night_chats=[(("Wolf1", "hunt the seer"), ("Wolf2", "agreed"))],
    )
    state = _night_state()

    decisions.night_actions(state)
    assert decisions.drain_drafts() == ()


def test_day_bids_drain_as_bid_drafts() -> None:
    """`bids()` stages one private `BID` draft per bidder; `drain_drafts()` flushes them."""
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        day_bids=[{"Wolf1": 5, "Vil1": 9}],
    )
    state = _day_state()

    bids = decisions.bids(state)
    assert bids == {"Wolf1": 5, "Vil1": 9}

    drafts = decisions.drain_drafts()
    by_bidder = {d.payload["bidder"]: d for d in drafts}
    assert set(by_bidder) == {"Wolf1", "Vil1"}
    assert by_bidder["Wolf1"].type == BID
    assert by_bidder["Wolf1"].payload == {"bidder": "Wolf1", "amount": 5}
    assert by_bidder["Wolf1"].recipients == ("Wolf1",)
    assert by_bidder["Vil1"].payload == {"bidder": "Vil1", "amount": 9}
    assert by_bidder["Vil1"].recipients == ("Vil1",)


def test_next_speech_drains_a_public_speech_draft_per_speaker() -> None:
    """`next_speech()` returns the scripted line per speaker and stages one public `SPEECH`.

    The driver calls `next_speech` once per resolved speaker; each call looks
    up that speaker's scripted message for the round and stages its draft, so
    the engine can drain + observe it before the next speaker.
    """
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        day_speeches=[(("Vil1", "I suspect Wolf2"), ("Wolf2", "I'm a villager"))],
    )
    state = _day_state()

    assert decisions.next_speech(state, "Vil1") == "I suspect Wolf2"
    first = decisions.drain_drafts()
    assert [d.payload for d in first] == [{"speaker": "Vil1", "message": "I suspect Wolf2"}]
    assert first[0].type == SPEECH
    assert first[0].recipients == ()

    assert decisions.next_speech(state, "Wolf2") == "I'm a villager"
    second = decisions.drain_drafts()
    assert [d.payload for d in second] == [{"speaker": "Wolf2", "message": "I'm a villager"}]


def test_next_reaction_drains_accuse_and_defend_drafts() -> None:
    """`next_reaction()` stages a public `ACCUSATION` / `DEFENSE` per scripted reactor.

    Indexed by round like speeches; an `"accuse"` entry stages an `ACCUSATION`
    (`{accuser, target, reason}`), a `"defend"` entry a `DEFENSE`
    (`{defender, defended, reason}`). A reactor with no scripted entry passes and
    stages nothing, so the driver can drain + observe each before the next.
    """
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        day_reactions=[
            (
                ("Vil2", "accuse", "Wolf1", "you dodged"),
                ("Doc", "defend", "Seer", "the seer is clean"),
            ),
        ],
    )
    state = _day_state()

    decisions.next_reaction(state, "Vil2")
    first = decisions.drain_drafts()
    assert len(first) == 1
    assert first[0].type == ACCUSATION
    assert first[0].recipients == ()
    assert first[0].payload == {"accuser": "Vil2", "target": "Wolf1", "reason": "you dodged"}

    decisions.next_reaction(state, "Doc")
    second = decisions.drain_drafts()
    assert len(second) == 1
    assert second[0].type == DEFENSE
    assert second[0].payload == {"defender": "Doc", "defended": "Seer", "reason": "the seer is clean"}

    # A reactor with no scripted entry passes silently — no draft.
    decisions.next_reaction(state, "Wolf1")
    assert decisions.drain_drafts() == ()


def test_next_reaction_rejects_an_unknown_kind() -> None:
    """A scripted reaction `kind` other than accuse/defend fails loud.

    A typo in the script (`"acuse"`) would otherwise silently fall through to a
    DEFENSE draft — a wrong transcript that no assertion would catch. Fail loud
    so the test author sees the mistake immediately (CLAUDE.md rule 11).
    """
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        day_reactions=[(("Vil2", "acuse", "Wolf1", "typo"),)],
    )
    with pytest.raises(ValueError, match=r"accuse.*defend"):
        decisions.next_reaction(_day_state(), "Vil2")


def test_next_reaction_returns_nothing_when_not_scripted() -> None:
    """With no `day_reactions`, every reactor passes and stages nothing.

    Legacy scripts (no reaction data) run the reaction round as all-pass, so the
    `nights=`, `days=`-only construction keeps working.
    """
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
    )
    decisions.next_reaction(_day_state(), "Vil2")
    assert decisions.drain_drafts() == ()


def test_drain_drafts_returns_a_tuple_and_empties_the_buffer() -> None:
    """`drain_drafts()` is idempotent after a flush — repeated calls return `()`."""
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        day_bids=[{"Wolf1": 5}],
    )

    decisions.bids(_day_state())
    first = decisions.drain_drafts()
    second = decisions.drain_drafts()

    assert isinstance(first, tuple)
    assert len(first) == 1
    assert second == ()
