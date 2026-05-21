"""Tests for `ScriptedDecisions` — the `DecisionSource` backed by a fixed script.

`ScriptedDecisions` is the test- and smoke-run-time wiring that feeds
pre-decided night/day actions, optional werewolf chat, optional bids, and
optional speeches into `run_game`. T29 extends it with three new staging
fields (chats, bids, speeches) so a scripted game can exercise the
dialogue / bidding paths without an LLM. The legacy two-field construction
(`nights=`, `days=` only) must keep working — every M2 test relies on it.
"""

from social_deduction_bench.engine import GameState, Phase
from social_deduction_bench.games.werewolf.day import DayActions
from social_deduction_bench.games.werewolf.events import (
    BID,
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


def test_speeches_default_is_empty_when_not_scripted() -> None:
    """`speeches()` returns an empty tuple when no day_speeches are scripted."""
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
    )

    assert decisions.speeches(_day_state(), ()) == ()


def test_night_chats_drain_as_werewolf_chat_drafts() -> None:
    """Per-round `night_chats` flush as `WEREWOLF_CHAT` drafts on drain.

    Each `(speaker, message)` becomes one `WEREWOLF_CHAT` draft whose
    `recipients` is the sorted living werewolf pack. The drafts flush once
    and the buffer empties; a second `drain_drafts()` returns `()`.
    """
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        night_chats=[(("Wolf1", "hunt the seer"), ("Wolf2", "agreed"))],
    )
    state = _night_state()

    # The driver calls night_actions first; chats are staged on that call.
    decisions.night_actions(state)
    drafts = decisions.drain_drafts()

    assert len(drafts) == 2
    assert all(d.type == WEREWOLF_CHAT for d in drafts)
    assert drafts[0].payload == {"speaker": "Wolf1", "message": "hunt the seer"}
    assert drafts[1].payload == {"speaker": "Wolf2", "message": "agreed"}
    assert drafts[0].recipients == ("Wolf1", "Wolf2")
    assert drafts[1].recipients == ("Wolf1", "Wolf2")

    # Buffer empties — a second drain returns nothing.
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


def test_day_speeches_drain_as_public_speech_drafts_in_order() -> None:
    """`speeches()` stages public `SPEECH` drafts in the scripted order."""
    decisions = ScriptedDecisions(
        nights=[NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        days=[DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})],
        day_speeches=[(("Vil1", "I suspect Wolf2"), ("Wolf2", "I'm a villager"))],
    )
    state = _day_state()

    speeches = decisions.speeches(state, ("Vil1", "Wolf2"))
    assert speeches == (("Vil1", "I suspect Wolf2"), ("Wolf2", "I'm a villager"))

    drafts = decisions.drain_drafts()
    assert len(drafts) == 2
    assert [d.type for d in drafts] == [SPEECH, SPEECH]
    assert [d.payload for d in drafts] == [
        {"speaker": "Vil1", "message": "I suspect Wolf2"},
        {"speaker": "Wolf2", "message": "I'm a villager"},
    ]
    assert all(d.recipients == () for d in drafts)


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
