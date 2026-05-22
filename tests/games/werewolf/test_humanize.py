"""Tests for the Werewolf event humanizer.

Agents reason over the brief's "Recent events" block. Raw `type + JSON`
(`[R1] kill_resolved {"victim":"Dave"}`) is illegible to small models — in a
real run a wolf conflated an ally's *exile* with a pack *night-kill* and
confessed in a public speech. `describe_events` translates the engine's events
into plain language, distinguishing how a player died, while never inventing
strategy. These tests pin the wording per event type, the night/day labelling,
the last-N-rounds window, and first-person rendering of the caller's own
private events.
"""

from __future__ import annotations

import pytest

from social_deduction_bench.engine import Event, Phase
from social_deduction_bench.games.werewolf.events import (
    ACCUSATION,
    BID,
    DEFENSE,
    DISCUSSION_RESOLVED,
    DOCTOR_PROTECT,
    EXILE_RESOLVED,
    GAME_OVER,
    KILL_BALLOTS,
    KILL_RESOLVED,
    SEER_INSPECT,
    SPEECH,
    TOOL_REJECTED,
    WEREWOLF_CHAT,
)
from social_deduction_bench.games.werewolf.humanize import describe_events


def _ev(
    seq: int, round_: int, phase: Phase, type_: str, payload: dict[str, object], recipients: tuple[str, ...] = ()
) -> Event:
    return Event(seq=seq, round=round_, phase=phase, type=type_, payload=payload, recipients=recipients)


def test_kill_resolved_names_the_werewolves_as_the_cause() -> None:
    """A night death must read as a werewolf kill, not a neutral "died".

    Distinguishing the night-kill from the day-exile is the whole point — a
    model that can't tell them apart misattributes who is responsible.
    """
    events = (_ev(0, 1, Phase.NIGHT, KILL_RESOLVED, {"victim": "Dave"}),)
    assert describe_events(events) == "Night R1: Dave was killed by the werewolves."


def test_kill_resolved_with_no_victim_reads_as_a_protected_night() -> None:
    """`victim=None` means the target was protected — say so, don't render null."""
    events = (_ev(0, 2, Phase.NIGHT, KILL_RESOLVED, {"victim": None}),)
    assert describe_events(events) == "Night R2: no one died (the target was protected)."


def test_exile_resolved_reads_as_a_village_vote_with_tally() -> None:
    """A day death must read as a village exile, with the vote tally for context."""
    events = (
        _ev(
            0,
            1,
            Phase.DAY,
            EXILE_RESOLVED,
            {"exiled": "Alice", "ballots": {"Carol": "Alice", "Grace": "Alice", "Bob": "Carol"}},
        ),
    )
    line = describe_events(events)
    assert line == "Day R1: Alice was exiled by village vote (Alice 2, Carol 1)."


def test_exile_resolved_tie_reads_as_no_exile() -> None:
    """`exiled=None` is a tie / all-abstain — no one leaves; say that plainly."""
    events = (_ev(0, 1, Phase.DAY, EXILE_RESOLVED, {"exiled": None, "ballots": {"Carol": "abstain"}}),)
    assert describe_events(events) == "Day R1: the vote tied — no one was exiled."


def test_accusation_renders_third_person_with_reason() -> None:
    """An accusation reads as "{accuser} accused {target}: reason" to onlookers.

    The reaction round is public deliberation; a villager reading the brief must
    see who accused whom and why, so the suspicion is legible (and the model can
    react to it next turn).
    """
    events = (_ev(0, 1, Phase.DAY, ACCUSATION, {"accuser": "Alice", "target": "Bob", "reason": "dodged the vote"}),)
    assert describe_events(events) == 'Day R1: Alice accused Bob: "dodged the vote"'


def test_accusation_renders_first_person_for_the_accuser() -> None:
    """The accuser sees its own accusation in the first person ("you accused …").

    Matches the pack-chat / bid first-person convention so a player's own move
    reads naturally in its brief rather than as a stilted third-person line.
    """
    events = (_ev(0, 1, Phase.DAY, ACCUSATION, {"accuser": "Alice", "target": "Bob", "reason": "dodged the vote"}),)
    assert describe_events(events, caller="Alice") == 'Day R1: you accused Bob: "dodged the vote"'


def test_defense_renders_third_person_with_reason() -> None:
    """A defense reads as "{defender} defended {defended}: reason" to onlookers.

    Third-party defense is the signal we most want legible — a wolf defending a
    wolf is a tell only if every reader can see who defended whom.
    """
    events = (
        _ev(0, 1, Phase.DAY, DEFENSE, {"defender": "Carol", "defended": "Bob", "reason": "Bob was quiet, not guilty"}),
    )
    assert describe_events(events) == 'Day R1: Carol defended Bob: "Bob was quiet, not guilty"'


def test_defense_renders_first_person_for_the_defender() -> None:
    """The defender sees its own defense in the first person ("you defended …").

    Includes the self-defense case (defender == defended) reading naturally.
    """
    events = (_ev(0, 1, Phase.DAY, DEFENSE, {"defender": "Bob", "defended": "Bob", "reason": "I was protecting"}),)
    assert describe_events(events, caller="Bob") == 'Day R1: you defended Bob: "I was protecting"'


def test_speech_quotes_the_speaker() -> None:
    events = (_ev(0, 1, Phase.DAY, SPEECH, {"speaker": "Bob", "message": "I trust Carol."}),)
    assert describe_events(events) == 'Day R1: Bob said: "I trust Carol."'


def test_seer_inspect_renders_first_person_for_the_seer() -> None:
    """The seer's own inspection is private; render it as "you inspected ...".

    Only the seer ever has this event in memory (it is routed to the seer
    alone), so the first-person framing is correct and never leaks.
    """
    events = (
        _ev(0, 1, Phase.NIGHT, SEER_INSPECT, {"target": "Wolf1", "faction": "werewolves"}, recipients=("Carol",)),
    )
    assert describe_events(events, caller="Carol") == "Night R1: your inspection of Wolf1 returned: werewolves."


def test_doctor_protect_renders_first_person_for_the_doctor() -> None:
    events = (_ev(0, 1, Phase.NIGHT, DOCTOR_PROTECT, {"target": "Eve"}, recipients=("Dave",)),)
    assert describe_events(events, caller="Dave") == "Night R1: you protected Eve."


def test_werewolf_chat_renders_packmate_and_self() -> None:
    """Pack chat shows the speaker; the caller's own line reads as "you"."""
    events = (
        _ev(
            0,
            1,
            Phase.NIGHT,
            WEREWOLF_CHAT,
            {"speaker": "Wolf2", "message": "take Vil1"},
            recipients=("Wolf1", "Wolf2"),
        ),
        _ev(1, 1, Phase.NIGHT, WEREWOLF_CHAT, {"speaker": "Wolf1", "message": "agreed"}, recipients=("Wolf1", "Wolf2")),
    )
    rendered = describe_events(events, caller="Wolf1")
    assert rendered == ('Night R1: pack chat — Wolf2: "take Vil1"\nNight R1: pack chat — you: "agreed"')


def test_tool_rejected_reads_as_a_rejected_call() -> None:
    events = (
        _ev(
            0,
            1,
            Phase.NIGHT,
            TOOL_REJECTED,
            {"tool": "submit_kill_vote", "args": {"target": "Wolf1"}, "reason": "cannot target the caller"},
            recipients=("Wolf1",),
        ),
    )
    assert (
        describe_events(events, caller="Wolf1")
        == "Night R1: your submit_kill_vote call was rejected (cannot target the caller)."
    )


def test_kill_ballots_render_the_pack_vote_map() -> None:
    """The pack's kill ballots (private to the pack) render as voter→target pairs."""
    events = (
        _ev(
            0,
            1,
            Phase.NIGHT,
            KILL_BALLOTS,
            {"ballots": {"Wolf2": "Vil1", "Wolf1": "Vil1"}},
            recipients=("Wolf1", "Wolf2"),
        ),
    )
    # Sorted by voter so the line is replayable regardless of dict order.
    assert describe_events(events, caller="Wolf1") == "Night R1: pack kill votes — Wolf1→Vil1, Wolf2→Vil1."


def test_bid_renders_first_person_amount() -> None:
    """A player's own bid is private to them; render it as "you bid N"."""
    events = (_ev(0, 1, Phase.DAY, BID, {"bidder": "Vil1", "amount": 80}, recipients=("Vil1",)),)
    assert describe_events(events, caller="Vil1") == "Day R1: you bid 80 for a speaking slot."


def test_discussion_resolved_renders_speaking_order() -> None:
    events = (_ev(0, 1, Phase.DAY, DISCUSSION_RESOLVED, {"speakers": ["Carol", "Bob"], "bids": {}}),)
    assert describe_events(events) == "Day R1: speaking order — Carol, Bob."


def test_discussion_resolved_surfaces_per_day_bids() -> None:
    """Bids are public at resolution; the line shows who bid high/low (eager vs.
    quiet), sorted by name for replayability — the per-day detail behind the
    cumulative speaking-budgets signal."""
    events = (
        _ev(
            0,
            1,
            Phase.DAY,
            DISCUSSION_RESOLVED,
            {"speakers": ["Carol", "Bob"], "bids": {"Bob": 30, "Carol": 40, "Alice": 0}},
        ),
    )
    assert describe_events(events) == "Day R1: speaking order — Carol, Bob; bids — Alice 0, Bob 30, Carol 40."


def test_unknown_event_type_is_omitted() -> None:
    """An unmodelled event type adds nothing to a brief — it is silently dropped,
    never crashing the render with a stray type+JSON line."""
    events = (_ev(0, 1, Phase.NIGHT, "some_future_event", {"x": 1}),)
    assert describe_events(events) == ""


def test_exile_tally_breaks_count_ties_by_name() -> None:
    """The exile tally is count-desc then name-asc, so ties render deterministically."""
    events = (
        _ev(
            0,
            1,
            Phase.DAY,
            EXILE_RESOLVED,
            {"exiled": "Alice", "ballots": {"P1": "Alice", "P2": "Alice", "P3": "Carol", "P4": "Bob"}},
        ),
    )
    # Alice 2 first; Bob and Carol both 1, tie broken by name (Bob before Carol).
    assert describe_events(events) == "Day R1: Alice was exiled by village vote (Alice 2, Bob 1, Carol 1)."


def test_game_over_is_omitted_from_the_brief() -> None:
    """The game is over — there is no next decision, so it adds nothing to a brief."""
    events = (_ev(0, 3, Phase.DAY, GAME_OVER, {"winner": "villagers"}),)
    assert describe_events(events) == ""


def test_last_n_rounds_keeps_only_the_recent_window() -> None:
    """The brief inlines a recent window; deeper history stays in `recall`.

    Mirrors `GameMemory.recall`'s window: keep events whose round is within the
    last N of the latest round seen.
    """
    events = (
        _ev(0, 1, Phase.NIGHT, KILL_RESOLVED, {"victim": "Dave"}),
        _ev(1, 2, Phase.NIGHT, KILL_RESOLVED, {"victim": "Eve"}),
        _ev(2, 3, Phase.NIGHT, KILL_RESOLVED, {"victim": "Frank"}),
    )
    rendered = describe_events(events, last_n_rounds=1)
    # Latest round is 3; with a window of 1 only round-3 events survive.
    assert rendered == "Night R3: Frank was killed by the werewolves."


def test_empty_and_zero_window_render_empty() -> None:
    assert describe_events(()) == ""
    events = (_ev(0, 1, Phase.NIGHT, KILL_RESOLVED, {"victim": "Dave"}),)
    assert describe_events(events, last_n_rounds=0) == ""


def test_negative_window_raises() -> None:
    """A negative window has no legal reading — fail loud (parity with recall)."""
    with pytest.raises(ValueError, match="last_n_rounds"):
        describe_events((), last_n_rounds=-1)
