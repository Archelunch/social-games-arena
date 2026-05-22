"""Render Werewolf events as legible, agent-facing prose.

`GameMemory` is game-agnostic and renders events as raw `type + JSON`
(`[R1] kill_resolved {"victim":"Dave"}`) — fine for deep dives, but illegible to
small models. In a real run a wolf could not tell a night-kill
(`kill_resolved`) from a village exile (`exile_resolved`) from those tokens and
conflated an ally's exile with a pack kill, then confessed in a public speech.
This module is the Werewolf-specific translation that lives under `games/` so
`agents/` carries no game vocabulary (the layering rule in `memory.py`).

It renders only facts the caller is entitled to: public events, plus the
caller's own private events (which is all `memory.events` ever holds for that
player). It states *what happened and how* — never strategy. Phase labelling
comes from `Event.phase`, so a night death and a day exile are unambiguous.
"""

from collections.abc import Mapping, Sequence

from social_deduction_bench.engine import Event, Phase
from social_deduction_bench.games.werewolf.events import (
    ABSTAIN,
    ACCUSATION,
    BID,
    DEFENSE,
    DISCUSSION_RESOLVED,
    DOCTOR_PROTECT,
    EXILE_RESOLVED,
    KILL_BALLOTS,
    KILL_RESOLVED,
    SEER_INSPECT,
    SPEECH,
    TOOL_REJECTED,
    WEREWOLF_CHAT,
)


def describe_events(events: Sequence[Event], *, last_n_rounds: int | None = None, caller: str | None = None) -> str:
    """Render `events` as one legible line each, newline-joined.

    `last_n_rounds` keeps only events whose round is within the last N of the
    latest round seen (mirrors `GameMemory.recall`'s window); `None` keeps all,
    `0` keeps none, and a negative window raises `ValueError`. `caller` selects
    first-person phrasing for that player's own private events (the seer's
    inspection, the doctor's protect, the player's bid, its own pack message);
    it never affects which events are shown, since `memory.events` already holds
    only routed events. Events that add nothing to a pending decision (the
    terminal `game_over`) are omitted, so an all-omitted input renders `""`.
    """
    if last_n_rounds is not None and last_n_rounds < 0:
        raise ValueError(f"last_n_rounds must be non-negative or None, got {last_n_rounds}")

    windowed = _window(events, last_n_rounds)
    lines = [line for event in windowed if (line := _describe(event, caller)) is not None]
    return "\n".join(lines)


def _window(events: Sequence[Event], last_n_rounds: int | None) -> tuple[Event, ...]:
    """Filter `events` to the last `last_n_rounds` rounds (mirrors `recall`)."""
    if last_n_rounds is None:
        return tuple(events)
    if not events or last_n_rounds == 0:
        return ()
    latest = max(event.round for event in events)
    cutoff = latest - last_n_rounds
    return tuple(event for event in events if event.round > cutoff)


def _label(event: Event) -> str:
    """Return the legible time stamp, e.g. `Night R1` / `Day R2`."""
    period = "Night" if event.phase is Phase.NIGHT else "Day"
    return f"{period} R{event.round}"


def _describe(event: Event, caller: str | None) -> str | None:
    """Render one event as a line, or `None` to omit it from a brief."""
    label = _label(event)
    payload = event.payload
    speaker_self = caller is not None

    if event.type == KILL_RESOLVED:
        victim = payload.get("victim")
        if victim is None:
            return f"{label}: no one died (the target was protected)."
        return f"{label}: {victim} was killed by the werewolves."

    if event.type == EXILE_RESOLVED:
        exiled = payload.get("exiled")
        if exiled is None:
            return f"{label}: the vote tied — no one was exiled."
        return f"{label}: {exiled} was exiled by village vote ({_tally(payload.get('ballots'))})."

    if event.type == SPEECH:
        return f'{label}: {payload.get("speaker")} said: "{payload.get("message")}"'

    if event.type == ACCUSATION:
        accuser = payload.get("accuser")
        who = "you" if speaker_self and accuser == caller else accuser
        return f'{label}: {who} accused {payload.get("target")}: "{payload.get("reason")}"'

    if event.type == DEFENSE:
        defender = payload.get("defender")
        who = "you" if speaker_self and defender == caller else defender
        return f'{label}: {who} defended {payload.get("defended")}: "{payload.get("reason")}"'

    if event.type == SEER_INSPECT:
        return f"{label}: your inspection of {payload.get('target')} returned: {payload.get('faction')}."

    if event.type == DOCTOR_PROTECT:
        return f"{label}: you protected {payload.get('target')}."

    if event.type == WEREWOLF_CHAT:
        speaker = payload.get("speaker")
        who = "you" if speaker_self and speaker == caller else speaker
        return f'{label}: pack chat — {who}: "{payload.get("message")}"'

    if event.type == KILL_BALLOTS:
        return f"{label}: pack kill votes — {_ballots(payload.get('ballots'))}."

    if event.type == BID:
        return f"{label}: you bid {payload.get('amount')} for a speaking slot."

    if event.type == TOOL_REJECTED:
        return f"{label}: your {payload.get('tool')} call was rejected ({payload.get('reason')})."

    if event.type == DISCUSSION_RESOLVED:
        speakers = payload.get("speakers")
        order = ", ".join(speakers) if isinstance(speakers, list) else str(speakers)
        bids = payload.get("bids")
        if isinstance(bids, Mapping) and bids:
            # Bids are public at resolution — surface who bid high/low (eager vs.
            # quiet). Sorted by name so the line is replayable.
            rendered = ", ".join(f"{name} {bids[name]}" for name in sorted(bids))
            return f"{label}: speaking order — {order}; bids — {rendered}."
        return f"{label}: speaking order — {order}."

    # game_over and any unmodelled type add nothing to a pending decision.
    return None


def _tally(ballots: object) -> str:
    """Summarise an exile-vote `ballots` map as `Name N, Name N` (count desc, name asc)."""
    if not isinstance(ballots, Mapping):
        return ""
    counts: dict[str, int] = {}
    for target in ballots.values():
        if isinstance(target, str) and target != ABSTAIN:
            counts[target] = counts.get(target, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{name} {count}" for name, count in ordered)


def _ballots(ballots: object) -> str:
    """Render a kill-ballot `voter -> target` map as `voter→target` pairs (voter asc)."""
    if not isinstance(ballots, Mapping):
        return ""
    return ", ".join(f"{voter}→{target}" for voter, target in sorted(ballots.items()))
