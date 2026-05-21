"""The full Werewolf game-loop driver (WEREWOLF_DESIGN.md §4).

`run_game` is the M2 capstone: it wires seeded setup, night/day resolution, and
the win check into a complete game, owning the single `EventLog` and the single
`GameRNG`. The seed is threaded once into one `GameRNG` and the only stochastic
point in a game (the night kill-vote tie-break) draws from it, so the same seed
plus the same scripted decisions replay byte-identically (invariant #4). Every
resolver draft is appended through the private-event guard before it enters the
log, so a leaky private draft fails loud rather than broadcasting hidden state
(invariant #2). The log is append-only, the basis for replay and post-hoc
metrics (invariant #5).

Decisions arrive through the `DecisionSource` seam: M2 scripts them, an M4 DSPy
agent adapter implements the same Protocol — the loop itself does not change.
"""

from collections.abc import Sequence
from typing import Protocol

from social_deduction_bench.engine import (
    Event,
    EventLog,
    EventStream,
    GameRNG,
    GameState,
    StreamHeader,
    advance_phase,
    assert_recipients_present,
)
from social_deduction_bench.games.werewolf.config import PRIVATE_EVENT_TYPES
from social_deduction_bench.games.werewolf.day import DayActions, resolve_day
from social_deduction_bench.games.werewolf.discussion import BiddingActions, resolve_discussion
from social_deduction_bench.games.werewolf.events import (
    DISCUSSION_RESOLVED,
    GAME_OVER,
    EventDraft,
)
from social_deduction_bench.games.werewolf.night import NightActions, resolve_night
from social_deduction_bench.games.werewolf.win import is_game_over, winner


class DecisionSource(Protocol):
    """Supplies the decided actions for each phase, and absorbs the resulting events.

    Each accessor receives the current `GameState` so an agent-backed source
    can decide from the live position. After each night and day resolution,
    the driver calls `observe` with the events that were just appended; an
    agent-backed source uses it to push routed observations into each player's
    `GameMemory`. The terminal `GAME_OVER` event is not routed through
    `observe` — no agent will consult its memory after the game ends. A
    scripted source may ignore `observe` entirely.

    Beyond the per-phase action accessors, three hooks support T29's dialogue
    surface: `bids` returns the day's bid map (one ReAct loop per alive player
    in an agent-backed source); `speeches` returns each chosen speaker's
    statement; `drain_drafts` returns any agent-staged event drafts
    (`WEREWOLF_CHAT`, `BID`, `SPEECH`, `TOOL_REJECTED`) since the last drain.
    The driver calls `drain_drafts` after each agent-facing step and routes
    the drafts through the same private-event guard as the resolvers'
    drafts.
    """

    def night_actions(self, state: GameState, /) -> NightActions: ...

    def day_actions(self, state: GameState, /) -> DayActions: ...

    def bids(self, state: GameState, /) -> dict[str, int]: ...

    def speeches(self, state: GameState, speakers: tuple[str, ...], /) -> tuple[tuple[str, str], ...]: ...

    def drain_drafts(self) -> tuple[EventDraft, ...]: ...

    def observe(self, state: GameState, new_events: tuple[Event, ...], /) -> None: ...


def _log_drafts(log: EventLog, state: GameState, drafts: Sequence[EventDraft]) -> None:
    """Append each resolver draft to the log, guarding private events first.

    Each draft passes through `assert_recipients_present` before it is logged so
    a declared-private draft emitted without recipients fails loud rather than
    entering the append-only transcript as a broadcast (invariant #2). Drafts are
    logged against `state`'s round and phase — the phase they were resolved in.
    """
    for draft in drafts:
        assert_recipients_present(draft.type, draft.recipients, PRIVATE_EVENT_TYPES)
        log.append(
            round=state.round,
            phase=state.phase,
            type=draft.type,
            payload=draft.payload,
            recipients=draft.recipients,
        )


def _drain_into_log(log: EventLog, state: GameState, decisions: DecisionSource) -> None:
    """Drain any agent-staged drafts and log them through the private-event guard."""
    _log_drafts(log, state, decisions.drain_drafts())


def _run_night(state: GameState, log: EventLog, decisions: DecisionSource, rng: GameRNG) -> GameState:
    """Resolve one night phase end-to-end and return the post-night state.

    Order: collect actions (which may stage `WEREWOLF_CHAT` / `TOOL_REJECTED`
    drafts via the adapter), drain those drafts, resolve the night, log the
    resolver's drafts, then route the appended events back through `observe`.
    """
    before = len(log.events)
    night_actions = decisions.night_actions(state)
    _drain_into_log(log, state, decisions)
    night = resolve_night(state, night_actions, rng)
    _log_drafts(log, state, night.drafts)
    decisions.observe(state, log.events[before:])
    return night.state


def _run_day(state: GameState, log: EventLog, decisions: DecisionSource, rng: GameRNG) -> GameState:
    """Resolve one day phase end-to-end and return the post-day state.

    Sub-phases in order: bidding → discussion resolution → speeches → exile
    vote. After each agent-facing step the adapter's staged drafts are
    drained through the private-event guard. The chosen speakers (returned by
    `speeches`) are cross-checked against the resolver's order — divergence
    is a caller bug, fail loud rather than silently log a contradictory
    transcript.
    """
    before = len(log.events)

    bid_map = decisions.bids(state)
    _drain_into_log(log, state, decisions)

    discussion = resolve_discussion(state, BiddingActions(bids=bid_map), rng)
    _log_drafts(
        log,
        state,
        (
            EventDraft(
                type=DISCUSSION_RESOLVED,
                payload={"speakers": list(discussion.speakers), "bids": dict(bid_map)},
                recipients=(),
            ),
        ),
    )

    given_speakers = decisions.speeches(state, discussion.speakers)
    given_order = tuple(speaker for speaker, _ in given_speakers)
    expected_prefix = discussion.speakers[: len(given_order)]
    if given_order and given_order != expected_prefix:
        raise RuntimeError(
            f"decisions.speeches returned {given_order!r} but resolver chose {discussion.speakers!r}",
        )
    _drain_into_log(log, state, decisions)

    day_actions = decisions.day_actions(state)
    _drain_into_log(log, state, decisions)
    day = resolve_day(state, day_actions)
    _log_drafts(log, state, day.drafts)
    decisions.observe(state, log.events[before:])
    return day.state


def run_game(
    roster: Sequence[tuple[str, str]],
    seed: int,
    decisions: DecisionSource,
    game_id: str = "werewolf",
    max_rounds: int = 20,
) -> EventStream:
    """Drive a complete Werewolf game and return its transcript (WEREWOLF_DESIGN.md §4).

    `run_game` is the single owner of the `EventLog` and the `GameRNG`: the seed
    is threaded once into one `GameRNG`, so the game is deterministic and
    replayable (invariant #4). Each night and day is resolved by the T11/T12
    resolvers and their drafts logged through the private-event guard
    (invariant #2) onto the append-only stream (invariant #5). Raises
    `RuntimeError` if the game does not terminate within `max_rounds` — a
    non-terminating script fails loud instead of hanging.
    """
    state = GameState.initial(roster)
    rng = GameRNG(seed)
    log = EventLog()

    while not is_game_over(state):
        if state.round > max_rounds:
            raise RuntimeError(f"game did not terminate within {max_rounds} rounds")

        state = _run_night(state, log, decisions, rng)
        if is_game_over(state):
            break

        state = advance_phase(state)
        state = _run_day(state, log, decisions, rng)
        if is_game_over(state):
            break

        state = advance_phase(state)

    final = winner(state)
    if final is None:
        raise RuntimeError("run_game exited its loop without a winner")

    log.append(
        round=state.round,
        phase=state.phase,
        type=GAME_OVER,
        payload={"winner": final.value},
    )

    return EventStream(
        header=StreamHeader(seed=seed, game_id=game_id, players=tuple(roster)),
        log=log,
    )
