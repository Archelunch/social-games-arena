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
from social_deduction_bench.games.werewolf.events import GAME_OVER, EventDraft
from social_deduction_bench.games.werewolf.night import NightActions, resolve_night
from social_deduction_bench.games.werewolf.win import is_game_over, winner


class DecisionSource(Protocol):
    """Supplies the decided actions for each phase.

    Scripted in M2; an M4 DSPy agent adapter implements the same Protocol — the
    loop does not change. Each accessor receives the current `GameState` so an
    agent-backed source can decide from the live position.
    """

    def night_actions(self, state: GameState, /) -> NightActions: ...

    def day_actions(self, state: GameState, /) -> DayActions: ...


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

        night = resolve_night(state, decisions.night_actions(state), rng)
        _log_drafts(log, state, night.drafts)
        state = night.state
        if is_game_over(state):
            break

        state = advance_phase(state)

        day = resolve_day(state, decisions.day_actions(state))
        _log_drafts(log, state, day.drafts)
        state = day.state
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
