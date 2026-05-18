"""Phase state machine and the terminal-detection hook.

Upholds benchmark invariant #1: the engine — not a game — owns phase and round
progression. A game never advances state itself; it calls `advance_phase`,
which derives the next position from the canonical WEREWOLF_DESIGN.md §4 loop
(night then day within a round; the next round opens after day).

The transition is a pure derivation over `GameState` — no RNG, no clock — so
the same input always yields the same next position (invariant #4:
deterministic, replayable).

Win conditions are game-specific and out of scope here. This module defines
only `TerminalCheck`, the seam a game's win condition plugs into, and
`is_terminal`, the helper the game loop (T14) calls. The engine forms no
opinion on what ends a game — Werewolf supplies its check in T13.
"""

from typing import Protocol

from social_deduction_bench.engine.state import GameState, Phase


def advance_phase(state: GameState) -> GameState:
    """Return the next phase position, deriving a new `GameState`.

    NIGHT -> DAY stays in the same round (a round's day follows its own
    night). DAY -> NIGHT opens the next round, incrementing `round`. Pure: the
    input snapshot is untouched, so an earlier recorded position stays intact.
    """
    match state.phase:
        case Phase.NIGHT:
            return state.with_phase(Phase.DAY)
        case Phase.DAY:
            return state.advanced_round().with_phase(Phase.NIGHT)
        case _:
            raise ValueError(f"unhandled phase: {state.phase}")


class TerminalCheck(Protocol):
    """A game's win-condition predicate — the engine's terminal-detection seam.

    Upholds invariant #1: the game-agnostic engine cannot know what ends a
    game, so a game supplies a callable answering "is this position
    terminal?". Werewolf implements this in T13.
    """

    def __call__(self, state: GameState, /) -> bool: ...


def is_terminal(state: GameState, check: TerminalCheck) -> bool:
    """Return whether the game has ended, per the game's `check`.

    A thin named seam so the game loop (T14) reads uniformly as
    `if is_terminal(...)`. The engine relays the game's verdict and forms no
    opinion of its own.
    """
    return check(state)
