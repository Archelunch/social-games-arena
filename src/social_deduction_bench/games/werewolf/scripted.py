"""A `DecisionSource` backed by pre-written per-round actions.

`ScriptedDecisions` plays a fixed script: a list of per-round night actions and
a list of per-round day actions. M2 has no agents, so tests and the T23 smoke
runs drive `run_game` with a script — it lives in `src/` like the determinism
harness because it is production wiring, not a test fixture. A fixed script
ignores the live `GameState`; the `state` argument is accepted to satisfy the
`DecisionSource` Protocol and intentionally unused.
"""

from dataclasses import dataclass, field

from social_deduction_bench.engine import GameState
from social_deduction_bench.games.werewolf.day import DayActions
from social_deduction_bench.games.werewolf.night import NightActions


@dataclass
class ScriptedDecisions:
    """A `DecisionSource` backed by a fixed list of per-round actions.

    Used by tests and the T23 smoke runs; lives in `src/` like the determinism
    harness. The per-round cursors are mutable state, so a fresh instance must
    be built per game run — replay determinism (invariant #4) compares the loop,
    not shared cursors.
    """

    nights: list[NightActions]
    days: list[DayActions]
    _night_cursor: int = field(default=0, init=False)
    _day_cursor: int = field(default=0, init=False)

    def night_actions(self, state: GameState, /) -> NightActions:
        """Return the next scripted night actions; fail loud if the script is exhausted.

        A fixed script ignores `state` — that is intentional. A cursor past the
        end means the game ran longer than the script anticipated, a caller bug.
        """
        if self._night_cursor >= len(self.nights):
            raise RuntimeError(f"scripted decisions: no night actions for round {self._night_cursor + 1}")
        actions = self.nights[self._night_cursor]
        self._night_cursor += 1
        return actions

    def day_actions(self, state: GameState, /) -> DayActions:
        """Return the next scripted day actions; fail loud if the script is exhausted.

        A fixed script ignores `state` — that is intentional. A cursor past the
        end means the game ran longer than the script anticipated, a caller bug.
        """
        if self._day_cursor >= len(self.days):
            raise RuntimeError(f"scripted decisions: no day actions for round {self._day_cursor + 1}")
        actions = self.days[self._day_cursor]
        self._day_cursor += 1
        return actions
