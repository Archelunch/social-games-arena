"""`ReActDecisionSource` — the T22 `DecisionSource` adapter.

Wires a per-player `react_decide` loop into the `run_game` driver. One ReAct
loop runs per acting player per phase: at night the werewolves vote a kill, the
seer inspects, the doctor protects; in the day every alive player votes (or
abstains) on the exile. After each phase, the driver calls `observe` with the
events just appended; the adapter routes them through `observations_for` into
each player's `GameMemory` (invariant #2 — the only path from the engine into
an agent's memory).

The adapter walks alive players in *roster* order, not role-grouped or
alphabetical order, so identical `(roster, scripted-LM-queue)` inputs replay to
identical decisions (invariant #4).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from dspy.clients.base_lm import BaseLM

from social_deduction_bench.agents.cognitive import (
    get_beliefs,
    get_plan,
    recall,
    remember,
    set_belief,
    set_plan,
)
from social_deduction_bench.agents.memory import GameMemory
from social_deduction_bench.agents.react import Commit, react_decide
from social_deduction_bench.engine import Event, GameState, Phase, observations_for
from social_deduction_bench.games.werewolf.cognitive import get_private_info, get_public_state
from social_deduction_bench.games.werewolf.day import DayActions
from social_deduction_bench.games.werewolf.night import NightActions
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.games.werewolf.tools import (
    ToolResult,
    doctor_protect,
    seer_inspect,
    submit_exile_vote,
    submit_kill_vote,
)


def _require_str_target(value: object, *, terminal: str, caller: str) -> str:
    """Narrow `Commit.value` to `str` for a target-shaped terminal; fail loud otherwise.

    Why: `Commit.value: object` is intentionally permissive at the loop layer,
    but `NightActions` / `DayActions` carry `str` targets. A non-string here
    would corrupt resolver behavior. Plain `assert` is stripped under `-O`, so
    use an explicit guard.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"agent {caller!r} committed {terminal!r} with non-str value {value!r}; expected a player name",
        )
    return value


_NIGHT_BRIEFS: Mapping[str, str] = {
    Role.WEREWOLF.value: "It is night. You are a werewolf. Choose a player to kill, then finish.",
    Role.SEER.value: "It is night. You are the seer. Choose a player to inspect, then finish.",
    Role.DOCTOR.value: "It is night. You are the doctor. Choose a player to protect, then finish.",
}
_DAY_BRIEF = "It is day. Cast your exile vote (or abstain), then finish."

_NIGHT_TERMINAL_BY_ROLE: Mapping[str, tuple[str, Callable[..., ToolResult]]] = {
    Role.WEREWOLF.value: ("submit_kill_vote", submit_kill_vote),
    Role.SEER.value: ("seer_inspect", seer_inspect),
    Role.DOCTOR.value: ("doctor_protect", doctor_protect),
}

_COGNITIVE_TOOLS: tuple[Callable[..., str], ...] = (
    recall,
    remember,
    get_beliefs,
    set_belief,
    get_plan,
    set_plan,
    get_public_state,
    get_private_info,
)


def _bind_cognitive(fn: Callable[..., str], state: GameState, memory: GameMemory, caller: str) -> Callable[..., str]:
    """Bind `(state, memory, caller)` into a cognitive tool, hiding them from the LLM-facing signature.

    DSPy's `Tool` derives its schema from `inspect.signature`; the wrapper drops
    the first three params and re-publishes the trailing ones so the LLM only
    sees the call-time arguments. `__name__` and `__doc__` are inherited so the
    LLM sees the original tool name and docstring.
    """
    original_sig = inspect.signature(fn)
    trailing = list(original_sig.parameters.values())[3:]
    new_sig = original_sig.replace(parameters=trailing)

    @functools.wraps(fn)
    def wrapper(**kwargs: object) -> str:
        return fn(state, memory, caller, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    return wrapper


def _bind_terminal(fn: Callable[..., ToolResult], state: GameState, caller: str) -> Callable[..., ToolResult]:
    """Bind `(state, caller)` into a game-action tool; the LLM sees only the trailing args."""
    original_sig = inspect.signature(fn)
    trailing = list(original_sig.parameters.values())[2:]
    new_sig = original_sig.replace(parameters=trailing)

    @functools.wraps(fn)
    def wrapper(**kwargs: object) -> ToolResult:
        return fn(state, caller, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    return wrapper


class ReActDecisionSource:
    """A `DecisionSource` that drives one `react_decide` loop per acting player per phase.

    Owns one `GameMemory` per roster name; routes engine events into the right
    memory via `observations_for` (invariant #2); aggregates per-player commits
    into `NightActions` / `DayActions` in roster order (invariant #4).
    """

    def __init__(
        self,
        *,
        roster: Sequence[tuple[str, str]],
        lm: BaseLM,
        max_iters: int = 10,
    ) -> None:
        self._roster: tuple[tuple[str, str], ...] = tuple(roster)
        self._lm = lm
        self._max_iters = max_iters
        self._memories: dict[str, GameMemory] = {name: GameMemory() for name, _ in self._roster}
        self._memories_view: Mapping[str, GameMemory] = MappingProxyType(self._memories)

    @property
    def memories(self) -> Mapping[str, GameMemory]:
        """Read-only view of per-player memories; mutation raises `TypeError`."""
        return self._memories_view

    def observe(self, state: GameState, new_events: tuple[Event, ...], /) -> None:
        """Route each new event to its recipients via `observations_for` (invariant #2).

        The driver hands a sliced view of just-appended events on each call;
        re-pushing prior events would corrupt per-player history, so we trust
        the slice and route only what arrived. Dead players are skipped — they
        will never be asked for an action again, so growing their memory is
        wasted work and confuses post-game inspection.
        """
        for name, _role in self._roster:
            if not state.is_alive(name):
                continue
            for event in observations_for(new_events, name):
                self._memories[name].record_event(event)

    def night_actions(self, state: GameState, /) -> NightActions:
        """Run one ReAct loop per acting living player; aggregate to `NightActions`.

        Walks roster order. Werewolves contribute kill votes; the seer (if
        alive) contributes one inspect; the doctor (if alive) contributes one
        protect; villagers do nothing at night.
        """
        if state.phase is not Phase.NIGHT:
            raise ValueError(f"night_actions requires the night phase, got {state.phase.value}")

        kill_votes: dict[str, str] = {}
        seer_target: str | None = None
        doctor_target: str | None = None

        for name, role in self._roster:
            if not state.is_alive(name):
                continue
            if role not in _NIGHT_TERMINAL_BY_ROLE:
                continue

            commit = self._run_one(state, name, role, _NIGHT_BRIEFS[role])
            value = _require_str_target(commit.value, terminal=commit.tool, caller=name)

            if role == Role.WEREWOLF.value:
                kill_votes[name] = value
            elif role == Role.SEER.value:
                seer_target = value
            elif role == Role.DOCTOR.value:
                doctor_target = value

        return NightActions(kill_votes=kill_votes, seer_inspect=seer_target, doctor_protect=doctor_target)

    def day_actions(self, state: GameState, /) -> DayActions:
        """Run one `submit_exile_vote` loop per alive player in roster order."""
        if state.phase is not Phase.DAY:
            raise ValueError(f"day_actions requires the day phase, got {state.phase.value}")

        exile_votes: dict[str, str] = {}
        for name, role in self._roster:
            if not state.is_alive(name):
                continue
            commit = self._run_one(state, name, role, _DAY_BRIEF)
            exile_votes[name] = _require_str_target(commit.value, terminal=commit.tool, caller=name)

        return DayActions(exile_votes=exile_votes)

    def _run_one(self, state: GameState, caller: str, role: str, decision_brief: str) -> Commit:
        """Wire one player's tools and drive a single `react_decide` loop."""
        memory = self._memories[caller]
        cognitive = [_bind_cognitive(fn, state, memory, caller) for fn in _COGNITIVE_TOOLS]

        if state.phase is Phase.NIGHT:
            terminal_name, terminal_fn = _NIGHT_TERMINAL_BY_ROLE[role]
        else:
            terminal_name, terminal_fn = "submit_exile_vote", submit_exile_vote

        terminals = {terminal_name: _bind_terminal(terminal_fn, state, caller)}

        return react_decide(
            caller=caller,
            cognitive_tools=cognitive,
            terminal_tools=terminals,
            decision_brief=decision_brief,
            lm=self._lm,
            max_iters=self._max_iters,
        )
