"""`ReActDecisionSource` — a `DecisionSource` backed by per-player ReAct loops.

One `react_decide` loop runs per acting player per phase: at night the
werewolves vote a kill, the seer inspects, the doctor protects; in the day
every alive player votes (or abstains) on the exile. After each phase, the
driver calls `observe` with the events just appended; the adapter routes them
through `observations_for` into each living player's `GameMemory`.

Iteration is roster order — deterministic and replayable for any fixed
LM queue.
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
from social_deduction_bench.games.werewolf.events import (
    BID,
    SPEECH,
    TOOL_REJECTED,
    WEREWOLF_CHAT,
    EventDraft,
)
from social_deduction_bench.games.werewolf.night import NightActions
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.games.werewolf.tools import (
    ToolResult,
    doctor_protect,
    seer_inspect,
    speak,
    submit_bid,
    submit_exile_vote,
    submit_kill_vote,
    werewolf_chat,
)

_EMPTY_INTERMEDIATES: Mapping[str, Callable[..., ToolResult]] = MappingProxyType({})


def _require_str_target(value: object, *, terminal: str, caller: str) -> str:
    """Narrow `Commit.value` to `str` for a target-shaped terminal; raise `TypeError` otherwise.

    `Commit.value: object` is permissive at the loop layer, but `NightActions`
    and `DayActions` carry `str` targets. `assert` is stripped under `-O`, so
    an explicit guard is used.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"agent {caller!r} committed {terminal!r} with non-str value {value!r}; expected a player name",
        )
    return value


def _require_str_message(value: object, *, terminal: str, caller: str) -> str:
    """Narrow `Commit.value` to `str` for a message-shaped terminal (`speak`).

    Mirrors `_require_str_target` but names the offending field `message` so a
    bad commit on a speech terminal surfaces the right noun in the trace.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"agent {caller!r} committed {terminal!r} with non-str value {value!r}; expected a message",
        )
    return value


def _sanitize_arg(value: object) -> object:
    """Coerce a single tool-call argument to a JSON primitive `Event` will accept.

    `pred.next_tool_args` is LLM-derived: it can contain shapes
    `Event.__post_init__` rejects (sets, nested dicts with non-string keys,
    custom objects). Any such value would crash the whole game when the
    `TOOL_REJECTED` draft is logged. Primitives pass through; everything else
    is stringified via `repr` so the audit trail survives without exploding.
    """
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return repr(value)


def _require_int_amount(value: object, *, terminal: str, caller: str) -> int:
    """Narrow `Commit.value` to `int` for an amount-shaped terminal; raise `TypeError` otherwise.

    `submit_bid` carries an integer; an explicit guard surfaces a bad commit
    at the loop layer rather than letting a wrong-typed value reach the
    resolver and crash later with a less helpful trace.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"agent {caller!r} committed {terminal!r} with non-int value {value!r}; expected a bid amount",
        )
    return value


_NIGHT_BRIEFS: Mapping[str, str] = {
    Role.WEREWOLF.value: (
        "It is night. You are a werewolf. You may use werewolf_chat to coordinate "
        "with your fellow werewolves, then commit a kill vote and finish."
    ),
    Role.SEER.value: "It is night. You are the seer. Choose a player to inspect, then finish.",
    Role.DOCTOR.value: "It is night. You are the doctor. Choose a player to protect, then finish.",
}
_DAY_BRIEF = "It is day. Cast your exile vote (or abstain), then finish."
_BID_BRIEF = "It is day. Bid for a speaking slot (0..100); higher bids speak first. Then finish."
_SPEECH_BRIEF = "It is day and you won a speaking slot. Speak to the village, then finish."

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
    """Bind `(state, memory, caller)` into a cognitive tool, exposing only trailing args.

    DSPy's `Tool` derives its schema from `inspect.signature`; the wrapper drops
    the first three params and re-publishes the trailing ones.
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
    """Bind `(state, caller)` into a game-action tool, exposing only trailing args."""
    original_sig = inspect.signature(fn)
    trailing = list(original_sig.parameters.values())[2:]
    new_sig = original_sig.replace(parameters=trailing)

    @functools.wraps(fn)
    def wrapper(**kwargs: object) -> ToolResult:
        return fn(state, caller, **kwargs)

    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    return wrapper


class ReActDecisionSource:
    """A `DecisionSource` that runs one `react_decide` loop per acting player per phase.

    Each roster seat is bound to its own `BaseLM` so a single game can seat a
    mix of models (cross-play). The `lms` mapping must cover the roster exactly:
    one entry per seat name, no extras. Mismatch is rejected at construction
    with a `ValueError` naming the offending names.
    """

    def __init__(
        self,
        *,
        roster: Sequence[tuple[str, str]],
        lms: Mapping[str, BaseLM],
        max_iters: int = 10,
    ) -> None:
        self._roster: tuple[tuple[str, str], ...] = tuple(roster)
        roster_names = {name for name, _ in self._roster}
        lm_names = set(lms.keys())
        missing = roster_names - lm_names
        extra = lm_names - roster_names
        if missing or extra:
            raise ValueError(
                f"lms must cover the roster exactly: missing={sorted(missing)}, extra={sorted(extra)}",
            )
        self._lms: Mapping[str, BaseLM] = MappingProxyType(dict(lms))
        self._max_iters = max_iters
        self._memories: dict[str, GameMemory] = {name: GameMemory() for name, _ in self._roster}
        self._memories_view: Mapping[str, GameMemory] = MappingProxyType(self._memories)
        self._pending_drafts: list[EventDraft] = []
        self._on_reject_by_caller: dict[str, Callable[[str, dict[str, object], str], None]] = {
            name: self._build_on_reject(name) for name, _ in self._roster
        }

    @property
    def memories(self) -> Mapping[str, GameMemory]:
        """Read-only view of per-player memories; mutation raises `TypeError`."""
        return self._memories_view

    @property
    def lms(self) -> Mapping[str, BaseLM]:
        """Read-only view of per-player LM seating; mutation raises `TypeError`."""
        return self._lms

    def observe(self, state: GameState, new_events: tuple[Event, ...], /) -> None:
        """Route each new event to its recipients via `observations_for`.

        Dead players are skipped — their memory will never be consulted again.
        """
        for name, _role in self._roster:
            if not state.is_alive(name):
                continue
            for event in observations_for(new_events, name):
                self._memories[name].record_event(event)

    def drain_drafts(self) -> tuple[EventDraft, ...]:
        """Return and clear staged event drafts (WEREWOLF_CHAT / BID / SPEECH / TOOL_REJECTED).

        The buffer is filled during `night_actions` / `bids` / `speeches` /
        `day_actions` calls when an LLM commits an intermediate action or
        when the engine rejects a tool call. `run_game` drains it after each
        phase and routes the drafts through the same private-event guard
        the resolvers use. Repeated calls after a flush return `()`.
        """
        drained = tuple(self._pending_drafts)
        self._pending_drafts.clear()
        return drained

    def _build_on_reject(self, caller: str) -> Callable[[str, dict[str, object], str], None]:
        """Build the `on_reject` callback that stages a private `TOOL_REJECTED` draft.

        Built once per caller in `__init__` and cached on
        `self._on_reject_by_caller`; `_invoke_react` looks it up by seat
        rather than allocating a fresh closure per ReAct loop.

        The `args` dict comes from the LLM's `next_tool_args` and may contain
        values the event-log's JSON guard rejects (sets, nested dicts with
        non-string keys, etc.). Sanitize each value to a JSON primitive
        before staging — a structurally-invalid payload would crash
        `Event.__post_init__` and abort the game in the middle of a rejection.
        """

        def on_reject(tool: str, args: dict[str, object], reason: str) -> None:
            self._pending_drafts.append(
                EventDraft(
                    type=TOOL_REJECTED,
                    payload={
                        "tool": tool,
                        "args": {key: _sanitize_arg(value) for key, value in args.items()},
                        "reason": reason,
                    },
                    recipients=(caller,),
                )
            )

        return on_reject

    def _bind_werewolf_chat(
        self, state: GameState, caller: str, living_pack: tuple[str, ...]
    ) -> Callable[..., ToolResult]:
        """Bind `werewolf_chat` so a valid call stages a `WEREWOLF_CHAT` draft.

        The recipients are the *currently-living* werewolf pack (computed once
        per night and captured in the closure). The wrapper exposes only the
        trailing `message` param to DSPy's `Tool` schema.
        """
        original_sig = inspect.signature(werewolf_chat)
        trailing = list(original_sig.parameters.values())[2:]
        new_sig = original_sig.replace(parameters=trailing)
        pending = self._pending_drafts

        @functools.wraps(werewolf_chat)
        def wrapper(**kwargs: object) -> ToolResult:
            message = kwargs.get("message", "")
            if not isinstance(message, str):
                return ToolResult(
                    valid=False,
                    reason=f"tool 'werewolf_chat' requires a str message, got {type(message).__name__}",
                )
            result = werewolf_chat(state, caller, message)
            if result.valid:
                pending.append(
                    EventDraft(
                        type=WEREWOLF_CHAT,
                        payload={"speaker": caller, "message": message},
                        recipients=living_pack,
                    )
                )
            return result

        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
        return wrapper

    def night_actions(self, state: GameState, /) -> NightActions:
        """Run one ReAct loop per acting living player; aggregate to `NightActions`.

        Werewolves contribute kill votes; the seer (if alive) contributes one
        inspect; the doctor (if alive) contributes one protect; villagers do
        nothing at night.
        """
        if state.phase is not Phase.NIGHT:
            raise ValueError(f"night_actions requires the night phase, got {state.phase.value}")

        # Pack identity is fixed for the night (no kills resolve until `resolve_night`),
        # so compute it once and reuse for every wolf's chat intermediate.
        living_pack = tuple(sorted(p.name for p in state.alive_players() if p.role == Role.WEREWOLF.value))

        kill_votes: dict[str, str] = {}
        seer_target: str | None = None
        doctor_target: str | None = None

        for name, role in self._roster:
            if not state.is_alive(name):
                continue
            if role not in _NIGHT_TERMINAL_BY_ROLE:
                continue

            commit = self._run_one(state, name, role, _NIGHT_BRIEFS[role], living_pack=living_pack)
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

    def bids(self, state: GameState, /) -> dict[str, int]:
        """Run one `submit_bid` loop per alive player; stage a private `BID` draft per commit."""
        if state.phase is not Phase.DAY:
            raise ValueError(f"bids requires the day phase, got {state.phase.value}")

        bids_map: dict[str, int] = {}
        for name, _role in self._roster:
            if not state.is_alive(name):
                continue
            commit = self._invoke_react(
                state=state,
                caller=name,
                decision_brief=_BID_BRIEF,
                terminals={"submit_bid": _bind_terminal(submit_bid, state, name)},
            )
            amount = _require_int_amount(commit.value, terminal=commit.tool, caller=name)
            bids_map[name] = amount
            self._pending_drafts.append(
                EventDraft(
                    type=BID,
                    payload={"bidder": name, "amount": amount},
                    recipients=(name,),
                )
            )
        return bids_map

    def speeches(self, state: GameState, speakers: tuple[str, ...], /) -> tuple[tuple[str, str], ...]:
        """Run one `speak` loop per speaker in order; stage a public `SPEECH` draft per commit."""
        if state.phase is not Phase.DAY:
            raise ValueError(f"speeches requires the day phase, got {state.phase.value}")

        out: list[tuple[str, str]] = []
        for speaker in speakers:
            if not state.is_alive(speaker):
                raise ValueError(f"speeches received a dead speaker: {speaker!r}")
            commit = self._invoke_react(
                state=state,
                caller=speaker,
                decision_brief=_SPEECH_BRIEF,
                terminals={"speak": _bind_terminal(speak, state, speaker)},
            )
            message = _require_str_message(commit.value, terminal=commit.tool, caller=speaker)
            out.append((speaker, message))
            self._pending_drafts.append(
                EventDraft(
                    type=SPEECH,
                    payload={"speaker": speaker, "message": message},
                    recipients=(),
                )
            )
        return tuple(out)

    def _run_one(
        self,
        state: GameState,
        caller: str,
        role: str,
        decision_brief: str,
        *,
        living_pack: tuple[str, ...] = (),
    ) -> Commit:
        """Wire one player's tools and drive a single `react_decide` loop.

        `living_pack` is the precomputed sorted living werewolf pack — passed
        in for night werewolf turns so it is computed once per night instead
        of once per wolf. The pack is identical for every wolf at the start
        of one night.
        """
        if state.phase is Phase.NIGHT:
            terminal_name, terminal_fn = _NIGHT_TERMINAL_BY_ROLE[role]
        else:
            terminal_name, terminal_fn = "submit_exile_vote", submit_exile_vote

        terminals = {terminal_name: _bind_terminal(terminal_fn, state, caller)}

        intermediates: Mapping[str, Callable[..., ToolResult]] = _EMPTY_INTERMEDIATES
        if state.phase is Phase.NIGHT and role == Role.WEREWOLF.value:
            intermediates = {"werewolf_chat": self._bind_werewolf_chat(state, caller, living_pack)}

        return self._invoke_react(
            state=state,
            caller=caller,
            decision_brief=decision_brief,
            terminals=terminals,
            intermediates=intermediates,
        )

    def _invoke_react(
        self,
        *,
        state: GameState,
        caller: str,
        decision_brief: str,
        terminals: Mapping[str, Callable[..., ToolResult]],
        intermediates: Mapping[str, Callable[..., ToolResult]] = _EMPTY_INTERMEDIATES,
    ) -> Commit:
        """Wire one player's cognitive tools + reject callback and run `react_decide`."""
        memory = self._memories[caller]
        cognitive = [_bind_cognitive(fn, state, memory, caller) for fn in _COGNITIVE_TOOLS]
        return react_decide(
            caller=caller,
            cognitive_tools=cognitive,
            intermediate_tools=intermediates,
            terminal_tools=terminals,
            decision_brief=decision_brief,
            lm=self._lms[caller],
            max_iters=self._max_iters,
            on_reject=self._on_reject_by_caller[caller],
        )
