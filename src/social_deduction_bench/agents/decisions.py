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

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from dspy.clients.base_lm import BaseLM

from social_deduction_bench.agents.cognitive import (
    recall,
    set_belief,
    set_plan,
)
from social_deduction_bench.agents.memory import GameMemory
from social_deduction_bench.agents.react import Commit, react_decide_async
from social_deduction_bench.agents.trajectory import (
    LMCallRecord,
    ReActStep,
    Trajectory,
    sanitize_arg,
)
from social_deduction_bench.engine import Event, GameState, Phase, observations_for
from social_deduction_bench.games.werewolf.day import DayActions
from social_deduction_bench.games.werewolf.events import (
    ACCUSATION,
    BID,
    DEFENSE,
    SPEECH,
    TOOL_REJECTED,
    WEREWOLF_CHAT,
    EventDraft,
)
from social_deduction_bench.games.werewolf.humanize import describe_events
from social_deduction_bench.games.werewolf.night import NightActions
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.games.werewolf.tools import (
    ACCUSE,
    DEFEND,
    PASS_TURN,
    Reaction,
    ToolResult,
    accuse,
    defend,
    doctor_protect,
    pass_turn,
    seer_inspect,
    speak,
    submit_bid,
    submit_exile_vote,
    submit_kill_vote,
    werewolf_chat,
)

logger = logging.getLogger(__name__)

_EMPTY_INTERMEDIATES: Mapping[str, Callable[..., ToolResult]] = MappingProxyType({})


def _build_tool_rejected_draft(caller: str, tool: str, args: dict[str, object], reason: str) -> EventDraft:
    """Build a private `TOOL_REJECTED` draft for `caller`'s rejected tool call.

    Sanitizes each arg value because the LLM may return shapes the event-log
    JSON guard rejects (sets, non-string-keyed dicts). Shared by the
    sequential `self._pending_drafts` path and the async per-coroutine path.
    """
    return EventDraft(
        type=TOOL_REJECTED,
        payload={
            "tool": tool,
            "args": {key: sanitize_arg(value) for key, value in args.items()},
            "reason": reason,
        },
        recipients=(caller,),
    )


@dataclass(frozen=True, slots=True)
class _DecisionResult:
    """One acting seat's gathered result, returned to the phase aggregator.

    Carries everything the aggregator needs to record the decision in roster
    order: the commit, the trajectory's structured steps + telemetry, the
    per-coroutine drafts buffer (`werewolf_chat`, `tool_rejected` entries
    that fired during this loop), and the seat's role for fan-out routing.
    The aggregator assigns `decision_seq`, appends to `self._trajectories`,
    and extends `self._pending_drafts` strictly in argument order so wall-
    clock completion order never leaks into the recorded transcript.
    """

    caller: str
    role: str
    commit: Commit
    steps: tuple[ReActStep, ...]
    lm_calls: tuple[LMCallRecord, ...]
    drafts: tuple[EventDraft, ...]


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


# Brief templates. The `{you}` and `{alive_csv}` placeholders are filled in
# at decision time so the model sees its own name and the current living
# roster in the first message — without that grounding, small models often
# hallucinate target names ("submit_kill_vote(target=Emma)" when Emma is
# not a player) instead of reading the live state. `_format_brief` also
# appends a per-caller context block (pack identity for wolves, plus the
# agent's plan / suspicions / recent events) so agents stop spending whole
# ReAct iterations on read-only cognitive calls to learn what could have
# been handed to them in the first message.
_NIGHT_BRIEFS: Mapping[str, str] = {
    Role.WEREWOLF.value: (
        "It is night. You are {you}, a werewolf. Living players: {alive_csv}. "
        "Commit your kill via `submit_kill_vote`; that ends your turn."
    ),
    Role.SEER.value: (
        "It is night. You are {you}, the seer. Living players: {alive_csv}. "
        "Inspect one player via `seer_inspect`; that ends your turn."
    ),
    Role.DOCTOR.value: (
        "It is night. You are {you}, the doctor. Living players: {alive_csv}. "
        "Protect one player via `doctor_protect`; that ends your turn."
    ),
}
_NIGHT_CHAT_BRIEF = (
    "It is night. You are {you}, a werewolf. Living players: {alive_csv}. "
    "Discuss tonight's kill with your pack: send one message via `werewolf_chat`; that ends your turn."
)
_DAY_BRIEF = (
    "It is day. The statements and reactions are done. You are {you}. Living players: {alive_csv}. "
    "Record who you now suspect with `set_belief` if you have a read, then vote to exile a living player "
    "(or `abstain`) via `submit_exile_vote`; that ends your turn."
)
_BID_BRIEF = (
    "It is day. You are {you}. Living players: {alive_csv}. "
    "Bid for a speaking slot (0 to your remaining speaking budget; higher bids speak first) "
    "via `submit_bid`; that ends your turn."
)
_SPEECH_BRIEF = (
    "It is day and you won a speaking slot. You are {you}. Living players: {alive_csv}. "
    "Address the village via `speak`; that ends your turn."
)
_REACTION_BRIEF = (
    "It is day. The statements are done and now every player reacts in turn. You are {you}. "
    "Living players: {alive_csv}. React in one short sentence: accuse a living player via `accuse`, "
    "defend a living player via `defend` (you may defend yourself), or stay silent via `pass_turn`. "
    "That ends your turn."
)

# How many recent rounds of events to inline in the brief's memory block.
# Deeper history stays accessible through the `recall` cognitive tool.
_BRIEF_RECALL_ROUNDS = 2

# Appended to briefs whose terminal supports note-on-commit (every action except
# the night-chat sub-phase, whose commit is the message itself).
_NOTE_HINT = (
    "You may add a short private `note` argument to your action to record reasoning for later rounds; omit it to skip."
)

# Appended to every brief. The cognitive tools are always in the toolbelt but
# were never named in the brief, so small models could not tell they exist or —
# worse — whether calling one ends the turn (a model once spent ~60s unsure).
# Naming them and stating they are optional + non-terminal removes that confusion
# and is what finally gets `set_belief` / `set_plan` used (the suspicion-accuracy
# signal depends on `set_belief` being written).
_COGNITIVE_HINT = (
    "Optional thinking tools (these do NOT end your turn — only the game action above does): "
    "`recall` reviews earlier rounds, `set_belief` records who you suspect, `set_plan` notes your strategy."
)

# How a player's own role reads in the brief's identity line. The seer/doctor are
# unique ("the"); werewolf/villager are not ("a"). Rendered only into the caller's
# OWN brief, so naming the role here leaks nothing (invariant #2). The day briefs
# (bid/speak/react/vote) never restated the role, so a wolf mid-day lost track and
# claimed "I am a villager" — this line keeps identity + role present every turn.
_ROLE_PHRASE: Mapping[str, str] = MappingProxyType(
    {
        Role.WEREWOLF.value: "a werewolf",
        Role.SEER.value: "the seer",
        Role.DOCTOR.value: "the doctor",
        Role.VILLAGER.value: "a villager",
    }
)


def _render_context(caller: str, role: str, state: GameState, memory: GameMemory) -> str:
    """Render the "What you know" block appended to the caller's brief.

    Built only from information the caller is entitled to: its own identity and
    role, (for a werewolf) its living allies, plus its own `GameMemory` (plan,
    suspicions, recent events). The identity line restates "you are X, <role>"
    every turn (the day briefs otherwise dropped the role). The ally line is gated
    on the caller being a werewolf and excludes the caller's own name, so a non-
    wolf brief never enumerates the wolves and a wolf does not see its own name
    beside its packmate's (which caused name/self confusion) — no hidden state
    leaks because the block is injected only into the caller's own brief.
    Pre-rendering this is what lets the toolbelt drop the read-only cognitive
    tools. Recent events are rendered in plain language by `describe_events` (so a
    night-kill reads differently from a day-exile) rather than as raw `recall` JSON.
    """
    role_phrase = _ROLE_PHRASE.get(role, f"a {role}")
    lines: list[str] = ["What you know:", f"- You are {caller}, {role_phrase}."]

    if role == Role.WEREWOLF.value:
        allies = sorted(p.name for p in state.alive_players() if p.role == Role.WEREWOLF.value and p.name != caller)
        if allies:
            ally_word = "ally" if len(allies) == 1 else "allies"
            lines.append(f"- Your werewolf {ally_word} (besides you): {', '.join(allies)}.")
        else:
            lines.append("- You are the only living werewolf — no allies remain.")

    # Public rules every player is entitled to know — not strategy. Stating the
    # win conditions and which channels are public fixes an information
    # deficiency (a wolf once outed itself in a public speech, not realising
    # `speak` is broadcast); whether a player acts on it is what we measure.
    lines.append(
        "- Win conditions: the village wins when every werewolf is exiled; "
        "the werewolves win when the werewolves equal or outnumber the other living players."
    )
    channels = "- Channels: `speak` and all votes are public — every living player sees them."
    if role == Role.WEREWOLF.value:
        channels += " `werewolf_chat` is private to your pack only."
    lines.append(channels)

    # Night-resolution mechanics — a public rule, not strategy. Fixes a concrete
    # misread: a doctor's "I protected X" claim was treated by the whole table
    # (and the doctor itself) as impossible "because only Y died", so the village
    # mis-exiled its own doctor. Guarding a player the wolves did not target is
    # normal and leaves no public trace — stating that stops the false-tell.
    lines.append(
        "- Night mechanics: the werewolves kill one player; the seer privately learns one player's faction; "
        "the doctor guards one player, who survives a werewolf attack only if guarded that very night. "
        "Guarding someone the werewolves did not target is normal and has no visible effect — it does NOT "
        "mean that player was attacked."
    )

    # Every player's remaining speaking-bid budget — surfaced on DAY briefs only,
    # the only phase where bids are spent (at night no one bids, so the line was
    # pure noise). This is PUBLIC information: each day's bids are broadcast in
    # `DISCUSSION_RESOLVED`, so the running spend (and thus the remaining budget)
    # of every player is common knowledge; it leaks nothing (invariant #2). The
    # label is explicit that this is a *bidding* resource for winning statement
    # slots, not a turn the reader can take now — a small model once read its
    # remaining budget as "I can still speak".
    if state.phase is Phase.DAY:
        budgets = ", ".join(f"{p.name} {p.bid_budget}" for p in state.alive_players())
        lines.append(f"- Speaking-bid budget remaining (public; lower = bid more for statement slots): {budgets}.")

    plan = memory.plan
    lines.append(f"- Plan: {plan}" if plan else "- Plan: none yet.")

    beliefs = memory.beliefs
    if beliefs:
        rendered = "; ".join(f"{player}={b.guess} ({b.confidence})" for player, b in sorted(beliefs.items()))
        lines.append(f"- Suspicions: {rendered}.")
    else:
        lines.append("- Suspicions: none yet.")

    recent = describe_events(memory.events, last_n_rounds=_BRIEF_RECALL_ROUNDS, caller=caller)
    if recent:
        indented = "\n".join(f"    {line}" for line in recent.splitlines())
        lines.append(f"- Recent events:\n{indented}")
    else:
        lines.append("- Recent events: nothing yet.")

    return "\n".join(lines)


def _format_brief(
    template: str, *, caller: str, role: str, state: GameState, memory: GameMemory, note_hint: bool = True
) -> str:
    """Fill `{you}` + `{alive_csv}` in a brief template and append the context block.

    Bound here (not at module load) so the alive list and the memory block
    reflect the current `GameState` / `GameMemory` — players die between
    rounds and memory grows each phase, so both must be read at decision
    time, not captured earlier. `note_hint` appends the note-on-commit hint
    for action briefs; the night-chat sub-phase passes `False` because its
    terminal (`werewolf_chat`) takes no `note`.
    """
    base = template.format(you=caller, alive_csv=", ".join(state.alive_names()))
    context = _render_context(caller, role, state, memory)
    brief = f"{base}\n\n{context}"
    if note_hint:
        brief = f"{brief}\n\n{_NOTE_HINT}"
    # The cognitive hint applies to every loop — the cognitive tools are always
    # bound, including the night-chat sub-phase (note_hint=False there).
    return f"{brief}\n\n{_COGNITIVE_HINT}"


_NIGHT_TERMINAL_BY_ROLE: Mapping[str, tuple[str, Callable[..., ToolResult]]] = {
    Role.WEREWOLF.value: ("submit_kill_vote", submit_kill_vote),
    Role.SEER.value: ("seer_inspect", seer_inspect),
    Role.DOCTOR.value: ("doctor_protect", doctor_protect),
}

# The day reaction round's terminals: every living player commits exactly one of
# these. `accuse` / `defend` stage a public event; `pass_turn` stages nothing.
_REACTION_TERMINALS: Mapping[str, Callable[..., ToolResult]] = MappingProxyType(
    {ACCUSE: accuse, DEFEND: defend, PASS_TURN: pass_turn}
)

# Cognitive tools exposed to every decision loop. The read-only tools
# (`get_beliefs`, `get_plan`, `get_private_info`, `get_public_state`) are
# intentionally NOT here — their content is pre-rendered into the brief by
# `_render_context`, so offering them only bloated the per-iter prompt and
# invited wrong-arg failures (e.g. `get_private_info(target=…)`). The free-text
# `remember` tool is gone too: it cost a whole extra ReAct iteration, and the
# terminal action now carries an optional `note` that records the same thing in
# the committing call (see `_bind_terminal`). The structured-write tools plus
# `recall` (full-history dives beyond the brief window) remain.
_COGNITIVE_TOOLS: tuple[Callable[..., str], ...] = (
    recall,
    set_belief,
    set_plan,
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


def _bind_terminal(
    fn: Callable[..., ToolResult], state: GameState, caller: str, memory: GameMemory
) -> Callable[..., ToolResult]:
    """Bind `(state, caller)` into a game-action tool, exposing trailing args + an optional `note`.

    The published signature gains a keyword-only `note: str = ""`. On a *valid*
    commit with a non-blank note, the note is written to `memory` at the current
    round — folding "record my reasoning + act" into one call instead of a
    separate `remember` iteration. A rejected call records nothing (the action
    never happened), and an absent / blank note records nothing. This enables,
    but never forces, single-call record-and-act; structured `set_belief` /
    `set_plan` stay available for agents that want them.
    """
    original_sig = inspect.signature(fn)
    trailing = list(original_sig.parameters.values())[2:]
    note_param = inspect.Parameter("note", inspect.Parameter.KEYWORD_ONLY, default="", annotation=str)
    new_sig = original_sig.replace(parameters=[*trailing, note_param])

    @functools.wraps(fn)
    def wrapper(**kwargs: object) -> ToolResult:
        note = kwargs.pop("note", "")
        result = fn(state, caller, **kwargs)
        if result.valid and isinstance(note, str) and note.strip():
            memory.remember(note, state.round)
        return result

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
        on_trajectory: Callable[[Trajectory], None] | None = None,
        on_decision_start: Callable[[str, str], None] | None = None,
        on_step: Callable[[str, str, ReActStep, tuple[LMCallRecord, ...]], None] | None = None,
        on_thought_chunk: Callable[[str, str, int, str], None] | None = None,
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
        self._role_by_name: Mapping[str, str] = MappingProxyType({name: role for name, role in self._roster})
        self._trajectories: list[Trajectory] = []
        self._decision_seq: int = 0
        self._on_trajectory: Callable[[Trajectory], None] | None = on_trajectory
        self._on_decision_start: Callable[[str, str], None] | None = on_decision_start
        self._on_step: Callable[[str, str, ReActStep, tuple[LMCallRecord, ...]], None] | None = on_step
        self._on_thought_chunk: Callable[[str, str, int, str], None] | None = on_thought_chunk

    @property
    def memories(self) -> Mapping[str, GameMemory]:
        """Read-only view of per-player memories; mutation raises `TypeError`."""
        return self._memories_view

    @property
    def lms(self) -> Mapping[str, BaseLM]:
        """Read-only view of per-player LM seating; mutation raises `TypeError`."""
        return self._lms

    @property
    def trajectories(self) -> tuple[Trajectory, ...]:
        """Read-only snapshot of accumulated per-decision trajectories (T30 sidecar).

        Each entry is one decision-point ReAct loop's record: thought trace,
        tool args, observations, plus per-LM-call telemetry. Decision order
        matches `decision_seq` (gap-free starting at 0). The returned tuple
        is a fresh snapshot — later loops grow the source's internal list
        but not the caller's snapshot.
        """
        return tuple(self._trajectories)

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
        `self._on_reject_by_caller`; the sequential code paths that still
        stage drafts directly (e.g. legacy entry points) look it up by seat
        rather than allocating a fresh closure per ReAct loop. The async
        fan-out path uses `_build_local_on_reject` instead so concurrent
        seats don't race on `self._pending_drafts`.

        The `args` dict comes from the LLM's `next_tool_args` and may contain
        values the event-log's JSON guard rejects (sets, nested dicts with
        non-string keys, etc.). Sanitize each value to a JSON primitive
        before staging — a structurally-invalid payload would crash
        `Event.__post_init__` and abort the game in the middle of a rejection.
        """

        def on_reject(tool: str, args: dict[str, object], reason: str) -> None:
            self._pending_drafts.append(_build_tool_rejected_draft(caller, tool, args, reason))

        return on_reject

    def _build_local_on_reject(
        self, caller: str, drafts: list[EventDraft]
    ) -> Callable[[str, dict[str, object], str], None]:
        """`on_reject` that appends to a per-coroutine local buffer.

        Used by the async fan-out so concurrent seats stage their
        `TOOL_REJECTED` drafts into private lists. The aggregator concatenates
        the lists in argument (= roster) order after `gather` resolves.
        """

        def on_reject(tool: str, args: dict[str, object], reason: str) -> None:
            drafts.append(_build_tool_rejected_draft(caller, tool, args, reason))

        return on_reject

    def _bind_werewolf_chat_local(
        self,
        state: GameState,
        caller: str,
        living_pack: tuple[str, ...],
        drafts: list[EventDraft],
    ) -> Callable[..., ToolResult]:
        """Bind `werewolf_chat` so a valid call stages a `WEREWOLF_CHAT` draft
        into a per-coroutine `drafts` list, not `self._pending_drafts`.

        The recipients are the *currently-living* werewolf pack (computed once
        per night and captured in the closure). The async fan-out passes a
        private list so wolves running in parallel don't race; the aggregator
        concatenates the lists in argument order.
        """
        original_sig = inspect.signature(werewolf_chat)
        trailing = list(original_sig.parameters.values())[2:]
        new_sig = original_sig.replace(parameters=trailing)

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
                drafts.append(
                    EventDraft(
                        type=WEREWOLF_CHAT,
                        payload={"speaker": caller, "message": message},
                        recipients=living_pack,
                    )
                )
            return result

        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
        return wrapper

    def night_chat(self, state: GameState, /) -> None:
        """Run one chat loop per living wolf — the first half of the two-phase night.

        Each wolf commits one `werewolf_chat` message (the message is the
        loop's commit). The drafts are drained and observed by the engine
        driver before `night_actions`, so a wolf reads its packmate's message
        before choosing a kill target. Non-wolves do nothing in this
        sub-phase. Loops fan out over `asyncio.gather` and reduce in roster
        order (determinism #4).

        A lone surviving wolf has no packmate to coordinate with, so the chat
        sub-phase is skipped entirely (no ReAct loop, no LM call, no draft) —
        coordination only earns its keep with two or more living wolves.
        """
        if state.phase is not Phase.NIGHT:
            raise ValueError(f"night_chat requires the night phase, got {state.phase.value}")
        living_wolves = [name for name, role in self._roster if role == Role.WEREWOLF.value and state.is_alive(name)]
        if len(living_wolves) < 2:
            return
        asyncio.run(self._gather_night_chat(state))

    async def _gather_night_chat(self, state: GameState) -> None:
        living_pack = tuple(sorted(p.name for p in state.alive_players() if p.role == Role.WEREWOLF.value))
        acting: list[tuple[str, str]] = [
            (name, role) for name, role in self._roster if state.is_alive(name) and role == Role.WEREWOLF.value
        ]
        tasks = [
            self._invoke_react_async(
                state=state,
                caller=name,
                role=role,
                decision_brief=_format_brief(
                    _NIGHT_CHAT_BRIEF, caller=name, role=role, state=state, memory=self._memories[name], note_hint=False
                ),
                terminal_name="werewolf_chat",
                terminal_fn=werewolf_chat,
                living_pack=living_pack,
                chat_terminal=True,
            )
            for name, role in acting
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            self._record_result(state, result)

    def night_actions(self, state: GameState, /) -> NightActions:
        """Run one ReAct loop per acting living player; aggregate to `NightActions`.

        Werewolves, seer, and doctor each decide from the same start-of-night
        state snapshot — there is no intra-phase observation that would force
        sequencing — so the four loops fan out over `asyncio.gather` for I/O
        concurrency. Results are reduced in roster order so determinism
        (invariant #4) is preserved regardless of completion order.
        """
        if state.phase is not Phase.NIGHT:
            raise ValueError(f"night_actions requires the night phase, got {state.phase.value}")
        return asyncio.run(self._gather_night(state))

    async def _gather_night(self, state: GameState) -> NightActions:
        living_pack = tuple(sorted(p.name for p in state.alive_players() if p.role == Role.WEREWOLF.value))
        acting: list[tuple[str, str]] = [
            (name, role) for name, role in self._roster if state.is_alive(name) and role in _NIGHT_TERMINAL_BY_ROLE
        ]
        tasks = [
            self._invoke_react_async(
                state=state,
                caller=name,
                role=role,
                decision_brief=_format_brief(
                    _NIGHT_BRIEFS[role], caller=name, role=role, state=state, memory=self._memories[name]
                ),
                terminal_name=_NIGHT_TERMINAL_BY_ROLE[role][0],
                terminal_fn=_NIGHT_TERMINAL_BY_ROLE[role][1],
                living_pack=living_pack,
            )
            for name, role in acting
        ]
        results = await asyncio.gather(*tasks)

        kill_votes: dict[str, str] = {}
        seer_target: str | None = None
        doctor_target: str | None = None
        for result in results:
            self._record_result(state, result)
            value = _require_str_target(result.commit.value, terminal=result.commit.tool, caller=result.caller)
            if result.role == Role.WEREWOLF.value:
                kill_votes[result.caller] = value
            elif result.role == Role.SEER.value:
                seer_target = value
            elif result.role == Role.DOCTOR.value:
                doctor_target = value
        return NightActions(kill_votes=kill_votes, seer_inspect=seer_target, doctor_protect=doctor_target)

    def day_actions(self, state: GameState, /) -> DayActions:
        """Fan-out one `submit_exile_vote` loop per alive player; reduce in roster order."""
        if state.phase is not Phase.DAY:
            raise ValueError(f"day_actions requires the day phase, got {state.phase.value}")
        return asyncio.run(self._gather_day_actions(state))

    async def _gather_day_actions(self, state: GameState) -> DayActions:
        acting: list[tuple[str, str]] = [(name, role) for name, role in self._roster if state.is_alive(name)]
        tasks = [
            self._invoke_react_async(
                state=state,
                caller=name,
                role=role,
                decision_brief=_format_brief(
                    _DAY_BRIEF, caller=name, role=role, state=state, memory=self._memories[name]
                ),
                terminal_name="submit_exile_vote",
                terminal_fn=submit_exile_vote,
            )
            for name, role in acting
        ]
        results = await asyncio.gather(*tasks)

        exile_votes: dict[str, str] = {}
        for result in results:
            self._record_result(state, result)
            exile_votes[result.caller] = _require_str_target(
                result.commit.value, terminal=result.commit.tool, caller=result.caller
            )
        return DayActions(exile_votes=exile_votes)

    def bids(self, state: GameState, /) -> dict[str, int]:
        """Fan-out one `submit_bid` loop per alive player; stage per-bidder drafts in roster order."""
        if state.phase is not Phase.DAY:
            raise ValueError(f"bids requires the day phase, got {state.phase.value}")
        return asyncio.run(self._gather_bids(state))

    async def _gather_bids(self, state: GameState) -> dict[str, int]:
        acting: list[tuple[str, str]] = [(name, role) for name, role in self._roster if state.is_alive(name)]
        tasks = [
            self._invoke_react_async(
                state=state,
                caller=name,
                role=role,
                decision_brief=_format_brief(
                    _BID_BRIEF, caller=name, role=role, state=state, memory=self._memories[name]
                ),
                terminal_name="submit_bid",
                terminal_fn=submit_bid,
            )
            for name, role in acting
        ]
        results = await asyncio.gather(*tasks)

        bids_map: dict[str, int] = {}
        for result in results:
            self._record_result(state, result)
            amount = _require_int_amount(result.commit.value, terminal=result.commit.tool, caller=result.caller)
            bids_map[result.caller] = amount
            self._pending_drafts.append(
                EventDraft(
                    type=BID,
                    payload={"bidder": result.caller, "amount": amount},
                    recipients=(result.caller,),
                )
            )
        return bids_map

    def next_speech(self, state: GameState, speaker: str, /) -> str:
        """Run one `speak` loop for `speaker`; stage a public `SPEECH` draft and return the message.

        Called once per resolved speaker by the driver, which drains and
        observes the staged `SPEECH` between speakers — so a later speaker's
        brief (built from its memory by `_format_brief`) carries the earlier
        speeches and the agent can react to them.
        """
        if state.phase is not Phase.DAY:
            raise ValueError(f"next_speech requires the day phase, got {state.phase.value}")
        if not state.is_alive(speaker):
            raise ValueError(f"next_speech received a dead speaker: {speaker!r}")
        return asyncio.run(self._run_next_speech(state, speaker))

    async def _run_next_speech(self, state: GameState, speaker: str) -> str:
        role = self._role_by_name[speaker]
        result = await self._invoke_react_async(
            state=state,
            caller=speaker,
            role=role,
            decision_brief=_format_brief(
                _SPEECH_BRIEF, caller=speaker, role=role, state=state, memory=self._memories[speaker]
            ),
            terminal_name="speak",
            terminal_fn=speak,
        )
        self._record_result(state, result)
        message = _require_str_message(result.commit.value, terminal=result.commit.tool, caller=result.caller)
        self._pending_drafts.append(
            EventDraft(
                type=SPEECH,
                payload={"speaker": result.caller, "message": message},
                recipients=(),
            )
        )
        return message

    def next_reaction(self, state: GameState, reactor: str, /) -> None:
        """Run one short reaction loop for `reactor`; stage its public draft.

        Called once per living player by the driver in the seeded reaction
        order, which drains and `observe`s the staged event between reactors —
        so a later reactor's brief (built from its memory) carries the earlier
        accusations and defenses and can respond to them. The reactor commits
        `accuse` / `defend` (a public draft) or `pass_turn` (no draft).
        """
        if state.phase is not Phase.DAY:
            raise ValueError(f"next_reaction requires the day phase, got {state.phase.value}")
        if not state.is_alive(reactor):
            raise ValueError(f"next_reaction received a dead reactor: {reactor!r}")
        asyncio.run(self._run_next_reaction(state, reactor))

    async def _run_next_reaction(self, state: GameState, reactor: str) -> None:
        role = self._role_by_name[reactor]
        try:
            result = await self._invoke_react_async(
                state=state,
                caller=reactor,
                role=role,
                decision_brief=_format_brief(
                    _REACTION_BRIEF, caller=reactor, role=role, state=state, memory=self._memories[reactor]
                ),
                terminal_fns=_REACTION_TERMINALS,
            )
        except Exception as err:
            # The reaction round is optional signal: a seat that fails to commit a
            # valid reaction (a truncated / unparseable LM response, or no commit
            # within max_iters) simply stays silent. Unlike a night kill or exile
            # vote, a missing reaction is a legal "pass", so degrade to one and log
            # it — a single bad reaction must never abort the whole benchmark game.
            logger.warning("reaction loop for %s failed (%s); treating as a pass", reactor, type(err).__name__)
            return
        self._record_result(state, result)
        self._stage_reaction_draft(result.caller, result.commit)

    def _stage_reaction_draft(self, reactor: str, commit: Commit) -> None:
        """Stage the public `ACCUSATION` / `DEFENSE` draft for a committed reaction.

        `pass_turn` stages nothing (silence is silent). `accuse` / `defend` carry
        a typed `Reaction` value; a non-`Reaction` value on those terminals is a
        loop bug, surfaced loud rather than logged as a malformed event.
        """
        if commit.tool == PASS_TURN:
            return
        reaction = commit.value
        if not isinstance(reaction, Reaction):
            raise TypeError(
                f"agent {reactor!r} committed {commit.tool!r} with non-Reaction value {commit.value!r}",
            )
        if commit.tool == ACCUSE:
            draft = EventDraft(
                type=ACCUSATION,
                payload={"accuser": reactor, "target": reaction.target, "reason": reaction.reason},
                recipients=(),
            )
        else:  # DEFEND
            draft = EventDraft(
                type=DEFENSE,
                payload={"defender": reactor, "defended": reaction.target, "reason": reaction.reason},
                recipients=(),
            )
        self._pending_drafts.append(draft)

    async def _invoke_react_async(
        self,
        *,
        state: GameState,
        caller: str,
        role: str,
        decision_brief: str,
        terminal_name: str = "",
        terminal_fn: Callable[..., ToolResult] | None = None,
        terminal_fns: Mapping[str, Callable[..., ToolResult]] | None = None,
        living_pack: tuple[str, ...] = (),
        chat_terminal: bool = False,
    ) -> _DecisionResult:
        """Drive one seat's ReAct loop and return a structured result.

        The async fan-out path. Stages all per-seat side-effects
        (`werewolf_chat`, `tool_rejected`) into a local `drafts` list so
        concurrent seats never race on `self._pending_drafts`; the
        aggregator concatenates those lists in argument order after
        `gather` resolves. Does not mutate `self._trajectories` or
        `self._decision_seq` either — `_record_result` does that in
        roster order.

        Terminal selection, in precedence order:
        - `chat_terminal=True` — the night chat sub-phase: `werewolf_chat`
          is the terminal (the message is the commit), no intermediates.
        - `terminal_fns` — a multi-terminal loop (the day reaction round:
          `accuse` / `defend` / `pass_turn`); each is bound via
          `_bind_terminal`, no intermediates.
        - otherwise — the single `terminal_fn`; a living werewolf at night
          additionally gets `werewolf_chat` as an intermediate so it can
          speak before committing its kill vote.
        """
        memory = self._memories[caller]
        cognitive = [_bind_cognitive(fn, state, memory, caller) for fn in _COGNITIVE_TOOLS]
        drafts: list[EventDraft] = []
        local_on_reject = self._build_local_on_reject(caller, drafts)

        terminals: Mapping[str, Callable[..., ToolResult]]
        intermediates: Mapping[str, Callable[..., ToolResult]] = _EMPTY_INTERMEDIATES
        if chat_terminal:
            terminals = {terminal_name: self._bind_werewolf_chat_local(state, caller, living_pack, drafts)}
        elif terminal_fns is not None:
            terminals = {name: _bind_terminal(fn, state, caller, memory) for name, fn in terminal_fns.items()}
        else:
            if terminal_fn is None:
                raise ValueError(f"_invoke_react_async for {caller!r} needs terminal_fn or terminal_fns")
            terminals = {terminal_name: _bind_terminal(terminal_fn, state, caller, memory)}
            if state.phase is Phase.NIGHT and role == Role.WEREWOLF.value:
                intermediates = {"werewolf_chat": self._bind_werewolf_chat_local(state, caller, living_pack, drafts)}

        # `react_decide_async` fires the sink once after a successful commit;
        # an uncommitted loop raises and `captured` stays empty.
        captured: list[tuple[tuple[ReActStep, ...], tuple[LMCallRecord, ...]]] = []

        def sink(steps: tuple[ReActStep, ...], lm_calls: tuple[LMCallRecord, ...]) -> None:
            captured.append((steps, lm_calls))

        if self._on_decision_start is not None:
            self._on_decision_start(caller, role)

        on_step_local = self._on_step

        def step_sink(step: ReActStep, iter_lm_calls: tuple[LMCallRecord, ...]) -> None:
            if on_step_local is not None:
                on_step_local(caller, role, step, iter_lm_calls)

        on_thought_local = self._on_thought_chunk
        thought_callback: Callable[[int, str], None] | None
        if on_thought_local is None:
            # No thought hook on this source → skip `dspy.streamify` entirely.
            # Streaming wraps the per-iter call in an async generator that
            # introduces real awaits; with shared-LM tests under `gather`
            # those awaits interleave seats and corrupt the answer queue.
            # Plain `acall` keeps the loop tight and parallel-safe.
            thought_callback = None
        else:
            captured_thought_fn = on_thought_local

            def thought_callback(iter_idx: int, text: str) -> None:
                captured_thought_fn(caller, role, iter_idx, text)

        commit = await react_decide_async(
            caller=caller,
            cognitive_tools=cognitive,
            intermediate_tools=intermediates,
            terminal_tools=terminals,
            decision_brief=decision_brief,
            lm=self._lms[caller],
            max_iters=self._max_iters,
            on_reject=local_on_reject,
            trace_sink=sink,
            step_sink=step_sink,
            on_thought_chunk=thought_callback,
        )

        steps, lm_calls = captured[0]
        return _DecisionResult(
            caller=caller,
            role=role,
            commit=commit,
            steps=steps,
            lm_calls=lm_calls,
            drafts=tuple(drafts),
        )

    def _record_result(self, state: GameState, result: _DecisionResult) -> None:
        """Apply one seat's `_DecisionResult` to the source's accumulators.

        Called by phase aggregators in argument (= roster) order. Assigns
        `decision_seq`, appends the trajectory, fires `on_trajectory`, and
        extends `self._pending_drafts` — all in a deterministic order
        regardless of the wall-clock order in which the gathered loops
        actually completed.
        """
        trajectory = Trajectory(
            decision_seq=self._decision_seq,
            round=state.round,
            phase=state.phase.value,
            caller=result.caller,
            role=result.role,
            terminal_tool=result.commit.tool,
            committed_value=result.commit.value,
            react_trajectory=result.steps,
            lm_calls=result.lm_calls,
        )
        self._trajectories.append(trajectory)
        self._decision_seq += 1
        if self._on_trajectory is not None:
            self._on_trajectory(trajectory)
        self._pending_drafts.extend(result.drafts)
