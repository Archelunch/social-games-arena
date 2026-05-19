"""Werewolf game-action tools (WEREWOLF_DESIGN.md §6).

The seven game-action tools are how a ReAct agent *commits* a move. Each tool
function turns a raw, parametrized invocation into a `ToolResult` carrying
either the parsed, validated argument or an informative rejection reason. This
upholds invariant #3: an illegal move (dead / unknown / self target, wrong
role, wrong phase, malformed argument) is rejected, never silently applied.

Scope is intentionally narrow: a tool call is a *pure verdict*. It reads
`GameState`, never mutates it, and emits no events. Turning an accepted call
into an `EventDraft` or accumulating it into `NightActions` / `DayActions` is
the job of the gating layer (T17) and the agent loop (T21) — out of scope here.

Role / phase / target-presence gates are delegated to the engine's
`validate_tool_call`; each tool declares its gates in
`WEREWOLF_TOOL_REQUIREMENTS`. Tool-specific argument rules — self-targeting is
forbidden, a chat / speech message must be non-empty, a bid must be
non-negative — are checked by the per-tool function once the generic gate
passes.

Two gates the engine validator cannot express, handled in the functions:

- `submit_exile_vote`'s target may be the `ABSTAIN` literal, which is not a
  player; its registry requirement therefore declares `requires_target=False`,
  and the function runs the player-target gates only on a non-abstain vote.
- `speak` is available only to the round's bid winners (§5) — a *dynamic* gate
  depending on the bid tally, not on static role / phase. Its requirement
  gates DAY phase only; the bid-winner restriction is deferred to T18.
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

from social_deduction_bench.engine import GameState, Phase, ToolCall, ToolRequirement, validate_tool_call
from social_deduction_bench.games.werewolf.events import ABSTAIN
from social_deduction_bench.games.werewolf.roles import Role

# Tool names. Distinct from the event-type constants in `events.py`: a tool name
# and an event type are different domains even where a string value coincides.
WEREWOLF_CHAT = "werewolf_chat"
SUBMIT_KILL_VOTE = "submit_kill_vote"
SEER_INSPECT = "seer_inspect"
DOCTOR_PROTECT = "doctor_protect"
SUBMIT_BID = "submit_bid"
SPEAK = "speak"
SUBMIT_EXILE_VOTE = "submit_exile_vote"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The verdict on a parametrized game-action tool call.

    `reason` is empty exactly when `valid` is `True`; on rejection it is an
    informative string the agent reads to self-correct inside its ReAct loop.
    `value` is the parsed, validated argument on success — a player name, a
    message, an int bid, or the `ABSTAIN` literal — and `None` on rejection.
    Frozen so a recorded verdict cannot be mutated after the fact (invariant #5).
    """

    valid: bool
    reason: str = ""
    value: object | None = None


# Per-tool role / phase / target-presence gates, consumed by `validate_tool_call`
# and (in T17) by the tool-gating layer. Read-only so a tool's contract cannot
# drift mid-game. `submit_exile_vote` declares `requires_target=False` — see the
# module docstring. `speak` declares no role — the bid-winner gate is T18's.
WEREWOLF_TOOL_REQUIREMENTS: MappingProxyType[str, ToolRequirement] = MappingProxyType(
    {
        WEREWOLF_CHAT: ToolRequirement(phase=Phase.NIGHT, role=Role.WEREWOLF.value),
        SUBMIT_KILL_VOTE: ToolRequirement(phase=Phase.NIGHT, role=Role.WEREWOLF.value, requires_target=True),
        SEER_INSPECT: ToolRequirement(phase=Phase.NIGHT, role=Role.SEER.value, requires_target=True),
        DOCTOR_PROTECT: ToolRequirement(phase=Phase.NIGHT, role=Role.DOCTOR.value, requires_target=True),
        SUBMIT_BID: ToolRequirement(phase=Phase.DAY),
        SPEAK: ToolRequirement(phase=Phase.DAY),
        SUBMIT_EXILE_VOTE: ToolRequirement(phase=Phase.DAY),
    }
)


def _gate(state: GameState, caller: str, tool: str, target: str | None) -> str | None:
    """Run the engine role / phase / target gate; return a rejection reason or `None`.

    Delegates to `validate_tool_call` against the tool's `WEREWOLF_TOOL_REQUIREMENTS`
    entry. `None` means the generic gate passed; a non-`None` string is the
    informative rejection reason, ready to surface in a `ToolResult`.
    """
    call = ToolCall(caller=caller, tool=tool, target=target)
    result = validate_tool_call(state, call, WEREWOLF_TOOL_REQUIREMENTS[tool])
    return None if result.valid else result.reason


def _target_tool(state: GameState, caller: str, tool: str, target: str) -> ToolResult:
    """Validate a night target tool: the engine gate, then forbid self-targeting.

    Shared by `submit_kill_vote`, `seer_inspect`, and `doctor_protect`. The
    engine gate (caller / role / phase / known-alive target) runs first; a
    living, real, non-self target is then required — self-targeting on a
    night-ability tool is forbidden (resolved ambiguity, WEREWOLF_DESIGN.md §6).
    """
    reason = _gate(state, caller, tool, target)
    if reason is not None:
        return ToolResult(valid=False, reason=reason)
    if target == caller:
        return ToolResult(valid=False, reason=f"tool {tool!r} cannot target the caller {caller!r}")
    return ToolResult(valid=True, value=target)


def _message_tool(state: GameState, caller: str, tool: str, message: str) -> ToolResult:
    """Validate a message tool: the engine gate, then require a non-empty message.

    Shared by `werewolf_chat` and `speak`. A whitespace-only message carries no
    content and is a malformed action, rejected like an empty one.
    """
    reason = _gate(state, caller, tool, None)
    if reason is not None:
        return ToolResult(valid=False, reason=reason)
    if not message.strip():
        return ToolResult(valid=False, reason=f"tool {tool!r} requires a non-empty message")
    return ToolResult(valid=True, value=message)


def werewolf_chat(state: GameState, caller: str, message: str) -> ToolResult:
    """A werewolf's private pack-chat message (night)."""
    return _message_tool(state, caller, WEREWOLF_CHAT, message)


def submit_kill_vote(state: GameState, caller: str, target: str) -> ToolResult:
    """A werewolf's joint-kill vote for `target` (night)."""
    return _target_tool(state, caller, SUBMIT_KILL_VOTE, target)


def seer_inspect(state: GameState, caller: str, target: str) -> ToolResult:
    """The seer's faction inspection of `target` (night)."""
    return _target_tool(state, caller, SEER_INSPECT, target)


def doctor_protect(state: GameState, caller: str, target: str) -> ToolResult:
    """The doctor's protection of `target` from the night kill (night)."""
    return _target_tool(state, caller, DOCTOR_PROTECT, target)


def submit_bid(state: GameState, caller: str, amount: int) -> ToolResult:
    """A player's bid for a discussion speaking slot (day).

    Rejects a negative bid — the only bid gate T15 owns. The upper bound and the
    top-K speaker selection belong to T18, which owns discussion bidding.
    """
    reason = _gate(state, caller, SUBMIT_BID, None)
    if reason is not None:
        return ToolResult(valid=False, reason=reason)
    if amount < 0:
        return ToolResult(valid=False, reason=f"tool {SUBMIT_BID!r} requires a non-negative amount, got {amount}")
    return ToolResult(valid=True, value=amount)


def speak(state: GameState, caller: str, message: str) -> ToolResult:
    """A player's public discussion statement (day).

    Gated to the day phase only. The bid-winner restriction (§5) is dynamic and
    is enforced by T18, not by this static gate.
    """
    return _message_tool(state, caller, SPEAK, message)


def submit_exile_vote(state: GameState, caller: str, target: str) -> ToolResult:
    """A player's exile vote for `target`, or the `ABSTAIN` literal (day).

    `ABSTAIN` is a legal value, not a player, so it cannot go through the
    engine's known-alive target gate; an abstain vote is validated for
    caller / phase only. A non-abstain vote runs the full player-target gate.
    """
    if target == ABSTAIN:
        reason = _gate(state, caller, SUBMIT_EXILE_VOTE, None)
        if reason is not None:
            return ToolResult(valid=False, reason=reason)
        return ToolResult(valid=True, value=ABSTAIN)

    # Derive the player-target gate from the registry entry so the phase gate
    # has a single source of truth — only `requires_target` differs.
    requirement = replace(WEREWOLF_TOOL_REQUIREMENTS[SUBMIT_EXILE_VOTE], requires_target=True)
    result = validate_tool_call(state, ToolCall(caller=caller, tool=SUBMIT_EXILE_VOTE, target=target), requirement)
    if not result.valid:
        return ToolResult(valid=False, reason=result.reason)
    return ToolResult(valid=True, value=target)
