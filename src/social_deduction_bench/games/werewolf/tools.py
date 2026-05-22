"""Werewolf game-action tools — the moves a player-agent can commit.

Each public function is one tool: it validates a player's tool call against the
game state and returns a `ToolResult` with the parsed argument, or a rejection
reason the agent can act on. Tool calls are pure — they read game state and
never mutate it.

Role, phase, and target gates are declared per tool in
`WEREWOLF_TOOL_REQUIREMENTS` and checked by the engine's `validate_tool_call`;
each function adds its own argument rules (no self-target, non-empty message,
non-negative bid).
"""

from dataclasses import dataclass, replace
from types import MappingProxyType

from social_deduction_bench.engine import GameState, Phase, ToolCall, ToolRequirement, validate_tool_call
from social_deduction_bench.games.werewolf.config import MAX_BID
from social_deduction_bench.games.werewolf.events import ABSTAIN
from social_deduction_bench.games.werewolf.roles import Role

# Tool names.
WEREWOLF_CHAT = "werewolf_chat"
SUBMIT_KILL_VOTE = "submit_kill_vote"
SEER_INSPECT = "seer_inspect"
DOCTOR_PROTECT = "doctor_protect"
SUBMIT_BID = "submit_bid"
SPEAK = "speak"
SUBMIT_EXILE_VOTE = "submit_exile_vote"
ACCUSE = "accuse"
DEFEND = "defend"
PASS_TURN = "pass_turn"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of a game-action tool call.

    `reason` is empty when `valid` is `True`; otherwise it explains the
    rejection. `value` holds the parsed argument on success and is `None` on
    rejection.
    """

    valid: bool
    reason: str = ""
    value: object | None = None


@dataclass(frozen=True, slots=True)
class Reaction:
    """A structured day-reaction commit: who it targets and the stated reason.

    Carried as the `ToolResult.value` of a valid `accuse` / `defend` call so the
    decision adapter can build the public `ACCUSATION` / `DEFENSE` event draft
    from a typed value rather than re-parsing the LLM kwargs (no raw dict at the
    tool→adapter boundary). `pass_turn` carries no `Reaction` (value is `None`).
    """

    target: str
    reason: str


# Role, phase, and target gates per tool, keyed by tool name. Read-only.
WEREWOLF_TOOL_REQUIREMENTS: MappingProxyType[str, ToolRequirement] = MappingProxyType(
    {
        WEREWOLF_CHAT: ToolRequirement(phase=Phase.NIGHT, role=Role.WEREWOLF.value),
        SUBMIT_KILL_VOTE: ToolRequirement(phase=Phase.NIGHT, role=Role.WEREWOLF.value, requires_target=True),
        SEER_INSPECT: ToolRequirement(phase=Phase.NIGHT, role=Role.SEER.value, requires_target=True),
        DOCTOR_PROTECT: ToolRequirement(phase=Phase.NIGHT, role=Role.DOCTOR.value, requires_target=True),
        SUBMIT_BID: ToolRequirement(phase=Phase.DAY),
        SPEAK: ToolRequirement(phase=Phase.DAY),
        SUBMIT_EXILE_VOTE: ToolRequirement(phase=Phase.DAY),
        ACCUSE: ToolRequirement(phase=Phase.DAY, requires_target=True),
        DEFEND: ToolRequirement(phase=Phase.DAY, requires_target=True),
        PASS_TURN: ToolRequirement(phase=Phase.DAY),
    }
)


def _gate(state: GameState, caller: str, tool: str, target: str | None) -> str | None:
    """Run the engine role/phase/target gate for `tool`; return a rejection reason, or `None` if it passes."""
    call = ToolCall(caller=caller, tool=tool, target=target)
    result = validate_tool_call(state, call, WEREWOLF_TOOL_REQUIREMENTS[tool])
    return None if result.valid else result.reason


def _target_tool(state: GameState, caller: str, tool: str, target: str) -> ToolResult:
    """Validate a night target tool: run the engine gate, then reject a self-target."""
    reason = _gate(state, caller, tool, target)
    if reason is not None:
        return ToolResult(valid=False, reason=reason)
    if target == caller:
        return ToolResult(valid=False, reason=f"tool {tool!r} cannot target the caller {caller!r}")
    return ToolResult(valid=True, value=target)


def _message_tool(state: GameState, caller: str, tool: str, message: str) -> ToolResult:
    """Validate a message tool: run the engine gate, then reject an empty or whitespace-only message."""
    reason = _gate(state, caller, tool, None)
    if reason is not None:
        return ToolResult(valid=False, reason=reason)
    if not message.strip():
        return ToolResult(valid=False, reason=f"tool {tool!r} requires a non-empty message")
    return ToolResult(valid=True, value=message)


def werewolf_chat(state: GameState, caller: str, message: str) -> ToolResult:
    """Send a message to your fellow werewolves in the private night chat.

    Only werewolves see it. `message` is the text to send.
    """
    return _message_tool(state, caller, WEREWOLF_CHAT, message)


def submit_kill_vote(state: GameState, caller: str, target: str) -> ToolResult:
    """Vote for the player the werewolves will kill tonight.

    `target` is the name of a living player other than yourself or a fellow werewolf.
    Your pack is listed in your decision brief, and you cannot target a packmate.
    """
    result = _target_tool(state, caller, SUBMIT_KILL_VOTE, target)
    if not result.valid:
        return result
    # By the time we get here `_target_tool` has confirmed `target` exists and
    # is alive. The role check forbids friendly fire on the pack — continues
    # the T15 "self-target forbidden" precedent for the packmate case.
    if state.player(target).role == Role.WEREWOLF.value:
        return ToolResult(
            valid=False,
            reason=(f"tool {SUBMIT_KILL_VOTE!r} cannot target a fellow werewolf ({target!r})"),
        )
    return result


def seer_inspect(state: GameState, caller: str, target: str) -> ToolResult:
    """Inspect one player to learn their faction (werewolf or villager).

    `target` is the name of a living player other than yourself.
    """
    return _target_tool(state, caller, SEER_INSPECT, target)


def doctor_protect(state: GameState, caller: str, target: str) -> ToolResult:
    """Protect one player from the werewolves' kill tonight; if attacked, they survive.

    `target` is the name of a living player other than yourself.
    """
    return _target_tool(state, caller, DOCTOR_PROTECT, target)


def submit_bid(state: GameState, caller: str, amount: int) -> ToolResult:
    """Bid for a speaking slot in today's discussion; the top bidders get to speak.

    `amount` is an integer from 0 to {max_bid}, and also no more than your
    remaining speaking budget (shown in your brief) — winners pay their bid, so
    the budget depletes across the game. Bid higher when you most want to speak.
    """
    reason = _gate(state, caller, SUBMIT_BID, None)
    if reason is not None:
        return ToolResult(valid=False, reason=reason)
    if amount < 0:
        return ToolResult(valid=False, reason=f"tool {SUBMIT_BID!r} requires a non-negative amount, got {amount}")
    if amount > MAX_BID:
        return ToolResult(
            valid=False,
            reason=f"tool {SUBMIT_BID!r} requires an amount at most {MAX_BID}, got {amount}",
        )
    remaining = state.player(caller).bid_budget
    if amount > remaining:
        return ToolResult(
            valid=False,
            reason=f"tool {SUBMIT_BID!r} amount {amount} exceeds your remaining speaking budget {remaining}",
        )
    return ToolResult(valid=True, value=amount)


# Bind the live `MAX_BID` into `submit_bid`'s docstring so the LLM-facing tool
# schema cannot drift from the engine's actual cap. Guard the `None` case rather
# than `assert` so import under `-O` (asserts stripped, docstrings set to `None`)
# does not raise `AttributeError`.
if submit_bid.__doc__ is not None:
    submit_bid.__doc__ = submit_bid.__doc__.format(max_bid=MAX_BID)


def speak(state: GameState, caller: str, message: str) -> ToolResult:
    """Make a public statement to every player in the day discussion.

    `message` is the text all players will see.
    """
    return _message_tool(state, caller, SPEAK, message)


def submit_exile_vote(state: GameState, caller: str, target: str) -> ToolResult:
    """Vote to exile one player, or abstain.

    `target` is the name of a living player, or "abstain" to cast no vote.
    """
    if target == ABSTAIN:
        reason = _gate(state, caller, SUBMIT_EXILE_VOTE, None)
        if reason is not None:
            return ToolResult(valid=False, reason=reason)
        return ToolResult(valid=True, value=ABSTAIN)

    # A non-abstain vote also needs a real, living player target.
    requirement = replace(WEREWOLF_TOOL_REQUIREMENTS[SUBMIT_EXILE_VOTE], requires_target=True)
    result = validate_tool_call(state, ToolCall(caller=caller, tool=SUBMIT_EXILE_VOTE, target=target), requirement)
    if not result.valid:
        return ToolResult(valid=False, reason=result.reason)
    return ToolResult(valid=True, value=target)


def accuse(state: GameState, caller: str, target: str, reason: str) -> ToolResult:
    """Publicly accuse a living player of being a werewolf, stating a brief reason.

    `target` is the name of a living player other than yourself; `reason` is a
    short, non-empty justification that every player will see.
    """
    gate = _gate(state, caller, ACCUSE, target)
    if gate is not None:
        return ToolResult(valid=False, reason=gate)
    if target == caller:
        return ToolResult(valid=False, reason=f"tool {ACCUSE!r} cannot target the caller {caller!r}")
    if not reason.strip():
        return ToolResult(valid=False, reason=f"tool {ACCUSE!r} requires a non-empty reason")
    return ToolResult(valid=True, value=Reaction(target=target, reason=reason))


def defend(state: GameState, caller: str, target: str, reason: str) -> ToolResult:
    """Publicly defend a living player against suspicion, stating a brief reason.

    `target` is any living player, including yourself; `reason` is a short,
    non-empty justification that every player will see.
    """
    gate = _gate(state, caller, DEFEND, target)
    if gate is not None:
        return ToolResult(valid=False, reason=gate)
    if not reason.strip():
        return ToolResult(valid=False, reason=f"tool {DEFEND!r} requires a non-empty reason")
    return ToolResult(valid=True, value=Reaction(target=target, reason=reason))


def pass_turn(state: GameState, caller: str) -> ToolResult:
    """Stay silent this reaction — make no accusation or defense.

    Use this when you have nothing to add to the discussion right now.
    """
    gate = _gate(state, caller, PASS_TURN, None)
    if gate is not None:
        return ToolResult(valid=False, reason=gate)
    return ToolResult(valid=True, value=None)
