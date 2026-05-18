"""Game-agnostic tool-call validation.

Upholds benchmark invariant #3: agents change state only via validated tool
calls, and illegal moves (dead target, wrong phase, wrong role) are rejected
with an informative error observation — never silently applied. `validate_tool_call`
forms a verdict by a pure read over `GameState`; it mutates nothing, so a
rejected call leaves the position byte-identical.

This module stays game-agnostic: it knows nothing of Werewolf roles or per-tool
rules. A game declares each tool's gates as a `ToolRequirement`; T07 supplies
only the generic primitive that checks a `ToolCall` against one.

Checks run in a fixed, first-failure order so a call that fails several gates
at once always reports the *same* first reason. That makes the recorded error
observation deterministic and byte-identical on every replay (invariant #4).
Building and appending the error `Event` is left to the game loop; T07 supplies
only the verdict and its reason string.
"""

from dataclasses import dataclass

from social_deduction_bench.engine.state import GameState, Phase


@dataclass(frozen=True, slots=True)
class ToolCall:
    """An agent's request to invoke a tool.

    `caller` and `target` are player names — the engine's stable identity keys
    (invariant #1). `target` is `None` for a tool that needs no player target
    (e.g. werewolf chat). Frozen so a request cannot be edited between
    validation and application.
    """

    caller: str
    tool: str
    target: str | None = None


@dataclass(frozen=True, slots=True)
class ToolRequirement:
    """The gates a game declares for one tool.

    A `None` field means "no constraint" — the validator must not invent a gate
    that was never declared. `phase` and `role` pin when and by whom the tool
    may be called; `requires_target` pins whether a player target is mandatory.
    Frozen because a tool's contract is fixed for the game.
    """

    phase: Phase | None = None
    role: str | None = None
    requires_target: bool = False


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """The verdict on a tool call.

    `reason` is empty exactly when `valid` is `True`; on rejection it is an
    informative string the agent reads to self-correct inside its ReAct loop.
    Frozen so a recorded error observation cannot be mutated after the fact,
    which would corrupt the append-only history (invariant #5).
    """

    valid: bool
    reason: str = ""


def validate_tool_call(state: GameState, call: ToolCall, requirement: ToolRequirement) -> ValidationResult:
    """Return the verdict on `call` against `requirement`, reading `state` only.

    This is a pure derivation: it mutates nothing, so a rejected call leaves the
    position byte-identical (invariant #3). Checks run in a fixed order and stop
    at the first failure, so a call failing several gates always reports the
    same reason and the recorded error observation is replayable (invariant #4):

    1. caller is a known player,
    2. caller is alive,
    3. caller's role matches `requirement.role` (when set),
    4. `state.phase` matches `requirement.phase` (when set),
    5. a target is present when `requirement.requires_target`,
    6. the target (when set) is a known player,
    7. the target (when set) is alive.

    Caller checks run before target checks so the verdict is deterministic when
    a call fails on both sides at once.
    """
    try:
        caller = state.player(call.caller)
    except KeyError:
        return ValidationResult(valid=False, reason=f"unknown caller '{call.caller}': not a player in this game")

    if not caller.alive:
        return ValidationResult(valid=False, reason=f"caller '{call.caller}' is dead and cannot act")

    if requirement.role is not None and caller.role != requirement.role:
        return ValidationResult(
            valid=False,
            reason=(
                f"caller '{call.caller}' has role '{caller.role}', "
                f"but tool '{call.tool}' requires role '{requirement.role}'"
            ),
        )

    if requirement.phase is not None and state.phase != requirement.phase:
        return ValidationResult(
            valid=False,
            reason=(
                f"tool '{call.tool}' requires phase '{requirement.phase.value}', "
                f"but the current phase is '{state.phase.value}'"
            ),
        )

    if requirement.requires_target and call.target is None:
        return ValidationResult(valid=False, reason=f"tool '{call.tool}' requires a target, but none was given")

    if call.target is not None:
        try:
            target = state.player(call.target)
        except KeyError:
            return ValidationResult(
                valid=False,
                reason=f"unknown target '{call.target}': not a player in this game",
            )
        if not target.alive:
            return ValidationResult(valid=False, reason=f"target '{call.target}' is dead and cannot be acted on")

    return ValidationResult(valid=True)
