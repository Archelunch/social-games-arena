"""Tests for game-agnostic tool-call validation (T07).

These encode benchmark invariant #3: agents change state only via validated
tool calls, and illegal moves (dead target, wrong phase, wrong role) are
rejected with an informative error observation — never silently applied.

The validator is a pure derivation over `GameState`: it forms a verdict and
touches nothing. Checks run in a fixed order so a call that fails several at
once always reports the *same* first reason — the error observation is itself
replayable (invariant #4). Building/appending the error `Event` is left to the
game loop (T17); T07 supplies only the verdict and its reason string.
"""

import dataclasses

import pytest

from social_deduction_bench.engine import (
    GameState,
    Phase,
    ToolCall,
    ToolRequirement,
    validate_tool_call,
)

# A canonical 4-player setup reused across tests: (name, role) pairs. Round 1,
# NIGHT — see GameState.initial.
PLAYERS = [("Alice", "villager"), ("Bob", "werewolf"), ("Cara", "seer"), ("Dan", "doctor")]


def _state() -> GameState:
    """A fresh round-1 NIGHT state with the canonical roster."""
    return GameState.initial(PLAYERS)


def test_valid_call_passes() -> None:
    """A call meeting every requirement is accepted with no reason.

    The happy path must succeed, or the validator would block legal play and
    the benchmark could never progress.
    """
    call = ToolCall(caller="Cara", tool="seer_inspect", target="Bob")
    requirement = ToolRequirement(phase=Phase.NIGHT, role="seer", requires_target=True)

    result = validate_tool_call(_state(), call, requirement)

    assert result.valid is True
    assert result.reason == ""


def test_unknown_caller_is_rejected() -> None:
    """A call from a name not in the roster is rejected and names the caller.

    Invariant #1: only engine-known players exist. An unknown caller is a
    routing/identity bug that must fail loud, not be quietly accepted.
    """
    call = ToolCall(caller="Mallory", tool="seer_inspect", target="Bob")

    result = validate_tool_call(_state(), call, ToolRequirement())

    assert result.valid is False
    assert "Mallory" in result.reason


def test_dead_caller_is_rejected() -> None:
    """A dead player cannot act.

    Invariant #3: night/day resolution kills players; a corpse issuing a tool
    call is an illegal move and must be rejected with an error observation.
    """
    state = _state().with_player_killed("Cara")
    call = ToolCall(caller="Cara", tool="seer_inspect", target="Bob")

    result = validate_tool_call(state, call, ToolRequirement())

    assert result.valid is False
    assert "Cara" in result.reason


def test_wrong_role_is_rejected() -> None:
    """A caller whose role does not match the tool's required role is rejected.

    Invariant #3, "wrong role": a villager calling `seer_inspect` is an illegal
    move. The reason names both the required and the actual role so the agent
    can correct itself inside its ReAct loop.
    """
    call = ToolCall(caller="Alice", tool="seer_inspect", target="Bob")
    requirement = ToolRequirement(role="seer")

    result = validate_tool_call(_state(), call, requirement)

    assert result.valid is False
    assert "seer" in result.reason
    assert "villager" in result.reason


def test_matching_role_passes() -> None:
    """The role gate accepts the correct role.

    The negative test alone could pass with a validator that rejects every
    call; this pins that a matching role is genuinely allowed through.
    """
    call = ToolCall(caller="Cara", tool="seer_inspect", target="Bob")
    requirement = ToolRequirement(role="seer")

    assert validate_tool_call(_state(), call, requirement).valid is True


def test_wrong_phase_is_rejected() -> None:
    """A night action attempted during the day is rejected.

    Invariant #3, "wrong phase": phase gates the decision points (DESIGN §5).
    A seer inspecting at day is an illegal move.
    """
    day = _state().with_phase(Phase.DAY)
    call = ToolCall(caller="Cara", tool="seer_inspect", target="Bob")
    requirement = ToolRequirement(phase=Phase.NIGHT)

    result = validate_tool_call(day, call, requirement)

    assert result.valid is False
    assert "night" in result.reason
    assert "day" in result.reason


def test_matching_phase_passes() -> None:
    """The phase gate accepts the correct phase.

    Pins that a matching phase is allowed through, so the phase check is not
    vacuously rejecting everything.
    """
    call = ToolCall(caller="Cara", tool="seer_inspect", target="Bob")
    requirement = ToolRequirement(phase=Phase.NIGHT)

    assert validate_tool_call(_state(), call, requirement).valid is True


def test_missing_required_target_is_rejected() -> None:
    """A tool that requires a target rejects a call with `target=None`.

    A targeted action (seer inspect, doctor protect) with no target cannot be
    resolved; the engine must reject it rather than resolve against nobody.
    """
    call = ToolCall(caller="Cara", tool="seer_inspect", target=None)
    requirement = ToolRequirement(requires_target=True)

    result = validate_tool_call(_state(), call, requirement)

    assert result.valid is False
    assert "target" in result.reason


def test_targetless_tool_without_target_passes() -> None:
    """A tool that needs no target is fine with `target=None`.

    `werewolf_chat` / `submit_bid` carry no player target; demanding one would
    block legal calls.
    """
    call = ToolCall(caller="Bob", tool="werewolf_chat", target=None)
    requirement = ToolRequirement(requires_target=False)

    assert validate_tool_call(_state(), call, requirement).valid is True


def test_unknown_target_is_rejected() -> None:
    """A call targeting a name not in the roster is rejected.

    Invariant #1: a target must be an engine-known player; an unknown target
    cannot be resolved and must fail loud.
    """
    call = ToolCall(caller="Cara", tool="seer_inspect", target="Mallory")
    requirement = ToolRequirement(requires_target=True)

    result = validate_tool_call(_state(), call, requirement)

    assert result.valid is False
    assert "Mallory" in result.reason


def test_dead_target_is_rejected() -> None:
    """A call targeting a dead player is rejected.

    Invariant #3, "dead target": this is the headline illegal move T07 exists
    to catch. Inspecting/protecting/killing a corpse must never be applied.
    """
    state = _state().with_player_killed("Bob")
    call = ToolCall(caller="Cara", tool="seer_inspect", target="Bob")
    requirement = ToolRequirement(requires_target=True)

    result = validate_tool_call(state, call, requirement)

    assert result.valid is False
    assert "Bob" in result.reason


def test_empty_requirement_accepts_any_alive_known_caller() -> None:
    """With no phase/role/target constraints, any alive known caller passes.

    `None` requirement fields mean "no constraint" — the validator must not
    invent a gate that was never declared (e.g. `submit_bid` has no role gate).
    """
    call = ToolCall(caller="Alice", tool="submit_bid", target=None)

    assert validate_tool_call(_state(), call, ToolRequirement()).valid is True


def test_checks_run_in_fixed_order_caller_before_target() -> None:
    """A call failing several checks reports the caller error first.

    Invariant #4: the verdict must be deterministic. With a dead caller *and* a
    dead target, the reason is fixed (caller, not target) so the recorded error
    observation is byte-identical on every replay.
    """
    state = _state().with_player_killed("Cara").with_player_killed("Bob")
    call = ToolCall(caller="Cara", tool="seer_inspect", target="Bob")
    requirement = ToolRequirement(requires_target=True)

    result = validate_tool_call(state, call, requirement)

    assert result.valid is False
    assert "Cara" in result.reason
    assert "Bob" not in result.reason


def test_role_check_runs_before_phase_check() -> None:
    """A call failing both the role and phase gate reports the role reason.

    Invariant #4: the documented check order (role before phase) must be
    pinned, not just caller-before-target. A villager calling a seer-only,
    night-only tool during the day fails both gates; if the order silently
    flipped, the recorded error observation would change and replay would
    diverge.
    """
    day = _state().with_phase(Phase.DAY)
    call = ToolCall(caller="Alice", tool="seer_inspect", target="Bob")
    requirement = ToolRequirement(role="seer", phase=Phase.NIGHT)

    result = validate_tool_call(day, call, requirement)

    assert result.valid is False
    assert "seer" in result.reason
    assert "night" not in result.reason


def test_phase_check_runs_before_missing_target_check() -> None:
    """A call failing both the phase gate and the target-presence gate reports phase.

    Invariant #4: pins the phase-before-target-presence hop of the check order.
    A night tool called at day with no target fails both; the phase reason must
    win so the error observation is deterministic.
    """
    day = _state().with_phase(Phase.DAY)
    call = ToolCall(caller="Cara", tool="seer_inspect", target=None)
    requirement = ToolRequirement(phase=Phase.NIGHT, requires_target=True)

    result = validate_tool_call(day, call, requirement)

    assert result.valid is False
    assert "night" in result.reason


def test_validation_does_not_mutate_state() -> None:
    """Forming a verdict is a pure read; the state snapshot is untouched.

    Invariant #3: validation never changes state — only a validated, applied
    move does. A rejected call must leave the position byte-identical.
    """
    before = _state()
    call = ToolCall(caller="Mallory", tool="seer_inspect", target="Bob")

    validate_tool_call(before, call, ToolRequirement())

    assert before == _state()


def test_result_is_frozen() -> None:
    """`ValidationResult` is immutable — a verdict cannot be edited after the fact.

    A recorded error observation that could be mutated would corrupt the
    append-only history (invariant #5).
    """
    result = validate_tool_call(_state(), ToolCall(caller="Alice", tool="submit_bid"), ToolRequirement())

    assert result.valid is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.valid = False  # type: ignore[misc]
