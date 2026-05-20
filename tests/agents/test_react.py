"""Tests for the per-decision-point ReAct loop primitive.

`react_decide` drives `dspy.ReAct.react` (the per-iteration predictor)
directly; each iteration consumes one `DummyLM` answer dict
(`next_thought` / `next_tool_name` / `next_tool_args`). There is no extract
step. Tests build cognitive and terminal closures by hand so the loop runs
against the real cognitive-tool layer and the real game-action tools.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from social_deduction_bench.agents import GameMemory
from social_deduction_bench.agents.cognitive import (
    get_beliefs,
    get_plan,
    recall,
    remember,
    set_belief,
    set_plan,
)
from social_deduction_bench.agents.react import Commit, react_decide
from social_deduction_bench.engine import GameState
from social_deduction_bench.games.werewolf.cognitive import get_private_info, get_public_state
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.games.werewolf.tools import (
    ToolResult,
    submit_exile_vote,
    submit_kill_vote,
)


def _roster() -> tuple[tuple[str, str], ...]:
    return (
        ("Wolf1", Role.WEREWOLF.value),
        ("Wolf2", Role.WEREWOLF.value),
        ("Seer1", Role.SEER.value),
        ("Doc1", Role.DOCTOR.value),
        ("Vil1", Role.VILLAGER.value),
        ("Vil2", Role.VILLAGER.value),
    )


def _state() -> GameState:
    return GameState.initial(_roster())


def _cognitive_closures(state: GameState, memory: GameMemory, caller: str) -> list[Callable[..., str]]:
    def recall_(last_n_rounds: int | None = None) -> str:
        """Read your own memory."""
        return recall(state, memory, caller, last_n_rounds=last_n_rounds)

    def remember_(note: str) -> str:
        """Write a note."""
        return remember(state, memory, caller, note)

    def get_beliefs_() -> str:
        """Read beliefs."""
        return get_beliefs(state, memory, caller)

    def set_belief_(player: str, guess: str, confidence: str, evidence: str) -> str:
        """Update one belief row."""
        return set_belief(state, memory, caller, player, guess, confidence, evidence)  # type: ignore[arg-type]

    def get_plan_() -> str:
        """Read plan."""
        return get_plan(state, memory, caller)

    def set_plan_(text: str) -> str:
        """Overwrite plan."""
        return set_plan(state, memory, caller, text)

    def get_public_state_() -> str:
        """Read the public position."""
        return get_public_state(state, memory, caller)

    def get_private_info_() -> str:
        """Read your private info."""
        return get_private_info(state, memory, caller)

    return [
        recall_,
        remember_,
        get_beliefs_,
        set_belief_,
        get_plan_,
        set_plan_,
        get_public_state_,
        get_private_info_,
    ]


def _kill_vote_terminal(state: GameState, caller: str) -> Callable[..., ToolResult]:
    def submit_kill_vote_(target: str) -> ToolResult:
        """Vote to kill a player tonight."""
        return submit_kill_vote(state, caller, target)

    return submit_kill_vote_


def _exile_vote_terminal(state: GameState, caller: str) -> Callable[..., ToolResult]:
    def submit_exile_vote_(target: str) -> ToolResult:
        """Cast your exile vote."""
        return submit_exile_vote(state, caller, target)

    return submit_exile_vote_


def _step(tool: str, args: dict[str, Any], thought: str = "step") -> dict[str, Any]:
    return {"next_thought": thought, "next_tool_name": tool, "next_tool_args": args}


def _finish(thought: str = "done") -> dict[str, Any]:
    return {"next_thought": thought, "next_tool_name": "finish", "next_tool_args": {}}


def _run(
    *,
    caller: str,
    cognitive_tools: Sequence[Callable[..., str]],
    terminal_tools: dict[str, Callable[..., ToolResult]],
    answers: list[dict[str, Any]],
    decision_brief: str = "Decide your next action.",
    max_iters: int = 10,
) -> Commit:
    return react_decide(
        caller=caller,
        cognitive_tools=cognitive_tools,
        terminal_tools=terminal_tools,
        decision_brief=decision_brief,
        lm=DummyLM(answers),
        max_iters=max_iters,
    )


def test_trivial_terminal_then_finish_commits_value() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("submit_kill_vote", {"target": "Vil1"}, "kill Vil1"),
        _finish(),
    ]
    commit = _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    assert commit == Commit(tool="submit_kill_vote", value="Vil1")


def test_cognitive_calls_before_terminal_show_up_in_trajectory() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("get_public_state", {}),
        _step("recall", {}),
        _step("submit_kill_vote", {"target": "Vil2"}),
        _finish(),
    ]
    lm = DummyLM(answers)
    commit = react_decide(
        caller="Wolf1",
        cognitive_tools=cognitive,
        terminal_tools=terminals,
        decision_brief="kill",
        lm=lm,
        max_iters=10,
    )

    assert commit == Commit(tool="submit_kill_vote", value="Vil2")
    assert len(lm.history) == 4


def test_invalid_terminal_returns_observation_loop_retries() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("submit_kill_vote", {"target": "Wolf1"}),
        _step("submit_kill_vote", {"target": "Vil1"}),
        _finish(),
    ]
    commit = _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    assert commit == Commit(tool="submit_kill_vote", value="Vil1")


def test_finish_without_committing_any_terminal_raises() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [_finish()]
    with pytest.raises(RuntimeError, match=r"Wolf1.*finished without a committed"):
        _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)


def test_max_iters_exceeded_without_finish_raises() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("get_public_state", {}),
        _step("recall", {}),
    ]
    with pytest.raises(RuntimeError, match=r"Wolf1.*finished without a committed"):
        _run(
            caller="Wolf1",
            cognitive_tools=cognitive,
            terminal_tools=terminals,
            answers=answers,
            max_iters=2,
        )


def test_remember_mutates_memory_notes() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("remember", {"note": "Vil1 looked suspicious"}),
        _step("submit_kill_vote", {"target": "Vil1"}),
        _finish(),
    ]
    _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    assert len(memory.notes) == 1
    assert memory.notes[0].text == "Vil1 looked suspicious"
    assert memory.notes[0].round == state.round


def test_set_belief_mutates_memory_beliefs() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step(
            "set_belief",
            {"player": "Vil1", "guess": "villager", "confidence": "low", "evidence": "quiet"},
        ),
        _step("submit_kill_vote", {"target": "Vil1"}),
        _finish(),
    ]
    _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    belief = memory.beliefs["Vil1"]
    assert belief.guess == "villager"
    assert belief.confidence == "low"
    assert belief.evidence == "quiet"


def test_cognitive_observation_starting_with_ok_does_not_terminate() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("set_plan", {"text": "find the seer"}),
        _step("submit_kill_vote", {"target": "Vil1"}),
        _finish(),
    ]
    commit = _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    assert commit.value == "Vil1"
    assert memory.plan == "find the seer"


def test_two_identical_runs_produce_identical_commits() -> None:
    state = _state()
    answers = [
        _step("get_public_state", {}),
        _step("submit_kill_vote", {"target": "Vil1"}),
        _finish(),
    ]

    m1 = GameMemory()
    c1 = _run(
        caller="Wolf1",
        cognitive_tools=_cognitive_closures(state, m1, "Wolf1"),
        terminal_tools={"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")},
        answers=list(answers),
    )

    m2 = GameMemory()
    c2 = _run(
        caller="Wolf1",
        cognitive_tools=_cognitive_closures(state, m2, "Wolf1"),
        terminal_tools={"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")},
        answers=list(answers),
    )

    assert c1 == c2


def test_lm_context_is_restored_after_react_decide() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("submit_kill_vote", {"target": "Vil1"}),
        _finish(),
    ]
    sentinel = DummyLM([{"sentinel": "untouched"}])
    previous_lm = dspy.settings.lm
    try:
        dspy.configure(lm=sentinel)
        _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)
        assert dspy.settings.lm is sentinel
    finally:
        dspy.configure(lm=previous_lm)


def test_exile_vote_terminal_with_abstain_value_is_passed_through() -> None:
    from social_deduction_bench.engine import advance_phase

    day_state = advance_phase(_state())
    memory = GameMemory()
    cognitive = _cognitive_closures(day_state, memory, "Vil1")
    terminals = {"submit_exile_vote": _exile_vote_terminal(day_state, "Vil1")}

    answers = [
        _step("submit_exile_vote", {"target": "abstain"}),
        _finish(),
    ]
    commit = _run(caller="Vil1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    assert commit == Commit(tool="submit_exile_vote", value="abstain")


def test_commit_dataclass_is_frozen_and_equatable() -> None:
    from dataclasses import FrozenInstanceError

    a = Commit(tool="submit_kill_vote", value="Vil1")
    b = Commit(tool="submit_kill_vote", value="Vil1")
    c = Commit(tool="submit_kill_vote", value="Vil2")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    with pytest.raises(FrozenInstanceError):
        a.tool = "x"  # type: ignore[misc]
