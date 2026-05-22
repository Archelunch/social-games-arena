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
    werewolf_chat,
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


def _chat_intermediate(state: GameState, caller: str) -> Callable[..., ToolResult]:
    def werewolf_chat_(message: str) -> ToolResult:
        """Send a private message to your fellow werewolves."""
        return werewolf_chat(state, caller, message)

    return werewolf_chat_


def _exile_vote_terminal(state: GameState, caller: str) -> Callable[..., ToolResult]:
    def submit_exile_vote_(target: str) -> ToolResult:
        """Cast your exile vote."""
        return submit_exile_vote(state, caller, target)

    return submit_exile_vote_


def _step(tool: str, args: dict[str, Any], thought: str = "step") -> dict[str, Any]:
    return {"next_thought": thought, "next_tool_name": tool, "next_tool_args": args}


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
    ]
    commit = _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    assert commit == Commit(tool="submit_kill_vote", value="Vil1")


def test_react_predict_omits_the_finish_tool() -> None:
    """Our finish-free predictor must not carry DSPy's auto-injected `finish`.

    `dspy.ReAct` injects a `finish` tool into both the `next_tool_name` choices
    and the instructions; re-introducing it would re-add the dead second
    round-trip P2 removed. This guards the tool table and the choice set.
    """
    from typing import get_args

    from social_deduction_bench.agents.react import _build_react_predict, _build_signature

    def submit_kill_vote(target: str) -> str:
        """Cast your kill vote."""
        return "ok"

    def recall() -> str:
        """Recall recent events."""
        return "ok"

    predict, tools_by_name = _build_react_predict(_build_signature(), [submit_kill_vote, recall])

    assert set(tools_by_name) == {"submit_kill_vote", "recall"}
    assert "finish" not in tools_by_name
    signature = predict.signature
    assert signature is not None
    next_tool_choices = set(get_args(signature.output_fields["next_tool_name"].annotation))
    assert next_tool_choices == {"submit_kill_vote", "recall"}


def test_cognitive_calls_before_terminal_show_up_in_trajectory() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("get_public_state", {}),
        _step("recall", {}),
        _step("submit_kill_vote", {"target": "Vil2"}),
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
    # Two cognitive reads then the terminal commit = 3 LM calls; the commit
    # ends the turn (no separate finish round-trip).
    assert len(lm.history) == 3


def test_invalid_terminal_returns_observation_loop_retries() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("submit_kill_vote", {"target": "Wolf1"}),
        _step("submit_kill_vote", {"target": "Vil1"}),
    ]
    commit = _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    assert commit == Commit(tool="submit_kill_vote", value="Vil1")


def test_loop_that_never_commits_a_terminal_raises() -> None:
    """A loop that only ever calls non-committing tools fails loud.

    There is no `finish` tool to end the turn early — only a terminal commit
    does. So a model that spends every iteration on cognitive reads exhausts
    `max_iters` without a `Commit`, which must raise rather than silently
    returning nothing (CLAUDE.md rule 11: fail loud).
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [_step("recall", {})] * 3
    with pytest.raises(RuntimeError, match=r"Wolf1.*finished without a committed"):
        _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers, max_iters=3)


def test_unparseable_lm_response_is_converted_to_a_no_commit_runtime_error() -> None:
    """A truncated / unparseable LM response ends the loop as a clean `RuntimeError`,
    not a raw `AdapterParseError`.

    A too-small `max_tokens` truncates the response mid-`next_thought`, so the
    required output fields never appear and the adapter raises
    `AdapterParseError` (a plain `Exception`, not `ValueError`). Before this was
    caught it propagated out of `run_game` and crashed a live game. The loop now
    converts any such parse failure into the same no-commit `RuntimeError` every
    other dead-end produces, so callers handle it uniformly (a vote exits cleanly;
    a day reaction degrades to a pass). An empty `DummyLM` answer queue triggers
    the sentinel response the JSON adapter cannot parse — and the raised error
    must NOT be an `AdapterParseError`.
    """
    from dspy.utils.exceptions import AdapterParseError

    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    with pytest.raises(RuntimeError, match=r"Wolf1.*finished without a committed") as excinfo:
        _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=[], max_iters=2)
    assert not isinstance(excinfo.value, AdapterParseError)


def test_remember_mutates_memory_notes() -> None:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("remember", {"note": "Vil1 looked suspicious"}),
        _step("submit_kill_vote", {"target": "Vil1"}),
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
    ]
    commit = _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    assert commit.value == "Vil1"
    assert memory.plan == "find the seer"


def test_two_identical_runs_produce_identical_commits() -> None:
    state = _state()
    answers = [
        _step("get_public_state", {}),
        _step("submit_kill_vote", {"target": "Vil1"}),
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


def test_intermediate_tool_call_does_not_terminate_the_loop() -> None:
    """A valid `intermediate_tools` call does NOT commit; the loop continues to the terminal.

    Intermediate tools (e.g. `werewolf_chat`) emit a side-effect (an event)
    and the LLM keeps going until a terminal tool fires. The final `Commit`
    must come from the terminal, never from the intermediate.
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    intermediates = {"werewolf_chat": _chat_intermediate(state, "Wolf1")}
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("werewolf_chat", {"message": "hunt the seer"}, "talk first"),
        _step("submit_kill_vote", {"target": "Vil1"}, "then vote"),
    ]
    commit = react_decide(
        caller="Wolf1",
        cognitive_tools=cognitive,
        intermediate_tools=intermediates,
        terminal_tools=terminals,
        decision_brief="kill",
        lm=DummyLM(answers),
        max_iters=10,
    )

    assert commit == Commit(tool="submit_kill_vote", value="Vil1")


def test_intermediate_tool_rejection_fires_on_reject_and_loop_retries() -> None:
    """A rejected intermediate call fires `on_reject(tool, args, reason)` and continues.

    The same callback fires for terminal rejections — this pins the
    intermediate path. The error observation goes back to the LLM as before so
    it can self-correct, and the loop is not aborted.
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    intermediates = {"werewolf_chat": _chat_intermediate(state, "Wolf1")}
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}
    rejections: list[tuple[str, dict[str, Any], str]] = []

    def on_reject(tool: str, args: dict[str, Any], reason: str) -> None:
        rejections.append((tool, args, reason))

    answers = [
        _step("werewolf_chat", {"message": "   "}, "blank chat"),  # rejected: empty message
        _step("werewolf_chat", {"message": "real talk"}, "retry chat"),
        _step("submit_kill_vote", {"target": "Vil1"}, "vote"),
    ]
    commit = react_decide(
        caller="Wolf1",
        cognitive_tools=cognitive,
        intermediate_tools=intermediates,
        terminal_tools=terminals,
        decision_brief="kill",
        lm=DummyLM(answers),
        max_iters=10,
        on_reject=on_reject,
    )

    assert commit == Commit(tool="submit_kill_vote", value="Vil1")
    assert len(rejections) == 1
    tool, args, reason = rejections[0]
    assert tool == "werewolf_chat"
    assert args == {"message": "   "}
    assert "non-empty" in reason


def test_terminal_tool_rejection_fires_on_reject_callback() -> None:
    """A rejected terminal call fires `on_reject` with the engine's rejection reason.

    The same callback path as intermediates — pins one rejection per invalid
    call, regardless of tool category. The LLM still sees the `error:`
    observation and the loop continues until a valid terminal commits.
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}
    rejections: list[tuple[str, dict[str, Any], str]] = []

    def on_reject(tool: str, args: dict[str, Any], reason: str) -> None:
        rejections.append((tool, args, reason))

    answers = [
        _step("submit_kill_vote", {"target": "Wolf1"}),  # self-target, rejected
        _step("submit_kill_vote", {"target": "Vil1"}),  # valid
    ]
    react_decide(
        caller="Wolf1",
        cognitive_tools=cognitive,
        terminal_tools=terminals,
        decision_brief="kill",
        lm=DummyLM(answers),
        max_iters=10,
        on_reject=on_reject,
    )

    assert len(rejections) == 1
    tool, args, reason = rejections[0]
    assert tool == "submit_kill_vote"
    assert args == {"target": "Wolf1"}
    assert "Wolf1" in reason


def test_on_reject_not_invoked_on_a_clean_run() -> None:
    """A run with no rejections never fires `on_reject`.

    The callback is invariant-protected: no spurious calls on valid trajectories.
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}
    rejections: list[tuple[str, dict[str, Any], str]] = []

    answers = [
        _step("submit_kill_vote", {"target": "Vil1"}),
    ]
    react_decide(
        caller="Wolf1",
        cognitive_tools=cognitive,
        terminal_tools=terminals,
        decision_brief="kill",
        lm=DummyLM(answers),
        max_iters=10,
        on_reject=lambda tool, args, reason: rejections.append((tool, args, reason)),
    )

    assert rejections == []


def test_default_on_reject_is_a_no_op() -> None:
    """`react_decide` accepts a rejected call without an `on_reject` callback.

    Back-compat with T22 tests that don't pass the new keyword: a rejected
    terminal still produces an `error:` observation the LLM sees and a retry
    can still commit. The default `None` callback must be a quiet no-op.
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("submit_kill_vote", {"target": "Wolf1"}),
        _step("submit_kill_vote", {"target": "Vil1"}),
    ]
    commit = _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)
    assert commit.value == "Vil1"


# --- T30: `trace_sink` plumbing ------------------------------------------


def test_default_trace_sink_is_a_no_op() -> None:
    """`react_decide` runs and commits exactly as before when `trace_sink` is omitted.

    Back-compat guard for every existing call site (e.g. T22 tests and
    the adapter's pre-T30 wiring): passing no sink must leave the legacy
    commit-return contract byte-identical.
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("submit_kill_vote", {"target": "Vil1"}),
    ]
    commit = react_decide(
        caller="Wolf1",
        cognitive_tools=cognitive,
        terminal_tools=terminals,
        decision_brief="kill",
        lm=DummyLM(answers),
        max_iters=10,
        # Note: no trace_sink kwarg.
    )

    assert commit == Commit(tool="submit_kill_vote", value="Vil1")


def test_trace_sink_fires_once_with_structured_steps_and_lm_calls() -> None:
    """`trace_sink` fires exactly once per successful commit with structured records.

    The sidecar (T30) needs both the agent's reasoning trace and the LM
    telemetry, paired per-iteration. The sink contract is: one call after
    `finish`, receiving the full ReAct trajectory and the LM-history slice
    for the loop.
    """
    from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep

    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    captured: list[tuple[tuple[ReActStep, ...], tuple[LMCallRecord, ...]]] = []

    def sink(steps: tuple[ReActStep, ...], calls: tuple[LMCallRecord, ...]) -> None:
        captured.append((steps, calls))

    answers = [
        _step("get_public_state", {}, thought="orient"),
        _step("submit_kill_vote", {"target": "Vil2"}, thought="kill Vil2"),
    ]
    lm = DummyLM(answers)
    commit = react_decide(
        caller="Wolf1",
        cognitive_tools=cognitive,
        terminal_tools=terminals,
        decision_brief="kill",
        lm=lm,
        max_iters=10,
        trace_sink=sink,
    )

    assert commit == Commit(tool="submit_kill_vote", value="Vil2")
    assert len(captured) == 1, "trace_sink must fire exactly once per commit"

    steps, calls = captured[0]
    # Iteration count matches the LM-history growth — one ReActStep per loop iteration.
    assert len(steps) == len(lm.history)
    # Step <-> LM-call symmetry. If these diverge, the per-iteration pairing is
    # broken and telemetry rows misalign with thoughts in the replay UI (T31).
    assert len(steps) == len(calls)
    assert tuple(step.iter for step in steps) == tuple(range(len(steps)))
    # The committed terminal shows up as the last non-finish step.
    assert any(step.tool == "submit_kill_vote" and step.args["target"] == "Vil2" for step in steps)
    # Each LM call carries a record. DummyLM emits dummy telemetry.
    for call in calls:
        assert call.model == "dummy"
        assert call.prompt_tokens == 0
        assert call.completion_tokens == 0
        assert call.cost_usd is None
        assert isinstance(call.latency_ms, float)
        assert call.latency_ms >= 0.0


def test_trace_sink_does_not_fire_when_loop_fails_to_commit() -> None:
    """On `RuntimeError` (no commit), `trace_sink` is NOT called.

    "One sidecar line per committed decision" stays simple: a failed loop
    has no `Commit` and therefore no `Trajectory`. T24's illegal-move metric
    will count failures via `TOOL_REJECTED` events and the loop's
    `RuntimeError`, not via partial trajectories.
    """
    from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep

    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    captured: list[tuple[tuple[ReActStep, ...], tuple[LMCallRecord, ...]]] = []

    def sink(steps: tuple[ReActStep, ...], calls: tuple[LMCallRecord, ...]) -> None:
        captured.append((steps, calls))

    answers = [_step("recall", {})] * 3  # cognitive-only: never commits a terminal
    with pytest.raises(RuntimeError, match=r"Wolf1.*finished without a committed"):
        react_decide(
            caller="Wolf1",
            cognitive_tools=cognitive,
            terminal_tools=terminals,
            decision_brief="kill",
            lm=DummyLM(answers),
            max_iters=3,
            trace_sink=sink,
        )

    assert captured == []


def test_trace_sink_captures_rejection_observation_in_step() -> None:
    """A rejected tool call surfaces as an `error:`-prefixed observation in `ReActStep`.

    The audit trail must show the LLM's bad-call attempt as a real step
    (with the rejection observation) before the retry that committed.
    Without this, the sidecar would hide the agent's mistakes.
    """
    from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep

    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    captured: list[tuple[tuple[ReActStep, ...], tuple[LMCallRecord, ...]]] = []

    def sink(steps: tuple[ReActStep, ...], calls: tuple[LMCallRecord, ...]) -> None:
        captured.append((steps, calls))

    answers = [
        _step("submit_kill_vote", {"target": "Wolf1"}),  # self-target -> rejected
        _step("submit_kill_vote", {"target": "Vil1"}),  # valid
    ]
    react_decide(
        caller="Wolf1",
        cognitive_tools=cognitive,
        terminal_tools=terminals,
        decision_brief="kill",
        lm=DummyLM(answers),
        max_iters=10,
        trace_sink=sink,
    )

    steps, _ = captured[0]
    # The first step's observation starts with "error:" (rejection); the second
    # step's starts with "ok:" (committed).
    rejection_step = next(s for s in steps if s.tool == "submit_kill_vote" and s.args["target"] == "Wolf1")
    success_step = next(s for s in steps if s.tool == "submit_kill_vote" and s.args["target"] == "Vil1")
    assert rejection_step.observation.startswith("error:")
    assert success_step.observation.startswith("ok:")


def test_lm_telemetry_survives_non_numeric_usage_fields() -> None:
    """A provider that returns `None` for `prompt_tokens` does not abort the loop.

    Regression guard: telemetry is observability, never a kill-switch. The
    coercion helper must downgrade non-numeric usage values to `0` so a
    quirky provider response cannot abort a live game mid-decision.
    """
    from social_deduction_bench.agents.react import _coerce_int

    assert _coerce_int(42) == 42
    assert _coerce_int(3.7) == 3
    assert _coerce_int(None) == 0
    assert _coerce_int("not a number") == 0
    assert _coerce_int(True) == 0  # bool is int — exclude as a usage signal
    assert _coerce_int(False) == 0


def test_lm_calls_for_iteration_logs_warning_on_empty_history(caplog: pytest.LogCaptureFixture) -> None:
    """An iteration that appends no LM-history entries logs a warning.

    A cache hit or `settings.disable_history` results in zero new history
    entries. The function returns `[]` so telemetry is missing but loud —
    silent drops would hide the cause (cache vs. bug) at the replay layer.
    """
    import logging

    from social_deduction_bench.agents.react import _lm_calls_for_iteration

    with caplog.at_level(logging.WARNING, logger="social_deduction_bench.agents.react"):
        records = _lm_calls_for_iteration([], elapsed_ms=1.0)

    assert records == []
    assert any("no LM-history entries" in record.message for record in caplog.records)


def test_lm_calls_for_iteration_divides_elapsed_across_multi_entries(caplog: pytest.LogCaptureFixture) -> None:
    """A multi-entry slice splits elapsed_ms evenly and warns.

    A provider with internal retries can append more than one history
    entry per `react.react` call. Splitting latency evenly conserves the
    total attributed to the iteration; a warning surfaces the unexpected
    shape for debugging.
    """
    import logging

    from social_deduction_bench.agents.react import _lm_calls_for_iteration

    history_slice: list[dict[str, object]] = [
        {"model": "m1", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "cost": 0.001, "uuid": "u1"},
        {"model": "m2", "usage": {"prompt_tokens": 20, "completion_tokens": 8}, "cost": 0.002, "uuid": "u2"},
    ]
    with caplog.at_level(logging.WARNING, logger="social_deduction_bench.agents.react"):
        records = _lm_calls_for_iteration(history_slice, elapsed_ms=100.0)

    assert len(records) == 2
    assert records[0].latency_ms == 50.0
    assert records[1].latency_ms == 50.0
    assert records[0].prompt_tokens == 10
    assert records[1].prompt_tokens == 20
    assert any("2 LM-history entries" in record.message for record in caplog.records)


def test_trace_sink_survives_bounded_lm_history_rotation() -> None:
    """`react_decide` captures LM calls correctly even when `lm.history` rotates.

    The original `len(lm.history)`-slicing approach silently mis-attributed
    entries when DSPy's `settings.max_history_size` rotation popped older
    entries. The uuid-marker scheme is robust: each iteration captures
    only entries with uuids not in the pre-iteration set.

    Simulate rotation by manually popping `lm.history` between iterations;
    the recorded steps must still equal the iteration count.
    """
    from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep

    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    captured: list[tuple[tuple[ReActStep, ...], tuple[LMCallRecord, ...]]] = []

    def sink(steps: tuple[ReActStep, ...], calls: tuple[LMCallRecord, ...]) -> None:
        captured.append((steps, calls))

    # Class that wraps DummyLM but rotates history (drops oldest) after each call.
    class _RotatingLM(DummyLM):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            out = super().__call__(*args, **kwargs)
            # Keep only the last 1 entry — simulates a bounded-history rotation.
            if len(self.history) > 1:
                del self.history[: len(self.history) - 1]
            return out

    answers = [
        _step("get_public_state", {}, thought="orient"),
        _step("submit_kill_vote", {"target": "Vil1"}, thought="kill"),
    ]
    lm = _RotatingLM(answers)
    react_decide(
        caller="Wolf1",
        cognitive_tools=cognitive,
        terminal_tools=terminals,
        decision_brief="kill",
        lm=lm,
        max_iters=10,
        trace_sink=sink,
    )

    steps, calls = captured[0]
    # Both iterations recorded (orient + the committing kill); the rotation
    # does NOT lose records.
    assert len(steps) == 2
    # Even though history was constantly rotated down to 1 entry, each
    # iteration's call was captured (the uuid-marker scheme decoupled
    # iteration capture from history length).
    assert len(calls) == 2


def test_react_step_args_is_immutable_after_construction() -> None:
    """`ReActStep.args` cannot be mutated after the step is recorded.

    The audit trail in the sidecar must be tamper-proof — a mutable inner
    mapping would let a later phase silently rewrite history. The `args`
    must reject `step.args["target"] = "X"` with `TypeError`.
    """
    from social_deduction_bench.agents.trajectory import ReActStep

    step = ReActStep(
        iter=0,
        thought="t",
        tool="submit_bid",
        args={"amount": 5},
        observation="ok",
    )
    with pytest.raises(TypeError):
        step.args["amount"] = 99  # type: ignore[index]
