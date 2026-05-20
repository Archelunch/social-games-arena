"""Tests for the per-decision-point ReAct loop primitive (T21).

The primitive is `react_decide(*, caller, cognitive_tools, terminal_tools,
decision_brief, lm, max_iters=10) -> Commit`. It drives `dspy.ReAct.react`
(the per-iteration predictor) directly — NOT `ReAct.forward` — so no extract
step is ever invoked. The terminal-tool wrappers write into a side-band slot
on a valid call; the LLM then calls `finish` to close the loop; we return the
slot's value. A loop that ends with an empty slot (either via `finish` or by
exhausting `max_iters`) raises `RuntimeError` — the contract is fail loud
(CLAUDE.md rule 11).

Tests use `dspy.utils.dummies.DummyLM` in list mode: each entry is the dict
of output-field values the LM should emit on that call. Each ReAct iteration
consumes exactly one entry (`next_thought`/`next_tool_name`/`next_tool_args`).
There is no extract entry. The tests build their cognitive and terminal
closures by hand so the surface covered by the loop is the real one — real
`GameMemory`, real cognitive tools from `agents/cognitive.py`, real
game-action tools from `games/werewolf/tools.py`.
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

# --------------------------- fixtures -------------------------------------


def _roster() -> tuple[tuple[str, str], ...]:
    """A small but realistic roster: 2 wolves, 1 seer, 1 doctor, 2 villagers.

    Keeping the roster compact keeps test failures readable; the loop primitive
    does not care how many players exist.
    """
    return (
        ("Wolf1", Role.WEREWOLF.value),
        ("Wolf2", Role.WEREWOLF.value),
        ("Seer1", Role.SEER.value),
        ("Doc1", Role.DOCTOR.value),
        ("Vil1", Role.VILLAGER.value),
        ("Vil2", Role.VILLAGER.value),
    )


def _state() -> GameState:
    """A canonical night-1 state used by most tests."""
    return GameState.initial(_roster())


# --------------------------- closure factories ----------------------------
#
# These mirror what `agents/decisions.py` will do for the production wiring
# (one closure per cognitive tool, one per terminal). Done by hand here so the
# tests exercise the real cognitive-tool layer and not a re-binding helper.


def _cognitive_closures(state: GameState, memory: GameMemory, caller: str) -> list[Callable[..., str]]:
    """Bind `(state, memory, caller)` into each cognitive tool, return ordered list."""

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
        # `confidence` arrives as a plain str from the LM; `set_belief` validates.
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


# --------------------------- LM scripting helpers -------------------------


def _step(tool: str, args: dict[str, Any], thought: str = "step") -> dict[str, Any]:
    """One ReAct iteration's output-field block."""
    return {"next_thought": thought, "next_tool_name": tool, "next_tool_args": args}


def _finish(thought: str = "done") -> dict[str, Any]:
    """The terminating `finish` tool call."""
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
    """Run `react_decide` with a `DummyLM` scripted from `answers`."""
    return react_decide(
        caller=caller,
        cognitive_tools=cognitive_tools,
        terminal_tools=terminal_tools,
        decision_brief=decision_brief,
        lm=DummyLM(answers),
        max_iters=max_iters,
    )


# --------------------------- tests ----------------------------------------


def test_trivial_terminal_then_finish_commits_value() -> None:
    """A single game-action call followed by `finish` returns its parsed value.

    This is the minimum happy path: the LLM does no cognition, just commits.
    The loop must propagate the `ToolResult.value` ('Vil1') into the returned
    `Commit`, and `Commit.tool` must name the terminator used.
    """
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
    """Cognitive tools called before the terminal are non-terminating.

    The loop must run cognition (here `get_public_state` then `recall`) and
    only commit on the terminal call — the test pins that ordering by reading
    the LM's call log (`DummyLM` consumes one answer per LM call). If a
    cognitive return value of `"ok: ..."` or anything else terminated, the LM
    would never reach the terminal answer.
    """
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
    # Four LM calls: get_public_state, recall, submit_kill_vote, finish.
    assert len(lm.history) == 4


def test_invalid_terminal_returns_observation_loop_retries() -> None:
    """An invalid game-action call surfaces as `error: ...` and the loop continues.

    Self-target is rejected by `submit_kill_vote` (T15 invariant). The first
    call writes nothing to the side-band slot; the second call succeeds. The
    returned `Commit.value` must be the *successful* target, not the rejected
    one. This is the contract that lets a real LLM recover from a hallucinated
    target without aborting the game.
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [
        _step("submit_kill_vote", {"target": "Wolf1"}),  # self-target, rejected
        _step("submit_kill_vote", {"target": "Vil1"}),  # legal
        _finish(),
    ]
    commit = _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)

    assert commit == Commit(tool="submit_kill_vote", value="Vil1")


def test_finish_without_committing_any_terminal_raises() -> None:
    """If the LLM finishes without committing a game action, fail loud.

    The loop is the agent's commitment surface; a silent "no decision" would
    break the driver, which requires an action per acting player per phase.
    Raising `RuntimeError` lets the caller surface the failure rather than
    treating an empty commit as a default ABSTAIN.
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    answers = [_finish()]
    with pytest.raises(RuntimeError, match=r"Wolf1.*finished without a committed"):
        _run(caller="Wolf1", cognitive_tools=cognitive, terminal_tools=terminals, answers=answers)


def test_max_iters_exceeded_without_finish_raises() -> None:
    """If the loop hits `max_iters` before `finish`, no commit happened — raise.

    Same root contract as the finish-without-commit case: an empty side-band
    slot must surface as a `RuntimeError`. The loop simply exits the
    iteration cap; whether the LLM was going to call `finish` next is
    irrelevant — without a commit, the driver cannot proceed.
    """
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}

    # Two cognitive calls fill `max_iters=2`; the for-loop exits without ever
    # reaching `finish`.
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
    """A scripted `remember` call appends to the player's memory.

    The cognitive layer mutates the agent's `GameMemory`; the loop must hand
    that mutation through so the agent can read it back later via `recall`.
    Pinning the count + payload guards against a wrap-layer copy that breaks
    side-effects.
    """
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
    """A scripted `set_belief` writes the row through to the player's memory.

    Same contract as `test_remember_mutates_memory_notes` but for the
    structured belief table — beliefs are the LLM's explicit suspicion record
    (§8) and would be useless if the loop discarded the mutation.
    """
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
    """A cognitive tool's `"ok: ..."`-shaped string does not look like a commit.

    Terminals signal commitment via the side-band slot, not via their string
    return. To pin that the wrapper layer is not pattern-matching observation
    text, we drive a scripted `set_plan` (which returns `"ok: plan set"`) and
    confirm the loop still runs to the explicit terminal afterwards.
    """
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
    """Same script + same inputs → byte-identical `Commit` (determinism, invariant #4).

    Two independent `GameMemory`s built from the same op sequence must converge.
    """
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
    """`react_decide` must use `dspy.context`, not `dspy.configure`.

    A global `dspy.configure` would leak the test's `DummyLM` into the next
    test or, worse, into a real LLM call. Pin the contract by installing a
    sentinel LM globally first, running the loop with a different LM, then
    asserting the sentinel is still in place — a tautology check (`None is
    None`) would not catch a `dspy.configure` regression.
    """
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
    """A second terminal type (`submit_exile_vote`) terminates the same way.

    The loop primitive is terminator-agnostic — any name in `terminal_tools`
    can commit. Pin this with an exile-vote scenario in the day phase so we
    are not implicitly hard-coding the night-kill case.
    """
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
    """`Commit` is a public value type — frozen, hashable, structurally equal.

    Callers (T22's `DecisionSource` adapter) assemble `NightActions` /
    `DayActions` from `Commit` values; equality semantics anchor those
    aggregations. Frozenness blocks accidental mutation by downstream code.
    """
    from dataclasses import FrozenInstanceError

    a = Commit(tool="submit_kill_vote", value="Vil1")
    b = Commit(tool="submit_kill_vote", value="Vil1")
    c = Commit(tool="submit_kill_vote", value="Vil2")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    with pytest.raises(FrozenInstanceError):
        a.tool = "x"  # type: ignore[misc]
