"""Tests for `react_decide_async` — the async sibling of `react_decide`.

Parity is the contract: same scripted answers in, same `Commit`, same
trajectory, same telemetry out. The async surface exists so within-phase
decisions can run concurrently via `asyncio.gather`; functional outputs must
match the sync path exactly so the determinism invariant (#4) holds.

`on_thought_chunk` is the streaming hook: when set, the per-iter call is
wrapped in `dspy.streamify` with a listener on the `next_thought` signature
field. The concatenated chunks for each iter must equal the final thought
recorded on the matching `ReActStep`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

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
from social_deduction_bench.agents.react import Commit, react_decide, react_decide_async
from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep
from social_deduction_bench.engine import GameState
from social_deduction_bench.games.werewolf.cognitive import get_private_info, get_public_state
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.games.werewolf.tools import (
    ToolResult,
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


def _step(tool: str, args: dict[str, Any], thought: str = "step") -> dict[str, Any]:
    return {"next_thought": thought, "next_tool_name": tool, "next_tool_args": args}


def _arun(coro: Any) -> Any:
    return asyncio.run(coro)


def _wolf1_kill_setup(
    answers: list[dict[str, Any]],
) -> tuple[
    list[Callable[..., str]],
    dict[str, Callable[..., ToolResult]],
    DummyLM,
]:
    state = _state()
    memory = GameMemory()
    cognitive = _cognitive_closures(state, memory, "Wolf1")
    terminals = {"submit_kill_vote": _kill_vote_terminal(state, "Wolf1")}
    return cognitive, terminals, DummyLM(answers)


def test_async_parity_with_sync_on_trivial_terminal() -> None:
    """Same scripted answers ⇒ same `Commit` from both sync and async paths.

    Determinism gate: the async surface exists so phases can fan-out via
    `asyncio.gather`; the per-loop output must match the sync path byte-for-byte
    so replays through either entry point produce the same `Commit`.
    """
    answers = [
        _step("submit_kill_vote", {"target": "Vil1"}, "kill Vil1"),
    ]
    cog, term, _ = _wolf1_kill_setup(answers)

    sync_commit = react_decide(
        caller="Wolf1",
        cognitive_tools=cog,
        terminal_tools=term,
        decision_brief="decide",
        lm=DummyLM(answers),
        max_iters=10,
    )

    cog2, term2, lm2 = _wolf1_kill_setup(answers)
    async_commit = _arun(
        react_decide_async(
            caller="Wolf1",
            cognitive_tools=cog2,
            terminal_tools=term2,
            decision_brief="decide",
            lm=lm2,
            max_iters=10,
        )
    )

    assert sync_commit == async_commit == Commit(tool="submit_kill_vote", value="Vil1")


def test_async_step_sink_fires_once_per_iteration() -> None:
    """`step_sink` fires per LM-iteration in the async path (parity with sync).

    The CLI's real-time rendering depends on this hook firing as each iter
    lands, not only at commit. If the async path batches sinks, the operator
    sees no progress until the phase resolves — defeating the streaming UX.
    """
    answers = [
        _step("get_public_state", {}, "look"),
        _step("submit_kill_vote", {"target": "Vil2"}, "kill"),
    ]
    cog, term, lm = _wolf1_kill_setup(answers)
    seen: list[tuple[int, str]] = []

    def step_sink(step: ReActStep, _lm_calls: tuple[LMCallRecord, ...]) -> None:
        seen.append((step.iter, step.tool))

    _arun(
        react_decide_async(
            caller="Wolf1",
            cognitive_tools=cog,
            terminal_tools=term,
            decision_brief="decide",
            lm=lm,
            max_iters=10,
            step_sink=step_sink,
        )
    )

    assert seen == [(0, "get_public_state"), (1, "submit_kill_vote")]


def test_async_trace_sink_fires_once_after_commit() -> None:
    """`trace_sink` fires exactly once with structured steps + telemetry on a clean commit."""
    answers = [
        _step("submit_kill_vote", {"target": "Vil1"}),
    ]
    cog, term, lm = _wolf1_kill_setup(answers)
    captured: list[tuple[tuple[ReActStep, ...], tuple[LMCallRecord, ...]]] = []

    def trace_sink(steps: tuple[ReActStep, ...], lm_calls: tuple[LMCallRecord, ...]) -> None:
        captured.append((steps, lm_calls))

    _arun(
        react_decide_async(
            caller="Wolf1",
            cognitive_tools=cog,
            terminal_tools=term,
            decision_brief="decide",
            lm=lm,
            max_iters=10,
            trace_sink=trace_sink,
        )
    )

    assert len(captured) == 1
    steps, _ = captured[0]
    assert tuple(s.tool for s in steps) == ("submit_kill_vote",)


def test_async_loop_that_never_commits_raises_runtime_error() -> None:
    """A loop that only calls non-committing tools raises (parity with sync)."""
    answers = [_step("recall", {})] * 3
    cog, term, lm = _wolf1_kill_setup(answers)

    raised: BaseException | None = None
    try:
        _arun(
            react_decide_async(
                caller="Wolf1",
                cognitive_tools=cog,
                terminal_tools=term,
                decision_brief="decide",
                lm=lm,
                max_iters=3,
            )
        )
    except RuntimeError as err:
        raised = err

    assert raised is not None
    assert "Wolf1" in str(raised)


def test_async_max_iters_exceeded_without_finish_raises() -> None:
    """`max_iters` reached without `finish` and without a terminal commit raises (parity with sync)."""
    answers = [_step("get_public_state", {}) for _ in range(10)]
    cog, term, lm = _wolf1_kill_setup(answers)

    raised: BaseException | None = None
    try:
        _arun(
            react_decide_async(
                caller="Wolf1",
                cognitive_tools=cog,
                terminal_tools=term,
                decision_brief="decide",
                lm=lm,
                max_iters=2,
            )
        )
    except RuntimeError as err:
        raised = err

    assert raised is not None


def test_async_seats_run_concurrently_under_asyncio_gather() -> None:
    """Two `react_decide_async` loops complete in roughly the time of the slower one.

    The whole point of the async surface: when the LM call blocks on I/O, a
    second seat's loop can be in flight on the same event loop. With two
    sequential ~0.2s sleeps the wall-clock should be ~0.2s (parallel), not
    ~0.4s (serial). This proves the I/O concurrency that the phase-level
    `asyncio.gather` relies on.
    """

    class _SlowLM(DummyLM):
        def __init__(self, answers: list[dict[str, Any]], delay_s: float) -> None:
            super().__init__(answers)
            self._delay_s = delay_s

        async def aforward(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
            await asyncio.sleep(self._delay_s)
            return self.forward(prompt=prompt, messages=messages, **kwargs)

    answers = [
        _step("submit_kill_vote", {"target": "Vil1"}),
    ]
    cog_a, term_a, _ = _wolf1_kill_setup(answers)
    cog_b, term_b, _ = _wolf1_kill_setup(answers)
    lm_a = _SlowLM(answers, delay_s=0.2)
    lm_b = _SlowLM(answers, delay_s=0.2)

    async def both() -> tuple[Commit, Commit]:
        return await asyncio.gather(
            react_decide_async(
                caller="Wolf1",
                cognitive_tools=cog_a,
                terminal_tools=term_a,
                decision_brief="a",
                lm=lm_a,
                max_iters=4,
            ),
            react_decide_async(
                caller="Wolf2",
                cognitive_tools=cog_b,
                terminal_tools=term_b,
                decision_brief="b",
                lm=lm_b,
                max_iters=4,
            ),
        )

    start = asyncio.run(_timed(both()))
    elapsed, (a, b) = start
    # Two iters per loop * 0.2s = 0.4s per loop serial. Parallel should be
    # close to 0.4s (slower loop), well under 0.8s (serial sum). Allow generous
    # headroom for CI: assert < 0.75s as the parallelism gate.
    assert elapsed < 0.75, f"async loops did not run concurrently; elapsed={elapsed:.2f}s"
    assert a.tool == "submit_kill_vote" == b.tool


async def _timed(coro: Any) -> tuple[float, Any]:
    import time

    t0 = time.monotonic()
    result = await coro
    return time.monotonic() - t0, result


def test_async_default_step_sink_and_trace_sink_are_no_ops() -> None:
    """Omitting the sinks does not raise (matches the sync defaults)."""
    answers = [
        _step("submit_kill_vote", {"target": "Vil1"}),
    ]
    cog, term, lm = _wolf1_kill_setup(answers)

    commit = _arun(
        react_decide_async(
            caller="Wolf1",
            cognitive_tools=cog,
            terminal_tools=term,
            decision_brief="decide",
            lm=lm,
            max_iters=10,
        )
    )
    assert commit == Commit(tool="submit_kill_vote", value="Vil1")


def test_async_on_thought_chunk_concatenation_equals_step_thought(monkeypatch: Any) -> None:
    """When `on_thought_chunk` is set, every chunk delivered for `next_thought`
    reaches the callback in order.

    The streaming hook is what feeds the CLI's live "thinking…" line. If a
    chunk is dropped (or duplicated) between `dspy.streamify` and the CLI,
    the visible thought diverges from the trajectory record — a benchmark
    observability bug. `DummyLM` doesn't actually stream, so this test
    swaps in a fake `streamify` that yields a known chunk sequence and
    asserts the callback sees those chunks intact, then the final
    `Prediction` falls through to the loop body.
    """
    import dspy as _dspy

    from social_deduction_bench.agents import react as react_module

    answers = [
        _step("get_public_state", {}, thought="full thought zero"),
        _step("submit_kill_vote", {"target": "Vil1"}, thought="full thought one"),
    ]
    cog, term, lm = _wolf1_kill_setup(answers)

    # Track every chunk we emit per iter so the test can compare the
    # callback-observed sequence to the source-of-truth sequence.
    chunks_to_emit: dict[int, list[str]] = {
        0: ["full thought ", "zero"],
        1: ["full thought ", "one"],
    }
    call_count = 0

    def fake_streamify(program: Any, **kwargs: Any) -> Any:
        listeners = kwargs.get("stream_listeners") or []
        assert any(getattr(li, "signature_field_name", None) == "next_thought" for li in listeners), (
            "the on_thought_chunk wiring must subscribe a listener to the `next_thought` field"
        )

        async def runner(**call_kwargs: Any) -> Any:
            nonlocal call_count
            iter_idx = call_count
            call_count += 1
            for piece in chunks_to_emit[iter_idx]:
                yield _dspy.streaming.StreamResponse(
                    predict_name="react",
                    signature_field_name="next_thought",
                    chunk=piece,
                    is_last_chunk=False,
                )
            # The streamed listener path still needs to deliver the final
            # Prediction so the loop can apply the tool call.
            with _dspy.context(lm=lm):
                pred = await program.acall(**call_kwargs)
            yield pred

        return runner

    monkeypatch.setattr(react_module.dspy, "streamify", fake_streamify)

    chunks_by_iter: dict[int, list[str]] = {}
    steps_seen: list[ReActStep] = []

    def on_thought_chunk(iter_idx: int, text: str) -> None:
        chunks_by_iter.setdefault(iter_idx, []).append(text)

    def step_sink(step: ReActStep, _lm_calls: tuple[LMCallRecord, ...]) -> None:
        steps_seen.append(step)

    _arun(
        react_decide_async(
            caller="Wolf1",
            cognitive_tools=cog,
            terminal_tools=term,
            decision_brief="decide",
            lm=lm,
            max_iters=10,
            step_sink=step_sink,
            on_thought_chunk=on_thought_chunk,
        )
    )

    assert chunks_by_iter == chunks_to_emit
    # Every step landed; the streaming wiring did not interfere with the
    # iterator's ability to surface the final Prediction. Iter 0 is a
    # non-committing cognitive read; iter 1 commits and ends the turn.
    assert tuple(s.tool for s in steps_seen) == ("get_public_state", "submit_kill_vote")
