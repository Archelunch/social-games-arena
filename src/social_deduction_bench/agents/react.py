"""Per-decision-point ReAct loop primitive.

Builds its own finish-free ReAct predictor (mirroring `dspy.ReAct`'s signature
construction but without the auto-injected `finish` tool) and drives it one
iteration at a time. The commitment is captured side-band the instant a terminal
tool reports a valid `ToolResult`, so the loop stops on commit — there is no
separate `finish` round-trip (which DSPy's `ReAct` advertises and which costs a
dead second LM call) and no `extract` call.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import dspy
from dspy.clients.base_lm import BaseLM
from dspy.streaming import StreamListener, StreamResponse
from dspy.utils.exceptions import AdapterParseError

from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep
from social_deduction_bench.games.werewolf.tools import ToolResult

logger = logging.getLogger(__name__)

RejectCallback = Callable[[str, dict[str, object], str], None]
"""Signature for `react_decide`'s `on_reject` hook: `(tool, args, reason) -> None`."""

TraceSink = Callable[[tuple[ReActStep, ...], tuple[LMCallRecord, ...]], None]
"""Signature for `react_decide`'s `trace_sink`: receives the per-iteration steps
and the per-LM-call telemetry of a completed loop. Fires once after a
successful commit; not called when the loop raises `RuntimeError`."""

StepSink = Callable[[ReActStep, tuple[LMCallRecord, ...]], None]
"""Signature for `react_decide`'s `step_sink`: fires once per iteration as soon
as the LM call returns and the iteration's `ReActStep` is built. Used by the
CLI to stream agent progress in real time — without it the user sees nothing
between the phase banner and the eventual commit (potentially 30+ seconds for
a small model). Fires for every iteration including `finish`."""

ThoughtChunkCallback = Callable[[int, str], None]
"""Signature for `react_decide_async`'s `on_thought_chunk` hook: receives
`(iter_idx, text_chunk)` for each streamed chunk of the `next_thought`
signature field. Fires zero or more times per iteration; the concatenated
chunks for one iter equal that iter's `ReActStep.thought`. When set, the
per-iter call is routed through `dspy.streamify`; when `None`, the loop
uses plain `acall` and no chunks are emitted."""

_EMPTY_INTERMEDIATES: Mapping[str, Callable[..., ToolResult]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Commit:
    """The decision a player-agent's loop committed to this turn."""

    tool: str
    value: object


class _Slot:
    """Mutable side-band the terminal wrappers write into on a valid call."""

    __slots__ = ("committed", "tool_name", "value")

    def __init__(self) -> None:
        self.committed: bool = False
        self.tool_name: str | None = None
        self.value: object = None


def _rename_cognitive(fn: Callable[..., str]) -> Callable[..., str]:
    """Strip a single trailing underscore from a cognitive closure's `__name__`.

    A caller that binds `(state, memory, caller)` with a closure named `recall_`
    (to avoid shadowing the imported `recall`) still gets `recall` as the
    LLM-facing tool name.
    """
    name = fn.__name__
    if not name.endswith("_"):
        return fn
    renamed_name = name.rstrip("_")

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> str:
        return fn(*args, **kwargs)

    wrapper.__name__ = renamed_name
    wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return wrapper


def _wrap_terminal(
    name: str,
    fn: Callable[..., ToolResult],
    slot: _Slot,
    on_reject: RejectCallback | None,
) -> Callable[..., str]:
    """Wrap a terminal tool: capture a valid commit into `slot`, surface a rejection through `on_reject`."""

    @functools.wraps(fn)
    def wrapper(**kwargs: object) -> str:
        result = fn(**kwargs)
        if result.valid:
            slot.committed = True
            slot.tool_name = name
            slot.value = result.value
            return f"ok: {name} committed with {kwargs}"
        if on_reject is not None:
            on_reject(name, dict(kwargs), result.reason)
        return f"error: {result.reason}"

    wrapper.__name__ = name
    wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return wrapper


def _wrap_intermediate(
    name: str,
    fn: Callable[..., ToolResult],
    on_reject: RejectCallback | None,
) -> Callable[..., str]:
    """Wrap an intermediate game-action tool.

    A valid call returns an `ok:` observation and the loop continues — no
    commit slot is set. A rejection fires `on_reject` and returns an `error:`
    observation. Intermediate tools never terminate the loop; only `terminal`
    tools (plus the implicit `finish`) do.
    """

    @functools.wraps(fn)
    def wrapper(**kwargs: object) -> str:
        result = fn(**kwargs)
        if result.valid:
            return f"ok: {name} called with {kwargs}"
        if on_reject is not None:
            on_reject(name, dict(kwargs), result.reason)
        return f"error: {result.reason}"

    wrapper.__name__ = name
    wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return wrapper


def _build_signature() -> type[dspy.Signature]:
    class DecisionSignature(dspy.Signature):
        """You are playing Werewolf. Decide and commit one game action by calling the appropriate tool."""

        decision_brief: str = dspy.InputField()
        committed_action: str = dspy.OutputField()

    return DecisionSignature


def _build_react_predict(
    signature: type[dspy.Signature], tools: Sequence[Callable[..., Any]]
) -> tuple[dspy.Predict, dict[str, dspy.Tool]]:
    """Build a finish-free ReAct predictor and its tool table.

    Mirrors `dspy.ReAct.__init__`'s signature construction but omits the
    auto-injected `finish` tool. A committing game-action tool ends the turn
    (the loop breaks on the captured commit), so there is nothing for a separate
    `finish` step to do — and advertising it (in the `next_tool_name` Literal and
    the instructions) is exactly what makes a model plan a wasted second
    round-trip. Cognitive tools are non-committing, so the loop simply continues
    after them.
    """
    tool_objs = [t if isinstance(t, dspy.Tool) else dspy.Tool(t) for t in tools]
    # `Tool.name` is typed `str | None`; our tools always carry a name, so coerce
    # to keep the table keyed by `str` (and the Literal below well-formed).
    tools_by_name: dict[str, dspy.Tool] = {str(t.name): t for t in tool_objs}

    inputs = ", ".join(f"`{k}`" for k in signature.input_fields)
    instr = [f"{signature.instructions}\n"] if signature.instructions else []
    instr.extend(
        [
            f"You are an agent taking a single turn. You are given {inputs} and your past trajectory so far.",
            "Use one or more of the supplied tools to decide and commit your move.",
            "Each turn, produce next_thought (your reasoning), next_tool_name, and next_tool_args; "
            "after each tool call you receive an observation appended to your trajectory.",
            "Your turn ends the moment you call the tool that commits your game action — "
            "there is no separate finish step.",
            "When selecting next_tool_name and next_tool_args, the tool must be one of:\n",
        ]
    )
    for idx, tool in enumerate(tools_by_name.values()):
        instr.append(f"({idx + 1}) {tool}")
    instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

    # Mirrors `dspy.ReAct.__init__`'s runtime construction. The static checker
    # can't model `Signature(fields, instructions)`'s positional call or a Literal
    # built from a runtime tuple; both are the documented DSPy idiom.
    react_signature = (
        dspy.Signature({**signature.input_fields}, "\n".join(instr))  # type: ignore[bad-argument-count]
        .append("trajectory", dspy.InputField(), type_=str)
        .append("next_thought", dspy.OutputField(), type_=str)
        .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools_by_name.keys())])  # type: ignore[invalid-literal]
        .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
    )
    return dspy.Predict(react_signature), tools_by_name


def _format_trajectory(trajectory: dict[str, object]) -> str:
    """Render the trajectory dict as labeled `[[ ## key ## ]]\\n{value}` blocks."""
    lines: list[str] = []
    for key, value in trajectory.items():
        lines.append(f"[[ ## {key} ## ]]\n{value}")
    return "\n\n".join(lines)


def _coerce_int(value: object) -> int:
    """Return `value` as `int` on a numeric shape; `0` on any non-numeric.

    Provider responses can hand back `None`, missing usage fields, or a
    stringified number that `litellm` failed to normalize. Telemetry is
    observability — a non-numeric must not crash `react_decide` and kill a
    live game. Mirrors the `cost_usd` shape guard below.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _lm_calls_for_iteration(history_slice: Sequence[Mapping[str, object]], elapsed_ms: float) -> list[LMCallRecord]:
    """Build one `LMCallRecord` per LM-history entry the iteration appended.

    Almost always one entry (one LM call per iteration). If the slice is
    empty (no provider call landed: cache short-circuit or
    `settings.disable_history`), no record is emitted and a warning logs.
    If it carries more than one entry (unexpected provider retry), the
    elapsed time is divided evenly across them so the total latency
    attributed to the iteration is conserved; a warning logs.
    """
    if not history_slice:
        logger.warning("react iteration appended no LM-history entries (cache hit or disable_history?)")
        return []
    if len(history_slice) > 1:
        logger.warning("react iteration appended %d LM-history entries; dividing latency evenly", len(history_slice))
    per_call_ms = elapsed_ms / len(history_slice)
    records: list[LMCallRecord] = []
    for entry in history_slice:
        usage = entry.get("usage") or {}
        cost = entry.get("cost")
        cost_usd: float | None = float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else None
        records.append(
            LMCallRecord(
                model=str(entry.get("model", "")),
                prompt_tokens=_coerce_int(usage.get("prompt_tokens")) if isinstance(usage, Mapping) else 0,
                completion_tokens=_coerce_int(usage.get("completion_tokens")) if isinstance(usage, Mapping) else 0,
                latency_ms=per_call_ms,
                cost_usd=cost_usd,
            )
        )
    return records


def react_decide(
    *,
    caller: str,
    cognitive_tools: Sequence[Callable[..., str]],
    terminal_tools: Mapping[str, Callable[..., ToolResult]],
    decision_brief: str,
    lm: BaseLM,
    intermediate_tools: Mapping[str, Callable[..., ToolResult]] = _EMPTY_INTERMEDIATES,
    max_iters: int = 10,
    on_reject: RejectCallback | None = None,
    trace_sink: TraceSink | None = None,
    step_sink: StepSink | None = None,
) -> Commit:
    """Run one ReAct decision under a scoped LM and return the committed action.

    Sync entry point. Drives the same loop as `react_decide_async` (no token
    streaming) by spinning up a private event loop with `asyncio.run`. The
    output is byte-identical to the async path for the same scripted LM —
    determinism (invariant #4) is preserved across both entry points.

    Existing sync callers (tests, the engine driver before async fan-out)
    keep their return-only contract; new callers that need concurrency or
    token streaming should call `react_decide_async` directly inside an
    existing event loop.
    """
    return asyncio.run(
        react_decide_async(
            caller=caller,
            cognitive_tools=cognitive_tools,
            terminal_tools=terminal_tools,
            decision_brief=decision_brief,
            lm=lm,
            intermediate_tools=intermediate_tools,
            max_iters=max_iters,
            on_reject=on_reject,
            trace_sink=trace_sink,
            step_sink=step_sink,
        )
    )


async def react_decide_async(
    *,
    caller: str,
    cognitive_tools: Sequence[Callable[..., str]],
    terminal_tools: Mapping[str, Callable[..., ToolResult]],
    decision_brief: str,
    lm: BaseLM,
    intermediate_tools: Mapping[str, Callable[..., ToolResult]] = _EMPTY_INTERMEDIATES,
    max_iters: int = 10,
    on_reject: RejectCallback | None = None,
    trace_sink: TraceSink | None = None,
    step_sink: StepSink | None = None,
    on_thought_chunk: ThoughtChunkCallback | None = None,
) -> Commit:
    """Run one ReAct decision under a scoped LM and return the committed action.

    Async sibling of `react_decide`. Each iteration calls our finish-free
    `predict.acall(...)` so multiple decisions can share an event loop
    under `asyncio.gather` — the engine driver fans out independent
    within-phase decisions this way. `dspy.context(lm=lm)` is contextvar-based,
    so per-task LM scoping is preserved across the gathered tasks.

    `on_thought_chunk(iter_idx, text)` opts the iteration into token-level
    streaming via `dspy.streamify`. When set, the wrapper subscribes to the
    `next_thought` signature field and forwards each chunk to the callback as
    it arrives from the LM; the concatenated chunks for an iter equal that
    iter's `ReActStep.thought`. When `None`, the iteration uses plain
    `acall` and emits no chunks — the path the sync wrapper takes.
    """
    slot = _Slot()
    renamed_cognitive = [_rename_cognitive(fn) for fn in cognitive_tools]
    wrapped_intermediate = [_wrap_intermediate(name, fn, on_reject) for name, fn in intermediate_tools.items()]
    wrapped_terminals = [_wrap_terminal(name, fn, slot, on_reject) for name, fn in terminal_tools.items()]
    tools = [*renamed_cognitive, *wrapped_intermediate, *wrapped_terminals]

    signature = _build_signature()
    predict, tools_by_name = _build_react_predict(signature, tools)

    streamed_caller = _build_streamed_caller(predict, on_thought_chunk) if on_thought_chunk is not None else None

    trajectory: dict[str, object] = {}
    react_steps: list[ReActStep] = []
    lm_calls: list[LMCallRecord] = []
    react_error: BaseException | None = None
    with dspy.context(lm=lm):
        for idx in range(max_iters):
            # Snapshot the LM-history uuid set BEFORE this iteration's call.
            # Slicing by `len(lm.history)` would lose entries under DSPy's
            # bounded `settings.max_history_size` (oldest entries pop on
            # overflow), and would silently misattribute when a single
            # `BaseLM` is shared across seats. Per-uuid identification is
            # robust to both.
            prev_uuids: set[object] = {entry.get("uuid") for entry in lm.history if isinstance(entry, Mapping)}
            t0 = time.monotonic()
            try:
                if streamed_caller is not None:
                    pred = await streamed_caller(idx, decision_brief, _format_trajectory(trajectory))
                else:
                    pred = await predict.acall(decision_brief=decision_brief, trajectory=_format_trajectory(trajectory))
            except (ValueError, AdapterParseError, BaseExceptionGroup) as err:
                # A truncated or otherwise unparseable LM response raises
                # `AdapterParseError` (which `dspy.streamify`'s task group re-raises
                # wrapped in an `ExceptionGroup`); a malformed call raises
                # `ValueError`. Either ends this loop with no commit, surfaced as
                # the `RuntimeError` below — the caller decides whether that is
                # fatal (a kill/exile vote) or a graceful pass (a day reaction).
                react_error = err
                break
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                observation = tools_by_name[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:
                observation = f"error: execution error in {pred.next_tool_name}: {err!r}"
            trajectory[f"observation_{idx}"] = observation

            tool_args_for_step = pred.next_tool_args if isinstance(pred.next_tool_args, Mapping) else {}
            step = ReActStep(
                iter=idx,
                thought=str(pred.next_thought),
                tool=str(pred.next_tool_name),
                args=tool_args_for_step,
                observation=str(observation),
            )
            react_steps.append(step)
            new_entries = [
                entry for entry in lm.history if isinstance(entry, Mapping) and entry.get("uuid") not in prev_uuids
            ]
            iter_lm_calls = tuple(_lm_calls_for_iteration(new_entries, elapsed_ms))
            lm_calls.extend(iter_lm_calls)

            if step_sink is not None:
                step_sink(step, iter_lm_calls)

            # A valid terminal commit ends the turn — there is no finish tool.
            # Cognitive / intermediate tools leave the slot uncommitted, so the
            # loop continues until a commit or `max_iters`.
            if slot.committed:
                break

    if not slot.committed:
        raise RuntimeError(
            f"agent {caller!r} finished without a committed game action",
        ) from react_error

    if trace_sink is not None:
        trace_sink(tuple(react_steps), tuple(lm_calls))

    assert slot.tool_name is not None
    return Commit(tool=slot.tool_name, value=slot.value)


def _build_streamed_caller(
    predict: dspy.Predict,
    on_thought_chunk: ThoughtChunkCallback,
) -> Callable[[int, str, str], Awaitable[Any]]:
    """Build a per-iter async caller that streams `next_thought` chunks.

    `dspy.streamify` returns an async-generator function; we iterate it once
    per ReAct iteration, forwarding every `StreamResponse` chunk to
    `on_thought_chunk(idx, text)` and returning the final `Prediction`. The
    listener uses `allow_reuse=True` because the predictor is invoked many
    times within one decision loop.
    """
    # `dspy.streamify` is typed as returning `Callable[[Any, Any], Awaitable[Any]]`,
    # but with `async_streaming=True` the returned object is actually a function
    # whose call yields an async iterator. Type-erase to `Any` so pyrefly does
    # not complain about kwargs / iteration shape; the runtime contract is
    # what we test against.
    streamed: object = dspy.streamify(
        predict,
        stream_listeners=[StreamListener(signature_field_name="next_thought", allow_reuse=True)],
        is_async_program=True,
        async_streaming=True,
    )

    async def call(idx: int, decision_brief: str, trajectory: str) -> Any:
        prediction: Any = None
        gen = streamed(decision_brief=decision_brief, trajectory=trajectory)  # type: ignore[operator]
        async for chunk in gen:  # type: ignore[union-attr]
            if isinstance(chunk, StreamResponse):
                if chunk.signature_field_name == "next_thought":
                    on_thought_chunk(idx, chunk.chunk)
            elif isinstance(chunk, dspy.Prediction):
                prediction = chunk
        if prediction is None:
            raise RuntimeError("streamify completed without yielding a Prediction")
        return prediction

    return call
