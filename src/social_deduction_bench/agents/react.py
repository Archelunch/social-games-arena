"""Per-decision-point ReAct loop primitive.

Drives `dspy.ReAct.react` (the per-iteration predictor) directly, not
`ReAct.forward` — `forward` would append an extra `extract` LM call that the
caller does not need, since the commitment is captured side-band when a
terminal tool reports a valid `ToolResult`.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import dspy
from dspy.clients.base_lm import BaseLM

from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep
from social_deduction_bench.games.werewolf.tools import ToolResult

logger = logging.getLogger(__name__)

RejectCallback = Callable[[str, dict[str, object], str], None]
"""Signature for `react_decide`'s `on_reject` hook: `(tool, args, reason) -> None`."""

TraceSink = Callable[[tuple[ReActStep, ...], tuple[LMCallRecord, ...]], None]
"""Signature for `react_decide`'s `trace_sink`: receives the per-iteration steps
and the per-LM-call telemetry of a completed loop. Fires once after a
successful commit; not called when the loop raises `RuntimeError`."""

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
        """Decide and commit one game action by calling the appropriate tool."""

        decision_brief: str = dspy.InputField()
        committed_action: str = dspy.OutputField()

    return DecisionSignature


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
) -> Commit:
    """Run one ReAct decision under a scoped LM and return the committed action.

    Each iteration is one LM call. Terminal-tool wrappers write into a private
    slot on a valid call; the LLM then emits `finish` to close the loop. An
    empty slot at loop exit raises `RuntimeError`.

    `intermediate_tools` are game-action tools that emit a side-effect but do
    not terminate the loop — used for `werewolf_chat` so the werewolves can
    speak (and emit `WEREWOLF_CHAT` events) before committing a kill vote.
    `on_reject(tool, args, reason)` fires once per `ToolResult(valid=False)`
    from either category; the adapter uses it to emit `TOOL_REJECTED` events.

    `trace_sink(steps, lm_calls)` fires once after a successful commit (T30
    sidecar). On `RuntimeError` (no commit) it is not called — a failed loop
    produces no `Trajectory`. Defaults to `None` so legacy callers keep their
    return-only contract.
    """
    slot = _Slot()
    renamed_cognitive = [_rename_cognitive(fn) for fn in cognitive_tools]
    wrapped_intermediate = [_wrap_intermediate(name, fn, on_reject) for name, fn in intermediate_tools.items()]
    wrapped_terminals = [_wrap_terminal(name, fn, slot, on_reject) for name, fn in terminal_tools.items()]
    tools = [*renamed_cognitive, *wrapped_intermediate, *wrapped_terminals]

    signature = _build_signature()
    react = dspy.ReAct(signature, tools=tools, max_iters=max_iters)

    trajectory: dict[str, object] = {}
    react_steps: list[ReActStep] = []
    lm_calls: list[LMCallRecord] = []
    react_error: ValueError | None = None
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
                pred = react.react(decision_brief=decision_brief, trajectory=_format_trajectory(trajectory))
            except ValueError as err:
                react_error = err
                break
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                observation = react.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:
                observation = f"Execution error in {pred.next_tool_name}: {err!r}"
            trajectory[f"observation_{idx}"] = observation

            tool_args_for_step = pred.next_tool_args if isinstance(pred.next_tool_args, Mapping) else {}
            react_steps.append(
                ReActStep(
                    iter=idx,
                    thought=str(pred.next_thought),
                    tool=str(pred.next_tool_name),
                    args=tool_args_for_step,
                    observation=str(observation),
                )
            )
            new_entries = [
                entry for entry in lm.history if isinstance(entry, Mapping) and entry.get("uuid") not in prev_uuids
            ]
            lm_calls.extend(_lm_calls_for_iteration(new_entries, elapsed_ms))

            if pred.next_tool_name == "finish":
                break

    if not slot.committed:
        raise RuntimeError(
            f"agent {caller!r} finished without a committed game action",
        ) from react_error

    if trace_sink is not None:
        trace_sink(tuple(react_steps), tuple(lm_calls))

    assert slot.tool_name is not None
    return Commit(tool=slot.tool_name, value=slot.value)
