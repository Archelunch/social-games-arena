"""Per-decision-point ReAct loop primitive.

Drives `dspy.ReAct.react` (the per-iteration predictor) directly, not
`ReAct.forward` — `forward` would append an extra `extract` LM call that the
caller does not need, since the commitment is captured side-band when a
terminal tool reports a valid `ToolResult`.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import dspy
from dspy.clients.base_lm import BaseLM

from social_deduction_bench.games.werewolf.tools import ToolResult


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


def _wrap_terminal(name: str, fn: Callable[..., ToolResult], slot: _Slot) -> Callable[..., str]:
    """Wrap a terminal tool to capture a valid commit into `slot` and return a string observation."""

    @functools.wraps(fn)
    def wrapper(**kwargs: object) -> str:
        result = fn(**kwargs)
        if result.valid:
            slot.committed = True
            slot.tool_name = name
            slot.value = result.value
            return f"ok: {name} committed with {kwargs}"
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


def react_decide(
    *,
    caller: str,
    cognitive_tools: Sequence[Callable[..., str]],
    terminal_tools: Mapping[str, Callable[..., ToolResult]],
    decision_brief: str,
    lm: BaseLM,
    max_iters: int = 10,
) -> Commit:
    """Run one ReAct decision under a scoped LM and return the committed action.

    Each iteration is one LM call. Terminal-tool wrappers write into a private
    slot on a valid call; the LLM then emits `finish` to close the loop. An
    empty slot at loop exit raises `RuntimeError`.
    """
    slot = _Slot()
    renamed_cognitive = [_rename_cognitive(fn) for fn in cognitive_tools]
    wrapped_terminals = [_wrap_terminal(name, fn, slot) for name, fn in terminal_tools.items()]
    tools = [*renamed_cognitive, *wrapped_terminals]

    signature = _build_signature()
    react = dspy.ReAct(signature, tools=tools, max_iters=max_iters)

    trajectory: dict[str, object] = {}
    react_error: ValueError | None = None
    with dspy.context(lm=lm):
        for idx in range(max_iters):
            try:
                pred = react.react(decision_brief=decision_brief, trajectory=_format_trajectory(trajectory))
            except ValueError as err:
                react_error = err
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                observation = react.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:
                observation = f"Execution error in {pred.next_tool_name}: {err!r}"
            trajectory[f"observation_{idx}"] = observation

            if pred.next_tool_name == "finish":
                break

    if not slot.committed:
        raise RuntimeError(
            f"agent {caller!r} finished without a committed game action",
        ) from react_error

    assert slot.tool_name is not None
    return Commit(tool=slot.tool_name, value=slot.value)
