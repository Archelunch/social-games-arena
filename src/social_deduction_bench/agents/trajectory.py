"""Per-decision trajectory sidecar (T30, WEREWOLF_DESIGN.md §11).

The sidecar is the agent-internal projection that lives next to the engine's
append-only `events.jsonl`. Each line is one decision-point ReAct loop with
the agent's thoughts, tool args, observations, and per-LM-call telemetry
(model, tokens, latency, cost). Joined back to the event log at viewing
time by `(round, phase, caller)` plus a strictly increasing `decision_seq`.

Why a sidecar and not more event types? Agent-internal signals (ReAct
thoughts, LM-call cost, latency) are not part of the engine's referee
contract (invariant #1) and would dilute the append-only event log
(invariant #5). They live in a parallel JSONL file.

Format mirrors `engine/events.py` so the two files share their integrity
story: one header line (seed + game_id + roster), then one trajectory per
line, `sort_keys=True` for byte-identical serialization of equal streams.
The reader is fail-loud on every malformed line.

Wall-clock and provider-cost fields (`latency_ms`, `cost_usd`) are
explicitly *not* part of the determinism contract — they vary run-to-run
even with the same seed. The engine's determinism harness compares
`EventStream`s, never `TrajectoryStream`s.
"""

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from social_deduction_bench.engine import StreamHeader


def sanitize_arg(value: object) -> object:
    """Coerce a single value to a JSON primitive the sidecar's JSONL round-trip accepts.

    `pred.next_tool_args` is LLM-derived: it can contain shapes JSONL
    rejects (sets, nested dicts with non-string keys, custom objects).
    Sanitization preserves audit-trail content as a `repr` string when the
    raw value is not a JSON primitive — losing the type but not the
    intent. Mirrors the existing T29 helper in `decisions.py`, hoisted
    here so both consumers (TOOL_REJECTED event drafts in the adapter and
    `ReActStep.args` / `Trajectory.committed_value` in the sidecar) share
    one definition.
    """
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return repr(value)


def _sanitize_args_mapping(args: Mapping[str, object]) -> dict[str, object]:
    """Apply `sanitize_arg` to every value of an LLM-derived args mapping."""
    return {key: sanitize_arg(value) for key, value in args.items()}


@dataclass(frozen=True, slots=True)
class ReActStep:
    """One iteration of a ReAct loop: thought, tool call, and observation.

    `iter` is the 0-based iteration index inside the owning `Trajectory`.
    `args` is the LLM's tool-call arguments, sanitized to JSON primitives at
    construction so the sidecar's JSONL round-trip never crashes on a
    non-primitive value the model emitted.
    """

    iter: int
    thought: str
    tool: str
    args: Mapping[str, object]
    observation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", MappingProxyType(_sanitize_args_mapping(self.args)))

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-ready dict."""
        return {
            "iter": self.iter,
            "thought": self.thought,
            "tool": self.tool,
            "args": dict(self.args),
            "observation": self.observation,
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, object]) -> "ReActStep":
        """Rebuild a `ReActStep` from its JSON dict (fail-loud on missing fields)."""
        return cls(
            iter=raw["iter"],  # type: ignore[arg-type]
            thought=raw["thought"],  # type: ignore[arg-type]
            tool=raw["tool"],  # type: ignore[arg-type]
            args=raw["args"],  # type: ignore[arg-type]
            observation=raw["observation"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class LMCallRecord:
    """One LM call's telemetry: model, token counts, latency, cost.

    `cost_usd is None` means the call was a cache hit (DSPy convention) or a
    non-billable LM (e.g. `DummyLM` in tests). The visualizer can render
    accordingly; the sidecar does not collapse these cases into a separate
    `cached` flag — the absence of cost already encodes "no billing event."

    `latency_ms` is wall-clock-dependent and varies run-to-run. The sidecar's
    determinism contract excludes it (and `cost_usd`) by design.
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float | None

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-ready dict; `cost_usd is None` survives the round-trip."""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, object]) -> "LMCallRecord":
        """Rebuild an `LMCallRecord` from its JSON dict."""
        return cls(
            model=raw["model"],  # type: ignore[arg-type]
            prompt_tokens=raw["prompt_tokens"],  # type: ignore[arg-type]
            completion_tokens=raw["completion_tokens"],  # type: ignore[arg-type]
            latency_ms=raw["latency_ms"],  # type: ignore[arg-type]
            cost_usd=raw["cost_usd"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One decision-point ReAct loop's record.

    `decision_seq` is the gap-free, monotonically-increasing index assigned
    by the adapter, the ordering key the visualizer joins on. `phase` is the
    engine's `Phase.value` string at decision time (before any
    `advance_phase`). `committed_value` is the parsed terminal-tool value
    (e.g. a target player name for a kill vote, an `int` for a bid),
    sanitized at construction so a future non-primitive terminal cannot
    crash the sidecar writer.
    """

    decision_seq: int
    round: int
    phase: str
    caller: str
    role: str
    terminal_tool: str
    committed_value: object
    react_trajectory: tuple[ReActStep, ...]
    lm_calls: tuple[LMCallRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "committed_value", sanitize_arg(self.committed_value))
        object.__setattr__(self, "react_trajectory", tuple(self.react_trajectory))
        object.__setattr__(self, "lm_calls", tuple(self.lm_calls))

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-ready dict for the JSONL stream."""
        return {
            "decision_seq": self.decision_seq,
            "round": self.round,
            "phase": self.phase,
            "caller": self.caller,
            "role": self.role,
            "terminal_tool": self.terminal_tool,
            "committed_value": self.committed_value,
            "react_trajectory": [step.to_json_dict() for step in self.react_trajectory],
            "lm_calls": [call.to_json_dict() for call in self.lm_calls],
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, object]) -> "Trajectory":
        """Rebuild a `Trajectory` from its JSON dict, validating shape fail-loud.

        Each nested list must be a list (a bare string would otherwise splay
        per character through `tuple(...)`).
        """
        for key in ("react_trajectory", "lm_calls"):
            if not isinstance(raw[key], list):
                raise ValueError(f"trajectory {key!r} must be a list, got {type(raw[key]).__name__}")
        return cls(
            decision_seq=raw["decision_seq"],  # type: ignore[arg-type]
            round=raw["round"],  # type: ignore[arg-type]
            phase=raw["phase"],  # type: ignore[arg-type]
            caller=raw["caller"],  # type: ignore[arg-type]
            role=raw["role"],  # type: ignore[arg-type]
            terminal_tool=raw["terminal_tool"],  # type: ignore[arg-type]
            committed_value=raw["committed_value"],
            react_trajectory=tuple(ReActStep.from_json_dict(s) for s in raw["react_trajectory"]),  # type: ignore[union-attr]
            lm_calls=tuple(LMCallRecord.from_json_dict(c) for c in raw["lm_calls"]),  # type: ignore[union-attr]
        )


@dataclass(frozen=True, slots=True)
class TrajectoryStream:
    """A complete sidecar: a `StreamHeader` plus a tuple of `Trajectory` rows.

    Reuses the engine's `StreamHeader` (seed, game_id, players) so the
    sidecar is self-contained — the visualizer can load a single file and
    know which game it belongs to. Serializes to JSONL — header on line 1,
    one trajectory per line thereafter — and reads back with full
    validation. `sort_keys=True` makes equal streams byte-identical (the
    same convention `EventStream` uses).
    """

    header: StreamHeader
    trajectories: tuple[Trajectory, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectories", tuple(self.trajectories))

    def to_jsonl_lines(self) -> Iterator[str]:
        """Yield the JSONL text: header line first, then one line per trajectory."""
        yield json.dumps(self.header.to_json_dict(), sort_keys=True)
        for trajectory in self.trajectories:
            yield json.dumps(trajectory.to_json_dict(), sort_keys=True)

    @classmethod
    def from_jsonl_lines(cls, lines: Iterable[str]) -> "TrajectoryStream":
        """Parse JSONL back into a `TrajectoryStream`, fail-loud on corruption.

        Rejects empty input (no header), any line that is not valid JSON,
        and any trajectory whose `decision_seq` is not the contiguous,
        strictly increasing sequence `0, 1, 2, ...`. The same integrity
        story the event log enforces, applied to the sidecar.
        """
        materialized = [line.rstrip("\n") for line in lines]
        if not materialized:
            raise ValueError("empty JSONL input: no header line")

        def _parse(raw_line: str, line_number: int) -> object:
            try:
                return json.loads(raw_line)
            except json.JSONDecodeError as e:
                raise ValueError(f"malformed JSONL: line {line_number} is not valid JSON") from e

        header = StreamHeader.from_json_dict(_parse(materialized[0], 1))  # type: ignore[arg-type]
        trajectories: list[Trajectory] = []
        for offset, raw_line in enumerate(materialized[1:]):
            trajectory = Trajectory.from_json_dict(_parse(raw_line, offset + 2))  # type: ignore[arg-type]
            if trajectory.decision_seq != offset:
                raise ValueError(
                    f"non-contiguous decision_seq: expected {offset}, found {trajectory.decision_seq}",
                )
            trajectories.append(trajectory)
        return cls(header=header, trajectories=tuple(trajectories))


def write_jsonl(stream: TrajectoryStream, path: Path) -> None:
    """Write `stream` to `path` as UTF-8 JSONL, one line per `to_jsonl_lines` entry."""
    with path.open("w", encoding="utf-8") as handle:
        for line in stream.to_jsonl_lines():
            handle.write(line)
            handle.write("\n")


def read_jsonl(path: Path) -> TrajectoryStream:
    """Read a UTF-8 JSONL file and rebuild the `TrajectoryStream`, validating on the way."""
    text = path.read_text(encoding="utf-8")
    return TrajectoryStream.from_jsonl_lines(text.splitlines())
