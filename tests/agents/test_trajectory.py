"""Tests for the per-decision trajectory sidecar (T30).

The sidecar is the agent-internal projection that lives next to the
engine's append-only `events.jsonl`. It carries signals that are NOT
part of the engine's referee contract (invariant #1) and would dilute
the event log (invariant #5): ReAct thoughts, tool args, observations,
plus LM-call telemetry (model, tokens, latency, cost). The sidecar's
JSONL round-trip must be lossless for every field that is deterministic
(everything except wall-clock latency / provider-cost), and the format
must mirror the event log's header-then-rows shape so the T31 replay
viewer can read both files with the same loader pattern.

Per CLAUDE.md rule 8: each test pins WHY the behavior matters — a
test that can't fail when the sidecar format silently drifts is wrong.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from social_deduction_bench.agents.trajectory import (
    LMCallRecord,
    ReActStep,
    Trajectory,
    TrajectoryStream,
    read_jsonl,
    write_jsonl,
)
from social_deduction_bench.engine import StreamHeader

PLAYERS = (
    ("Wolf1", "werewolf"),
    ("Wolf2", "werewolf"),
    ("Seer1", "seer"),
    ("Doc1", "doctor"),
    ("Vil1", "villager"),
    ("Vil2", "villager"),
    ("Vil3", "villager"),
)


def _step(
    *,
    iter: int = 0,
    thought: str = "consider voting",
    tool: str = "submit_kill_vote",
    args: dict[str, object] | None = None,
    observation: str = "ok: submit_kill_vote committed with {'target': 'Vil1'}",
) -> ReActStep:
    return ReActStep(
        iter=iter,
        thought=thought,
        tool=tool,
        args=args if args is not None else {"target": "Vil1"},
        observation=observation,
    )


def _lm_call(
    *,
    model: str = "dummy",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: float = 1.5,
    cost_usd: float | None = None,
) -> LMCallRecord:
    return LMCallRecord(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def _trajectory(
    *,
    decision_seq: int = 0,
    round: int = 1,
    phase: str = "night",
    caller: str = "Wolf1",
    role: str = "werewolf",
    terminal_tool: str = "submit_kill_vote",
    committed_value: object = "Vil1",
    react_trajectory: tuple[ReActStep, ...] | None = None,
    lm_calls: tuple[LMCallRecord, ...] | None = None,
) -> Trajectory:
    return Trajectory(
        decision_seq=decision_seq,
        round=round,
        phase=phase,
        caller=caller,
        role=role,
        terminal_tool=terminal_tool,
        committed_value=committed_value,
        react_trajectory=react_trajectory if react_trajectory is not None else (_step(),),
        lm_calls=lm_calls if lm_calls is not None else (_lm_call(),),
    )


def _sample_stream() -> TrajectoryStream:
    """A small multi-trajectory stream re-used across round-trip tests."""
    return TrajectoryStream(
        header=StreamHeader(seed=42, game_id="g-t30", players=PLAYERS),
        trajectories=(
            _trajectory(decision_seq=0),
            _trajectory(
                decision_seq=1,
                phase="day",
                caller="Vil1",
                role="villager",
                terminal_tool="submit_exile_vote",
                committed_value="Wolf1",
                react_trajectory=(
                    _step(iter=0, thought="bid first", tool="submit_bid", args={"amount": 5}),
                    _step(
                        iter=1,
                        thought="vote exile",
                        tool="submit_exile_vote",
                        args={"target": "Wolf1"},
                        observation="ok: submit_exile_vote committed with {'target': 'Wolf1'}",
                    ),
                ),
                lm_calls=(_lm_call(latency_ms=2.0), _lm_call(latency_ms=3.0)),
            ),
        ),
    )


# --- Value-type schema ----------------------------------------------------


def test_react_step_is_frozen() -> None:
    """A `ReActStep` cannot be mutated after construction.

    Each step is recorded history; a mutable step could be edited after the
    fact, silently corrupting the sidecar that audits agent reasoning.
    """
    step = _step()
    with pytest.raises(FrozenInstanceError):
        step.iter = 99  # type: ignore[misc]


def test_lm_call_record_is_frozen() -> None:
    """`LMCallRecord` is immutable for the same auditability reason."""
    record = _lm_call()
    with pytest.raises(FrozenInstanceError):
        record.model = "other"  # type: ignore[misc]


def test_trajectory_is_frozen() -> None:
    """`Trajectory` is immutable so a recorded decision cannot be retconned."""
    traj = _trajectory()
    with pytest.raises(FrozenInstanceError):
        traj.decision_seq = 99  # type: ignore[misc]


def test_trajectory_stream_is_frozen() -> None:
    """`TrajectoryStream` is immutable; trajectories tuple is set at construction."""
    stream = _sample_stream()
    with pytest.raises(FrozenInstanceError):
        stream.trajectories = ()  # type: ignore[misc]


def test_value_types_are_structurally_equatable() -> None:
    """Two trajectories built from equal inputs compare equal.

    Pins the test pattern the determinism-style tests rely on: equality is
    field-by-field, so an extra field added without updating tests would
    break callers comparing trajectories.
    """
    assert _trajectory() == _trajectory()
    assert _step() == _step()
    assert _lm_call() == _lm_call()


# --- to_json_dict / from_json_dict round-trip -----------------------------


def test_react_step_json_round_trip() -> None:
    """A `ReActStep` survives `to_json_dict -> from_json_dict` losslessly.

    Single-step round-trips have to be lossless before composing them in
    a stream: a dropped field here corrupts every trajectory downstream.
    """
    original = _step(args={"target": "Wolf1", "force": True, "extra": None})
    restored = ReActStep.from_json_dict(original.to_json_dict())

    assert restored == original


def test_lm_call_record_json_round_trip() -> None:
    """An `LMCallRecord` survives the dict round-trip, including `cost_usd = None`.

    Cache hits and the `DummyLM` both surface `cost_usd is None`. A
    round-trip that coerced `None` -> 0.0 would falsely report cost on
    every cache-hit row.
    """
    original = _lm_call(model="qwen/qwen3.5-9b", prompt_tokens=120, completion_tokens=40, cost_usd=None)
    restored = LMCallRecord.from_json_dict(original.to_json_dict())

    assert restored == original

    original2 = _lm_call(cost_usd=0.000123)
    restored2 = LMCallRecord.from_json_dict(original2.to_json_dict())

    assert restored2 == original2


def test_trajectory_json_round_trip_preserves_every_field() -> None:
    """A `Trajectory` with mixed primitive + nested children survives the round-trip.

    The visualizer (T31) reads `react_trajectory[i].thought` and
    `lm_calls[i].cost_usd` directly off the parsed record; a silent drop
    of either would degrade the replay view.
    """
    original = _trajectory(
        react_trajectory=(
            _step(iter=0, args={"message": "kill seer"}),
            _step(iter=1, tool="submit_kill_vote", args={"target": "Vil1"}),
        ),
        lm_calls=(_lm_call(latency_ms=1.0), _lm_call(latency_ms=2.0, cost_usd=0.0001)),
    )

    restored = Trajectory.from_json_dict(original.to_json_dict())

    assert restored == original


# --- JSONL stream serialization -------------------------------------------


def test_jsonl_first_line_is_the_header() -> None:
    """Line 1 of the JSONL is the header; loading it alone yields a header-only stream.

    Mirrors the event-stream contract: one header line, then one trajectory
    per line, so external tooling can rely on the file shape.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    header_only = TrajectoryStream.from_jsonl_lines([lines[0]])

    assert header_only.header == _sample_stream().header
    assert header_only.trajectories == ()


def test_jsonl_one_trajectory_per_line() -> None:
    """A stream of N trajectories serializes to exactly N+1 lines."""
    stream = _sample_stream()
    lines = list(stream.to_jsonl_lines())

    assert len(lines) == len(stream.trajectories) + 1


def test_jsonl_every_line_is_valid_json() -> None:
    """Each serialized line parses as JSON.

    A malformed line breaks the line-oriented streaming reader and is
    invisible until parse time; verify at write time.
    """
    for line in _sample_stream().to_jsonl_lines():
        json.loads(line)


def test_jsonl_round_trip_preserves_every_trajectory_field() -> None:
    """Full sidecar round-trip preserves every field on every nested record.

    The agent's audit trail is only useful if it round-trips losslessly:
    a dropped `thought` field would make replays inscrutable to T31.
    """
    original = _sample_stream()
    restored = TrajectoryStream.from_jsonl_lines(original.to_jsonl_lines())

    assert restored.header == original.header
    assert restored.trajectories == original.trajectories


def test_jsonl_round_trip_handles_empty_trajectories() -> None:
    """A stream with a header and zero trajectories round-trips losslessly.

    A scripted-decisions game produces zero trajectories. The sidecar must
    still be a valid file (header only) — never "empty file" or "no header."
    """
    stream = TrajectoryStream(header=StreamHeader(seed=7, game_id="g", players=PLAYERS), trajectories=())
    restored = TrajectoryStream.from_jsonl_lines(stream.to_jsonl_lines())

    assert restored.header == stream.header
    assert restored.trajectories == ()


def test_jsonl_serialization_is_byte_identical_for_equal_streams() -> None:
    """Two equal streams serialize to identical text with canonically sorted keys.

    Pins `sort_keys=True` — without it, two equal-content streams could
    differ on dict-insertion order and confound test comparisons.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    assert lines == list(_sample_stream().to_jsonl_lines())

    for line in lines:
        assert json.dumps(json.loads(line), sort_keys=True) == line


# --- write_jsonl / read_jsonl --------------------------------------------


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    """`write_jsonl` / `read_jsonl` round-trip a stream through the filesystem.

    The on-disk file is what T31 actually consumes — the in-memory
    round-trip is necessary but not sufficient.
    """
    path = tmp_path / "game.trajectories.jsonl"
    stream = _sample_stream()

    write_jsonl(stream, path)
    restored = read_jsonl(path)

    assert restored == stream


# --- read-back validation (fail loud) ------------------------------------


def test_from_jsonl_lines_rejects_empty_input() -> None:
    """An empty JSONL input has no header and must be rejected.

    Silently returning an empty stream would mask a truncated or missing
    file (CLAUDE.md rule 11: fail loud).
    """
    with pytest.raises(ValueError, match="header"):
        TrajectoryStream.from_jsonl_lines([])


def test_from_jsonl_lines_rejects_malformed_json() -> None:
    """A line that is not valid JSON must be rejected with a line-number message."""
    lines = list(_sample_stream().to_jsonl_lines())
    corrupted = [lines[0], "{not json"]

    with pytest.raises(ValueError, match="line 2"):
        TrajectoryStream.from_jsonl_lines(corrupted)


def test_from_jsonl_lines_rejects_non_contiguous_decision_seq() -> None:
    """JSONL whose trajectory `decision_seq` values have a gap is rejected.

    A gap means a trajectory was lost between write and read. A corrupt
    sidecar must fail loud, never load silently — mirrors the event log's
    non-contiguous-seq guard so the two files share their integrity story.
    """
    # Build a 3-trajectory stream so dropping the middle row leaves a real gap.
    stream = TrajectoryStream(
        header=StreamHeader(seed=1, game_id="g", players=PLAYERS),
        trajectories=(_trajectory(decision_seq=0), _trajectory(decision_seq=1), _trajectory(decision_seq=2)),
    )
    full_lines = list(stream.to_jsonl_lines())
    gapped = [full_lines[0], full_lines[1], full_lines[3]]  # drop seq=1

    with pytest.raises(ValueError, match="decision_seq"):
        TrajectoryStream.from_jsonl_lines(gapped)


def test_from_jsonl_lines_rejects_out_of_order_decision_seq() -> None:
    """JSONL with reversed trajectory order is rejected.

    The sidecar's append-only ordering reflects the order ReAct loops
    committed; a reordered file misrepresents that history.
    """
    stream = TrajectoryStream(
        header=StreamHeader(seed=1, game_id="g", players=PLAYERS),
        trajectories=(_trajectory(decision_seq=0), _trajectory(decision_seq=1), _trajectory(decision_seq=2)),
    )
    lines = list(stream.to_jsonl_lines())
    reordered = [lines[0], *reversed(lines[1:])]

    with pytest.raises(ValueError, match="decision_seq"):
        TrajectoryStream.from_jsonl_lines(reordered)


# --- Args / committed_value sanitization ---------------------------------


def test_react_step_sanitizes_non_primitive_args() -> None:
    """A `ReActStep` built with a non-JSON-primitive arg coerces to a stringified form.

    `pred.next_tool_args` is LLM-derived: it can contain shapes the JSONL
    round-trip rejects (sets, custom objects). Sanitizing at construction
    means the audit trail survives, mirroring the T29 `_sanitize_arg` pin
    in `decisions.py`.
    """
    step = ReActStep(
        iter=0,
        thought="t",
        tool="submit_bid",
        args={"amount": {1, 2, 3}},
        observation="ok",
    )

    sanitized = step.args["amount"]
    assert isinstance(sanitized, str)
    # Stringification preserves the original repr so the audit trail still
    # carries the LLM's intent — we lose type but not content.
    assert "1" in sanitized
    assert "2" in sanitized


def test_trajectory_sanitizes_committed_value() -> None:
    """A `Trajectory` with a non-primitive `committed_value` is coerced at construction.

    Should never happen for the registered terminals (`submit_*` return
    primitives), but the sanitizer is a belt-and-braces guard so a future
    terminal returning a non-primitive cannot crash the sidecar writer.
    """
    traj = _trajectory(committed_value={"a", "b"})
    restored = Trajectory.from_json_dict(traj.to_json_dict())

    # Sanitized to a str — survives the JSONL round-trip.
    assert isinstance(traj.committed_value, str)
    assert restored == traj


# --- Determinism caveat documented as a test -----------------------------


def test_latency_ms_is_a_non_negative_float() -> None:
    """`LMCallRecord.latency_ms` is a float >= 0.

    The sidecar excludes latency from determinism guarantees by design
    (wall-clock dependent). Pin only the type and sign so a regression
    that flipped it to an `int` or negative would fail loud.
    """
    record = _lm_call(latency_ms=0.0)
    assert isinstance(record.latency_ms, float)
    assert record.latency_ms >= 0.0
