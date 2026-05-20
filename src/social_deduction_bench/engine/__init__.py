"""Game-agnostic referee: state, phases, event log, RNG."""

from social_deduction_bench.engine.determinism import assert_deterministic, assert_streams_identical
from social_deduction_bench.engine.events import (
    Event,
    EventLog,
    EventStream,
    StreamHeader,
    assert_recipients_present,
    read_jsonl,
    write_jsonl,
)
from social_deduction_bench.engine.observation import observations_for, public_events
from social_deduction_bench.engine.phase import TerminalCheck, advance_phase, is_terminal
from social_deduction_bench.engine.rng import GameRNG
from social_deduction_bench.engine.state import GameState, Phase, PlayerState
from social_deduction_bench.engine.validation import (
    ToolCall,
    ToolRequirement,
    ValidationResult,
    available_tools,
    validate_tool_call,
)

__all__ = [
    "Event",
    "EventLog",
    "EventStream",
    "GameRNG",
    "GameState",
    "Phase",
    "PlayerState",
    "StreamHeader",
    "TerminalCheck",
    "ToolCall",
    "ToolRequirement",
    "ValidationResult",
    "advance_phase",
    "assert_deterministic",
    "assert_recipients_present",
    "assert_streams_identical",
    "available_tools",
    "is_terminal",
    "observations_for",
    "public_events",
    "read_jsonl",
    "validate_tool_call",
    "write_jsonl",
]
