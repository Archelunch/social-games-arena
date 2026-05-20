"""Real-LLM smoke game (T23).

One 7-player Werewolf game driven by a real OpenRouter LLM through DSPy.
Two locks against accidental cost:

1. File-level `@pytest.mark.smoke` and `addopts = "-m 'not smoke'"` in
   `pyproject.toml` — the default `poetry run pytest` does not collect
   this test. An explicit `-m smoke` is required.
2. Inside the test: skip if `OPENROUTER_API_KEY` is unset (so even
   `-m smoke` is a no-op on a machine without the key).

This is a proof-of-life test — it asserts the loop survives a real LLM
round-trip, not win rate, illegal-move rate, or determinism (real LLMs
do not replay byte-identically without an exact prompt-cache hit).
"""

from __future__ import annotations

import json

import dspy
import pytest

from social_deduction_bench import settings
from social_deduction_bench.agents.decisions import ReActDecisionSource
from social_deduction_bench.engine.events import EventStream
from social_deduction_bench.games.werewolf.config import PRIVATE_EVENT_TYPES
from social_deduction_bench.games.werewolf.events import GAME_OVER
from social_deduction_bench.games.werewolf.loop import run_game
from social_deduction_bench.games.werewolf.roles import Role

pytestmark = pytest.mark.smoke

_DEFAULT_MODEL = "qwen/qwen3.5-9b"

ROSTER: tuple[tuple[str, str], ...] = (
    ("Wolf1", Role.WEREWOLF.value),
    ("Wolf2", Role.WEREWOLF.value),
    ("Seer1", Role.SEER.value),
    ("Doc1", Role.DOCTOR.value),
    ("Vil1", Role.VILLAGER.value),
    ("Vil2", Role.VILLAGER.value),
    ("Vil3", Role.VILLAGER.value),
)


def test_real_llm_smoke_game_reaches_terminal_state() -> None:
    cfg = settings.load()
    if cfg.openrouter_api_key is None:
        pytest.skip(
            "OPENROUTER_API_KEY not set; to run this smoke game export OPENROUTER_API_KEY "
            "and rerun with `poetry run pytest -m smoke`",
        )
    model = cfg.smoke_model or _DEFAULT_MODEL

    lms = {
        name: dspy.LM(
            f"openrouter/{model}",
            api_key=cfg.openrouter_api_key,
            temperature=0.0,
            cache=False,
            max_tokens=512,
        )
        for name, _ in ROSTER
    }
    source = ReActDecisionSource(roster=ROSTER, lms=lms, max_iters=20)

    stream = run_game(ROSTER, seed=42, decisions=source, max_rounds=4)
    events = stream.log.events

    # Liveness: the loop reached a terminal state. `run_game` raises
    # `RuntimeError` on `max_rounds` exhaustion, so a non-empty log
    # ending in `GAME_OVER` means every phase resolved without the
    # adapter raising on a dead-ended ReAct loop.
    assert events, "event stream is empty"
    assert events[-1].type == GAME_OVER, f"expected last event GAME_OVER, got {events[-1].type!r}"
    assert events[-1].payload["winner"] in {"werewolves", "villagers"}

    # Invariant #2 — every event whose type the game declares private
    # MUST carry at least one recipient. The engine's
    # `assert_recipients_present` guard upholds this at log time; this
    # test verifies the contract end-to-end under real-LLM traffic
    # (where a hypothetical engine regression would leak).
    for event in events:
        if event.type in PRIVATE_EVENT_TYPES:
            assert event.recipients, f"private event {event.type!r} at seq {event.seq} carries no recipients"

    # Invariant #5 — JSONL transcript round-trips losslessly.
    lines = list(stream.to_jsonl_lines())
    assert lines, "transcript produced no JSONL lines"
    for line in lines:
        json.loads(line)  # raises on malformed JSON
    rebuilt = EventStream.from_jsonl_lines(lines)
    assert rebuilt.log.events == stream.log.events
