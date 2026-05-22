"""Real-LLM concurrent tournament smoke (T27).

A tiny 2-model, 1-game-per-pair sweep driven by a real OpenRouter LLM through the
production runner, exercising the thread-pool concurrency path against a live API.
Two locks against accidental cost (mirroring the T23 smoke game):

1. File-level `@pytest.mark.smoke` + `addopts = "-m 'not smoke'"` — the default
   `poetry run pytest` does not collect this test.
2. Inside the test: skip if `OPENROUTER_API_KEY` is unset.

Proof-of-life only: it asserts the sweep completes, persists a summary, and rates
the cross pair — not win rate or determinism (real LLMs do not replay
byte-identically).
"""

from __future__ import annotations

import json

import pytest

from social_deduction_bench import settings
from social_deduction_bench.cli import _DEFAULT_NAMES
from social_deduction_bench.tournament_cli import run_sweep

pytestmark = pytest.mark.smoke

_MODEL = "qwen/qwen3.5-9b"


def test_real_llm_tournament_smoke_completes_and_persists(tmp_path) -> None:
    cfg = settings.load()
    if cfg.openrouter_api_key is None:
        pytest.skip(
            "OPENROUTER_API_KEY not set; to run this smoke sweep export OPENROUTER_API_KEY "
            "and rerun with `poetry run pytest -m smoke`",
        )
    model = cfg.smoke_model or _MODEL

    # Two distinct model labels backed by the same underlying model so the cross
    # pair produces a real cross-model rating (not skipped as self-play). The label
    # is the rated identity; the resolver maps it to the bare model id the runner
    # actually calls (the runner prefixes `openrouter/` itself).
    result = run_sweep(
        models=[f"{model}#x", f"{model}#y"],
        games_per_pair=1,
        seed=42,
        names=_DEFAULT_NAMES,
        output_dir=tmp_path,
        api_key=cfg.openrouter_api_key,
        concurrency=2,
        max_iters=20,
        max_tokens=512,
        temperature=0.0,
        reasoning=False,
        model_resolver=lambda label: label.split("#", 1)[0],
    )

    assert len(result.games) == 3  # (x,x), (x,y), (y,y)
    assert (tmp_path / "summary.json").exists()
    data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert len(data["games"]) == 3
    # The one cross game rates both labels; the two self-play games are skipped.
    assert {row["model"] for row in data["leaderboard"]} == {f"{model}#x", f"{model}#y"}
