"""Tests for the `sdb-tournament` CLI.

The tournament CLI schedules a seeded pairwise cross-play sweep, runs every game,
and writes each game's artifacts plus a tournament-level `summary.json`. Tests
exercise the `--dry-run` path (scripted games, no `OPENROUTER_API_KEY`, no API
calls) to verify the full pipeline — schedule → run (concurrently) → persist →
summarize — and the absent-key guard for the real-LLM path.

Per CLAUDE.md "Safety & Permissions": a paid sweep must never start by accident, so
the real path fails loud without a key rather than silently calling the API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from social_deduction_bench.tournament_cli import main


def _invoke(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    """Run `main(argv)` with no API key set and capture the exit code."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        return main(argv) or 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1


def _game_dirs(output_dir: Path) -> list[Path]:
    return sorted(p.parent for p in output_dir.glob("*/events.jsonl"))


def test_dry_run_writes_per_game_artifacts_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Full plumbing with no key: 2 models incl. self-pairs -> (A,A),(A,B),(B,B), 2
    # games each = 6 scripted games, run concurrently, each persisted, plus summary.json.
    code = _invoke(
        [
            "--dry-run",
            "--models",
            "A,B",
            "--games-per-pair",
            "2",
            "--seed",
            "1",
            "--concurrency",
            "4",
            "--output-dir",
            str(tmp_path),
        ],
        monkeypatch,
    )
    assert code == 0

    game_dirs = _game_dirs(tmp_path)
    assert len(game_dirs) == 6
    for d in game_dirs:
        assert (d / "events.jsonl").exists()
        assert (d / "manifest.json").exists()

    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    # The two cross (A-vs-B) games rate; the four self-play games are skipped.
    assert {row["model"] for row in data["leaderboard"]} == {"A", "B"}
    assert data["n_games"] == 2
    assert data["n_skipped"] == 4
    assert len(data["games"]) == 6
    # The dry-run script is a deterministic werewolves-win; pin it so the test would
    # fail if a game mis-resolved (e.g. a broken scripted source ending villager-win).
    assert all(row["winner"] == "werewolves" for row in data["games"])


def test_dry_run_summary_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Scripted games + seeded schedule + a timestamp-free summary → byte-identical
    # summaries across two runs (invariant #4 spirit at the tournament level).
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    args = ["--dry-run", "--models", "A,B", "--games-per-pair", "2", "--seed", "1", "--output-dir"]
    assert _invoke([*args, str(out_a)], monkeypatch) == 0
    assert _invoke([*args, str(out_b)], monkeypatch) == 0
    assert (out_a / "summary.json").read_text(encoding="utf-8") == (out_b / "summary.json").read_text(encoding="utf-8")


def test_real_run_without_api_key_fails_loud_and_spends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real mode (no --dry-run) with no key must exit non-zero before any game runs:
    # no summary, no game directories — i.e. no API spend.
    code = _invoke(
        ["--models", "A,B", "--games-per-pair", "2", "--seed", "1", "--output-dir", str(tmp_path)],
        monkeypatch,
    )
    assert code == 2
    assert not (tmp_path / "summary.json").exists()
    assert _game_dirs(tmp_path) == []
