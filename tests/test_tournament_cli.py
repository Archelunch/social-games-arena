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
import shutil
from pathlib import Path

import pytest

from social_deduction_bench.cli import _DEFAULT_NAMES
from social_deduction_bench.games.werewolf.tournament import run_tournament, schedule_tournament
from social_deduction_bench.rating.manifest import read_json as read_manifest
from social_deduction_bench.tournament_cli import _exit_code, main, run_sweep


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


# --- resume from disk ----------------------------------------------------


def _created_ats(output_dir: Path) -> dict[str, str]:
    """Map each game-id dir to its manifest `created_at` (a re-run rewrites it)."""
    return {p.parent.name: read_manifest(p).created_at for p in output_dir.glob("*/manifest.json")}


def test_resume_reaggregates_without_rerunning_completed_games(tmp_path: Path) -> None:
    # A crash that lost only the in-memory summary must be recoverable WITHOUT
    # re-running (re-paying for) completed games. Resuming a fully-complete run
    # re-runs nothing — every game's manifest timestamp is unchanged — but rebuilds
    # summary.json.
    out = tmp_path / "run"
    kwargs = {"models": ["A", "B"], "games_per_pair": 2, "seed": 1, "names": _DEFAULT_NAMES, "dry_run": True}
    run_sweep(output_dir=out, api_key=None, **kwargs)
    before = _created_ats(out)
    (out / "summary.json").unlink()

    run_sweep(output_dir=out, api_key=None, resume=True, **kwargs)
    after = _created_ats(out)
    assert after == before  # nothing re-run
    assert (out / "summary.json").exists()


def test_resume_reruns_only_missing_games(tmp_path: Path) -> None:
    # A sweep that died partway resumes by running ONLY the games whose dirs are
    # missing, reusing the rest, then re-aggregating the full set.
    out = tmp_path / "run"
    kwargs = {"models": ["A", "B"], "games_per_pair": 2, "seed": 1, "names": _DEFAULT_NAMES, "dry_run": True}
    run_sweep(output_dir=out, api_key=None, **kwargs)
    before = _created_ats(out)

    victim = sorted(before)[0]
    shutil.rmtree(out / victim)

    result = run_sweep(output_dir=out, api_key=None, resume=True, **kwargs)
    after = _created_ats(out)
    assert set(after) == set(before)  # the missing game was re-created
    assert after[victim] != before[victim]  # it was actually re-run
    for game_id in before:
        if game_id != victim:
            assert after[game_id] == before[game_id]  # the rest were reused, not re-run
    assert len(result.games) == len(before)  # the full set is aggregated


def test_resume_treats_dir_without_manifest_as_incomplete(tmp_path: Path) -> None:
    # A torn write (crash after events.jsonl but before manifest.json, which is
    # written last) must NOT be treated as a completed game — loading it would lose
    # the seat->model identity and mis-rate the game. Resume must re-run it.
    out = tmp_path / "run"
    kwargs = {"models": ["A", "B"], "games_per_pair": 2, "seed": 1, "names": _DEFAULT_NAMES, "dry_run": True}
    run_sweep(output_dir=out, api_key=None, **kwargs)
    before = _created_ats(out)

    victim = sorted(before)[0]
    (out / victim / "manifest.json").unlink()  # simulate the torn write
    assert (out / victim / "events.jsonl").exists()  # events survived; manifest did not

    run_sweep(output_dir=out, api_key=None, resume=True, **kwargs)
    after = _created_ats(out)
    assert victim in after  # manifest re-created -> the torn dir was re-run
    assert after[victim] != before[victim]  # fresh run, new timestamp
    for game_id in before:
        if game_id != victim:
            assert after[game_id] == before[game_id]  # whole dirs reused, not re-run


# --- exit code: a totally-failed sweep is not a success ------------------


def test_exit_code_is_nonzero_when_every_scheduled_game_failed() -> None:
    matchups = schedule_tournament(["A", "B"], games_per_pair=1, seed=1, names=_DEFAULT_NAMES)

    def always_fail(_m: object) -> object:
        raise RuntimeError("boom")

    result = run_tournament(matchups, always_fail, skip_failures=True)  # type: ignore[arg-type]
    assert result.games == ()
    assert _exit_code(result, len(matchups)) == 1


def test_exit_code_is_zero_when_some_games_completed(tmp_path: Path) -> None:
    result = run_sweep(
        models=["A", "B"],
        games_per_pair=1,
        seed=1,
        names=_DEFAULT_NAMES,
        output_dir=tmp_path,
        api_key=None,
        dry_run=True,
    )
    assert _exit_code(result, len(result.matchups)) == 0


def test_exit_code_is_zero_for_an_empty_schedule() -> None:
    result = run_tournament((), lambda _m: None)  # type: ignore[arg-type,return-value]
    assert _exit_code(result, 0) == 0
