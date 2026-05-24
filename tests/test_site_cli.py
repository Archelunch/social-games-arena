"""Tests for the `sdb-site` CLI (T28).

The CLI globs a run directory, builds the deterministic payload, and writes the static
site. It fails loud (exit 2, nothing written) when no game dir is usable, so an empty
or all-in-progress run never produces a misleading empty site.
"""

from __future__ import annotations

from pathlib import Path

from social_deduction_bench.site_cli import main


def test_cli_zero_valid_game_dirs_exits_2(tmp_path: Path) -> None:
    run = tmp_path / "empty"
    run.mkdir()
    out = tmp_path / "site"
    assert main(["--run", str(run), "--out", str(out)]) == 2
    assert not out.exists()  # no artifacts on failure


def test_cli_builds_site_from_run_dir(tmp_path: Path, two_game_run: Path) -> None:
    out = tmp_path / "site"
    assert main(["--run", str(two_game_run), "--out", str(out)]) == 0
    assert (out / "data.json").exists()
    assert (out / "index.html").exists()
    assert (out / "styles.css").exists()
    assert (out / "app.js").exists()
    # Per-game replay files are written alongside, one per completed game.
    replays = sorted(p.name for p in (out / "games").glob("*.json"))
    assert replays == ["g0000-A-vs-B.json", "g0001-B-vs-A.json"]
