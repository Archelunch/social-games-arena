"""`sdb-tournament` — run a seeded pairwise cross-play sweep from the terminal.

Entry point: `poetry run sdb-tournament`. Schedules every model pair (including
self-pairs) into `games_per_pair` seeded games, runs them (optionally concurrently),
writes each game's artifacts and a tournament-level `summary.json`, and prints a
short leaderboard.

Two run modes:

- `--dry-run`: scripted werewolves-win games (no API key, no API calls) for CLI
  plumbing tests.
- real: each seat plays its label's model through the production ReAct runner.

Per CLAUDE.md "Safety & Permissions": a paid sweep must never start by accident, so
real mode fails loud (exit 2) with no output written when `OPENROUTER_API_KEY` is
unset, before any game runs.
"""

from __future__ import annotations

import argparse
import multiprocessing
import secrets
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from social_deduction_bench import settings
from social_deduction_bench.agents.trajectory import TrajectoryStream
from social_deduction_bench.cli import (
    _DEFAULT_NAMES,
    _build_react_source,
    _final_position,
    _git_sha,
    _memories_from,
    _trajectories_from,
    _write_outputs,
)
from social_deduction_bench.engine import EventStream
from social_deduction_bench.games.werewolf.day import DayActions
from social_deduction_bench.games.werewolf.events import GAME_OVER
from social_deduction_bench.games.werewolf.loop import run_game
from social_deduction_bench.games.werewolf.metrics import GameMetrics, extract_game_metrics, extract_run_dir
from social_deduction_bench.games.werewolf.night import NightActions
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.games.werewolf.scripted import ScriptedDecisions
from social_deduction_bench.games.werewolf.tournament import (
    Matchup,
    TournamentResult,
    run_tournament,
    schedule_tournament,
    write_tournament_summary,
)
from social_deduction_bench.rating.manifest import RunManifest


def _wolves_win_script(roster: Sequence[tuple[str, str]]) -> ScriptedDecisions:
    """Build a deterministic werewolves-win script for ANY roster.

    The dry-run driver: unlike `cli._build_scripted_source` (which hardcodes
    Alice/Bob as wolves), this reads the actual wolf seats from `roster` — roles are
    dealt from each game's seed, so a fixed-name script would target the wrong seats.
    Wolves kill living villagers one per night (no seer/doctor staged, so kills land)
    until they reach parity; days between nights are all-abstain. The terminal night
    ends the game.
    """
    wolves = [n for n, r in roster if r == Role.WEREWOLF.value]
    living = [n for n, r in roster if r != Role.WEREWOLF.value]
    nights: list[NightActions] = []
    while len(wolves) < len(living):
        victim = living.pop(0)
        nights.append(NightActions(kill_votes={w: victim for w in wolves}))
    days = [DayActions(exile_votes={}) for _ in range(len(nights) - 1)]
    return ScriptedDecisions(nights=nights, days=days)


def _winner_and_rounds(stream: EventStream) -> tuple[str, int]:
    """Read the winning faction (from GAME_OVER) and final round from the event stream."""
    winner: str = next(e for e in stream.log.events if e.type == GAME_OVER).payload["winner"]  # type: ignore[assignment]
    rounds = _final_position(stream)[0]
    return winner, rounds


def run_one_scripted_game(matchup: Matchup, *, output_dir: Path) -> GameMetrics:
    """Run one scripted werewolves-win game for `matchup` and persist its artifacts.

    Used by `--dry-run`: no LMs are seated. The manifest carries `matchup.seat_models`
    so `extract_game_metrics` resolves each seat's real model label (not the `unknown`
    sentinel), giving the leaderboard genuine cross-model identities.
    """
    source = _wolves_win_script(matchup.roster)
    stream = run_game(matchup.roster, matchup.seed, source, game_id=matchup.game_id)
    winner, rounds = _winner_and_rounds(stream)

    manifest = RunManifest(
        game_id=matchup.game_id,
        seed=matchup.seed,
        players=matchup.roster,
        models=matchup.seat_models,
        model_arg="scripted",
        temperature=0.0,
        max_tokens=0,
        max_iters=0,
        reasoning=False,
        git_sha=_git_sha(),
        created_at=datetime.now(UTC).isoformat(),
        winner=winner,
        rounds=rounds,
    )
    _write_outputs(stream, (), {}, manifest, output_dir / matchup.game_id, matchup.game_id)
    return extract_game_metrics(stream, TrajectoryStream(header=stream.header, trajectories=()), manifest)


@dataclass(frozen=True, slots=True)
class _GameJob:
    """A picklable description of one game to run in an isolated child process.

    `resolved_models` is the seat -> bare-model map (the `model_resolver` is applied
    in the parent so no closure crosses the process boundary); `matchup.seat_models`
    keeps the LABELS the manifest records for rating. The job carries everything the
    child needs so nothing unpicklable (a lambda, a callback) is sent across.
    """

    matchup: Matchup
    output_dir: str
    dry_run: bool
    api_key: str | None
    resolved_models: tuple[tuple[str, str], ...]
    max_iters: int
    max_tokens: int
    temperature: float
    reasoning: bool


def _run_game_job(job: _GameJob) -> None:
    """Child-process entry point: run one game and persist its artifacts.

    Returns nothing — the parent reads the result from disk via `extract_run_dir`,
    so no `GameMetrics` is pickled back. Runs in a fresh process so every file
    descriptor it opens (per-phase event loops, litellm's HTTP sockets) is reclaimed
    by the OS when the process exits, which is what keeps a long sweep from leaking
    fds until it hits the per-process cap.
    """
    output_dir = Path(job.output_dir)
    if job.dry_run:
        run_one_scripted_game(job.matchup, output_dir=output_dir)
        return
    if job.api_key is None:
        raise ValueError("a real game job requires api_key")

    source = _build_react_source(
        roster=job.matchup.roster,
        seat_models=dict(job.resolved_models),
        api_key=job.api_key,
        max_iters=job.max_iters,
        max_tokens=job.max_tokens,
        temperature=job.temperature,
        reasoning=job.reasoning,
        on_trajectory=None,
        on_decision_start=None,
        on_step=None,
        on_thought_chunk=None,
    )
    stream = run_game(job.matchup.roster, job.matchup.seed, source, game_id=job.matchup.game_id)
    winner, rounds = _winner_and_rounds(stream)
    trajectories = _trajectories_from(source)
    memories = _memories_from(source)

    manifest = RunManifest(
        game_id=job.matchup.game_id,
        seed=job.matchup.seed,
        players=job.matchup.roster,
        models=job.matchup.seat_models,
        model_arg="tournament",
        temperature=job.temperature,
        max_tokens=job.max_tokens,
        max_iters=job.max_iters,
        reasoning=job.reasoning,
        git_sha=_git_sha(),
        created_at=datetime.now(UTC).isoformat(),
        winner=winner,
        rounds=rounds,
    )
    _write_outputs(stream, trajectories, memories, manifest, output_dir / job.matchup.game_id, job.matchup.game_id)


def run_one_game_isolated(job: _GameJob) -> GameMetrics:
    """Run one game in its own spawned process, then read its persisted result.

    Process isolation is the fd-leak containment: litellm opens HTTP sockets under
    each per-phase `asyncio.run` loop that the OS only reclaims on process exit, so
    an in-process sweep accumulates fds until it hits the per-process cap and every
    later game dies with `OSError: Too many open files`. A fresh child per game keeps
    each game's fds bounded and reclaimed on exit; the parent only holds the child's
    pipe. A non-zero exit code (crash, OOM, kill) surfaces as a failure the tournament
    skips, and — having written no manifest — `--resume` re-runs it.
    """
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_run_game_job, args=(job,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"game {job.matchup.game_id} failed in its worker process (exit code {proc.exitcode})")
    return extract_run_dir(Path(job.output_dir) / job.matchup.game_id)


def run_sweep(
    *,
    models: Sequence[str],
    games_per_pair: int,
    seed: int,
    names: Sequence[str],
    output_dir: Path,
    api_key: str | None,
    concurrency: int = 1,
    max_iters: int = 9,
    max_tokens: int = 8000,
    temperature: float = 1.0,
    reasoning: bool = False,
    dry_run: bool = False,
    resume: bool = False,
    skip_failures: bool = False,
    model_resolver: Callable[[str], str] = lambda label: label,
) -> TournamentResult:
    """Schedule, run, and persist a full cross-play sweep, returning the result.

    `dry_run` uses scripted werewolves-win games (no `api_key` needed); otherwise each
    seat plays its label's resolved model through the real runner. `resume` reuses any
    already-completed game directory (re-running only the missing games — so a crashed
    sweep finishes without re-paying for finished games), then re-aggregates the full
    set. `skip_failures` keeps the sweep going past a single game's crash. Writes each
    game's artifacts under `output_dir/<game_id>/` plus `output_dir/summary.json`.
    """
    matchups = schedule_tournament(models, games_per_pair=games_per_pair, seed=seed, names=names)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _play(m: Matchup) -> GameMetrics:
        if dry_run:
            return run_one_scripted_game(m, output_dir=output_dir)
        if api_key is None:
            raise ValueError("run_sweep requires api_key in real mode (pass dry_run=True for scripted games)")
        # Each real game runs in its own process so its file descriptors are
        # reclaimed on exit (see run_one_game_isolated) — the parent stays under the
        # per-process fd cap no matter how long the sweep runs.
        job = _GameJob(
            matchup=m,
            output_dir=str(output_dir),
            dry_run=False,
            api_key=api_key,
            resolved_models=tuple((name, model_resolver(label)) for name, label in m.seat_models),
            max_iters=max_iters,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning=reasoning,
        )
        return run_one_game_isolated(job)

    def runner(m: Matchup) -> GameMetrics:
        game_dir = output_dir / m.game_id
        # `manifest.json` is the completeness sentinel: `_write_outputs` writes it
        # LAST (after events/trajectories/memories), so its presence means the dir
        # is whole. Gating on `events.jsonl` would treat a torn write (crash between
        # events and manifest) as complete, and `extract_run_dir` would then fall
        # back to inferred/`unknown` models and mis-rate that game.
        if resume and (game_dir / "manifest.json").exists():
            return extract_run_dir(game_dir)
        return _play(m)

    result = run_tournament(matchups, runner, max_concurrency=concurrency, skip_failures=skip_failures)
    write_tournament_summary(result, output_dir / "summary.json")
    return result


def _default_output_dir() -> str:
    return f"games/{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-tournament"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdb-tournament",
        description="Run a seeded pairwise cross-play Werewolf sweep and write a TrueSkill summary.",
    )
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model labels to sweep.")
    parser.add_argument("--games-per-pair", type=int, default=10, help="Games per model pair (default: 10).")
    parser.add_argument(
        "--seed",
        type=int,
        default=secrets.randbelow(2**31),
        help="Master seed for the schedule. Default: random in [0, 2**31).",
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Max games run in parallel (default: 10).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_default_output_dir(),
        help="Output directory. Default: ./games/<UTC-timestamp>-tournament/.",
    )
    parser.add_argument("--max-iters", type=int, default=9, help="Max ReAct iterations per decision (default: 9).")
    parser.add_argument(
        "--max-tokens", type=int, default=8000, help="Max completion tokens per LM call (default: 8000)."
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0).")
    parser.add_argument("--reasoning", action="store_true", help="Enable the model's reasoning/thinking tokens.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use scripted werewolves-win games (no API key, no API calls) for plumbing tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse already-completed game directories under --output-dir (re-run only "
            "the missing games), then re-aggregate. Pass the SAME --models/--games-per-pair/"
            "--seed so the schedule matches. Use to recover a crashed sweep without re-paying."
        ),
    )
    return parser


def _exit_code(result: TournamentResult, n_scheduled: int) -> int:
    """0 normally; 1 when games were scheduled but every one failed.

    A totally-failed unattended sweep must not look like success (it would otherwise
    exit 0 with an empty leaderboard). An empty schedule (`n_scheduled == 0`) is a
    no-op, not a failure.
    """
    if n_scheduled and not result.games:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 2 when a real run lacks an API key.

    Returns 1 when games were scheduled but all of them failed.
    """
    args = _build_parser().parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Fail loud BEFORE scheduling or creating any output, so a keyless real run
    # spends nothing and leaves no artifacts (CLAUDE.md "Safety & Permissions").
    api_key: str | None = None
    if not args.dry_run:
        cfg = settings.load()
        if cfg.openrouter_api_key is None:
            print(
                "error: OPENROUTER_API_KEY environment variable is not set.\n"
                "  Set it (e.g. `export OPENROUTER_API_KEY=sk-...`) or pass `--dry-run` to skip API calls.",
                file=sys.stderr,
            )
            return 2
        api_key = cfg.openrouter_api_key

    result = run_sweep(
        models=models,
        games_per_pair=args.games_per_pair,
        seed=args.seed,
        names=_DEFAULT_NAMES,
        output_dir=Path(args.output_dir),
        api_key=api_key,
        concurrency=args.concurrency,
        max_iters=args.max_iters,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning=args.reasoning,
        dry_run=args.dry_run,
        resume=args.resume,
        # A real unattended sweep should survive one game's crash; a dry-run stays
        # strict so a scripting bug fails loud.
        skip_failures=not args.dry_run,
    )

    print(f"tournament: {result.leaderboard.n_games} rated, {result.leaderboard.n_skipped} skipped")
    for rating in result.leaderboard.ratings:
        print(f"  {rating.model}: skill={rating.skill:.2f} mu={rating.mu:.2f} W{rating.wins}-L{rating.losses}")
    if result.failed:
        failed_ids = ", ".join(m.game_id for m in result.failed)
        print(f"warning: {len(result.failed)} game(s) failed: {failed_ids}", file=sys.stderr)
    print(f"summary: {Path(args.output_dir) / 'summary.json'}")
    return _exit_code(result, len(result.matchups) + len(result.failed))


if __name__ == "__main__":
    raise SystemExit(main())
