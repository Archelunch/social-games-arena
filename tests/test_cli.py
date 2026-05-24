"""Tests for the `sdb-werewolf` CLI.

The CLI is the user-facing entry point for running a Werewolf game. Tests
exercise the dry-run path (no `OPENROUTER_API_KEY` needed, no API calls)
to verify the plumbing: arg parsing, file paths, summary panel, and the
absent-API-key fail-loud guard for the real-LLM path.

Per CLAUDE.md rule 8: each test pins WHY the behavior matters so a CLI
regression — wrong output path, missing file, silent fallback — fails
loud rather than slipping through.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from social_deduction_bench.agents.trajectory import TrajectoryStream
from social_deduction_bench.cli import (
    _DEFAULT_NAMES,
    _build_manifest,
    _build_parser,
    _model_arg_summary,
    _resolve_seat_models,
    _seeded_roster,
    main,
    silence_litellm_logging_worker,
)
from social_deduction_bench.engine import EventStream
from social_deduction_bench.games.werewolf.config import DEFAULT_ROLE_COUNTS
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.rating.manifest import read_json as read_manifest


def _invoke(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    """Run `main(argv)` with no API key set and capture the exit code.

    `main` raises `SystemExit` on bad invocations or returns a non-zero
    code on handled mid-game failures; both shapes surface as the
    returned int so tests can assert against either. Tests that need
    the key MUST set it explicitly via `monkeypatch.setenv`.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        return main(argv) or 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1


# --- seeded role assignment ----------------------------------------------


def test_seeded_roster_varies_the_role_layout_with_the_seed() -> None:
    """The role column is dealt from the seed — it is not a fixed seat→role map.

    This is the fix for the seat-role bug: with a hardcoded roster every game had
    Alice=werewolf, Carol=seer, etc., so the wolves' blind night-1 kill always hit
    the seer and a model seated at a name always played the same role (breaking
    cross-play fairness). Distinct seeds must therefore yield distinct layouts.
    """
    layouts = {tuple(role for _, role in _seeded_roster(seed)) for seed in range(6)}
    assert len(layouts) > 1  # the seed actually moves roles around between seats


def test_seeded_roster_is_deterministic_for_a_fixed_seed() -> None:
    """Same seed → identical roster, every time (invariant #4).

    The role deal is a seeded draw; it must replay byte-identically or a recorded
    game would reconstruct a different board.
    """
    assert _seeded_roster(42) == _seeded_roster(42)


def test_seeded_roster_keeps_names_in_order_and_preserves_the_role_counts() -> None:
    """Only the role column is shuffled: names stay in `_DEFAULT_NAMES` order and
    the dealt roles are exactly the 2/1/1/3 default multiset.

    Shuffling names instead of roles, or dropping/duplicating a role, would either
    change the LM-seating contract or make the game unwinnable for a faction.
    """
    roster = _seeded_roster(7)
    assert tuple(name for name, _ in roster) == _DEFAULT_NAMES
    role_counts = Counter(role for _, role in roster)
    assert role_counts == Counter({r.value: c for r, c in DEFAULT_ROLE_COUNTS.items()})
    # every dealt role is a real Role (no typo/None slipped through the deal)
    assert all(role in Role.__members__.values() for _, role in roster)


# --- dry-run produces both JSONL files -----------------------------------


def test_dry_run_writes_events_and_trajectories_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--dry-run --output-dir DIR` writes all three artifact files to `DIR/<game-id>/`.

    The CLI's whole point is producing the artifacts. If any file is
    missing, the visualization (T31) and metric extraction (T24) have
    nothing to read.
    """
    exit_code = _invoke(
        [
            "--dry-run",
            "--seed",
            "42",
            "--output-dir",
            str(tmp_path),
            "--game-id",
            "dry-test",
        ],
        monkeypatch,
    )

    assert exit_code == 0
    events_path = tmp_path / "dry-test" / "events.jsonl"
    trajectories_path = tmp_path / "dry-test" / "trajectories.jsonl"
    memories_path = tmp_path / "dry-test" / "memories.json"
    assert events_path.exists()
    assert trajectories_path.exists()
    assert memories_path.exists()


def test_dry_run_memories_file_round_trips_to_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`memories.json` is valid JSON with `game_id` and `memories` keys.

    The dump is the agent-private slice (plan + beliefs + notes) the
    operator needs to inspect after a real game. The dry-run path uses
    `ScriptedDecisions` (no `GameMemory`) so `memories` is the empty
    object — the file convention still holds.
    """
    import json

    _invoke(
        ["--dry-run", "--seed", "3", "--output-dir", str(tmp_path), "--game-id", "g3"],
        monkeypatch,
    )

    payload = json.loads((tmp_path / "g3" / "memories.json").read_text(encoding="utf-8"))
    assert payload["game_id"] == "g3"
    assert payload["memories"] == {}


def test_dry_run_events_file_is_a_valid_event_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The written events.jsonl reads back through `EventStream.from_jsonl_lines`.

    Invariant #5: every game is a valid, replayable JSONL transcript.
    A corrupt write would make replay impossible — pin the round-trip.
    """
    _invoke(
        ["--dry-run", "--seed", "7", "--output-dir", str(tmp_path), "--game-id", "g7"],
        monkeypatch,
    )

    events_path = tmp_path / "g7" / "events.jsonl"
    text = events_path.read_text(encoding="utf-8")
    stream = EventStream.from_jsonl_lines(text.splitlines())

    assert stream.header.seed == 7
    assert stream.header.game_id == "g7"
    # GAME_OVER terminator must be present.
    assert any(event.type == "game_over" for event in stream.log.events)


def test_dry_run_trajectories_file_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The written trajectories.jsonl reads back through `TrajectoryStream.from_jsonl_lines`.

    Even in dry-run (ScriptedDecisions has no trajectories), the file
    must exist with a valid header so T31 can read it uniformly across
    real and dry runs.
    """
    _invoke(
        ["--dry-run", "--seed", "11", "--output-dir", str(tmp_path), "--game-id", "g11"],
        monkeypatch,
    )

    path = tmp_path / "g11" / "trajectories.jsonl"
    text = path.read_text(encoding="utf-8")
    stream = TrajectoryStream.from_jsonl_lines(text.splitlines())

    assert stream.header.seed == 11
    assert stream.header.game_id == "g11"
    # ScriptedDecisions has no trajectories — file is header-only, valid.
    assert stream.trajectories == ()


def test_dry_run_seed_threads_through_to_event_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The `--seed N` argument lands as `StreamHeader.seed` in the output file.

    Replay reproducibility (invariant #4) depends on the seed being
    captured verbatim. A CLI bug that swallowed the seed would silently
    break determinism for everyone using this entrypoint.
    """
    _invoke(
        ["--dry-run", "--seed", "12345", "--output-dir", str(tmp_path), "--game-id", "g"],
        monkeypatch,
    )

    stream = EventStream.from_jsonl_lines((tmp_path / "g" / "events.jsonl").read_text("utf-8").splitlines())
    assert stream.header.seed == 12345


# --- mid-game failure handling -------------------------------------------


def test_help_documents_temperature_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """`--help` advertises the temperature knob.

    Temperature is the operator's main lever when a small model gets
    stuck in a repetition loop (DSPy itself logs this guidance). The
    flag must be discoverable from `--help`.
    """
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "--temperature" in capsys.readouterr().out


def test_adapter_parse_failure_exits_cleanly_with_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mid-game `AdapterParseError` produces actionable stderr, not a stack trace.

    Small models (e.g. qwen3.5-9b) commonly emit unparseable ReAct
    responses. The user must see the suggested knobs (--temperature,
    --max-tokens, stronger --model) instead of a raw traceback they
    have to grep through.
    """
    from typing import ClassVar

    from dspy.utils.exceptions import AdapterParseError

    from social_deduction_bench import cli as cli_module

    class _StubSig:
        output_fields: ClassVar[dict[str, object]] = {"committed_action": None}

    def fail(*_a: object, **_kw: object) -> None:
        raise AdapterParseError(
            adapter_name="ChatAdapter",
            signature=_StubSig,  # type: ignore[arg-type]
            lm_response="garbage",
            message="The LM returned an empty or null response.",
        )

    monkeypatch.setattr(cli_module, "run_game", fail)
    exit_code = _invoke(
        ["--dry-run", "--seed", "1", "--output-dir", str(tmp_path), "--game-id", "fail"],
        monkeypatch,
    )

    assert exit_code == 3
    err = capsys.readouterr().err
    assert "DSPy could not parse" in err
    assert "--temperature" in err
    assert "--max-tokens" in err


def test_uncommitted_runtime_error_exits_cleanly_with_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A `RuntimeError` from react_decide ("no commit") surfaces with iteration guidance.

    When a model uses cognitive tools indefinitely without calling a
    terminal one, react_decide raises. The CLI must explain what to do
    (raise --max-iters or pick a stronger model) instead of crashing.
    """
    from social_deduction_bench import cli as cli_module

    def fail(*_a: object, **_kw: object) -> None:
        raise RuntimeError("agent 'Wolf1' finished without a committed game action")

    monkeypatch.setattr(cli_module, "run_game", fail)
    exit_code = _invoke(
        ["--dry-run", "--seed", "1", "--output-dir", str(tmp_path), "--game-id", "no-commit"],
        monkeypatch,
    )

    assert exit_code == 3
    err = capsys.readouterr().err
    assert "without committing" in err
    assert "--max-iters" in err


def test_unrelated_exception_still_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-LM exception (e.g. KeyError) is not swallowed by the failure handler.

    The handler converts known LM failure modes to clean exits; everything
    else must keep its stack trace so real bugs stay visible.
    """
    from social_deduction_bench import cli as cli_module

    def fail(*_a: object, **_kw: object) -> None:
        raise KeyError("unexpected!")

    monkeypatch.setattr(cli_module, "run_game", fail)
    with pytest.raises(KeyError, match="unexpected"):
        main(["--dry-run", "--seed", "1", "--output-dir", str(tmp_path), "--game-id", "bug"])


# --- missing API key in real mode ----------------------------------------


def test_real_mode_without_api_key_exits_with_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without `OPENROUTER_API_KEY`, the real-mode CLI exits non-zero with a clear message.

    The CLI must not silently make API calls without a key. CLAUDE.md
    "Ask first" rule: paid LLM API runs require explicit setup. The
    error must name the missing env var so the user knows what to do.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        main(["--output-dir", str(tmp_path), "--game-id", "no-key"])

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "OPENROUTER_API_KEY" in combined


# --- help / version ------------------------------------------------------


def test_help_flag_exits_zero_and_documents_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    """`--help` exits zero and the help text names every public flag.

    A CLI without working `--help` is hostile. The text must teach the
    user the dry-run path so they can experiment without API cost.
    """
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    help_text = captured.out
    for flag in ("--seed", "--model", "--output-dir", "--game-id", "--dry-run", "--ascii", "--quiet"):
        assert flag in help_text


def test_unknown_flag_exits_non_zero() -> None:
    """Argparse rejects unknown flags with a non-zero exit.

    Silently ignoring an unknown flag would mask typos and produce
    surprising behavior on the next run.
    """
    with pytest.raises(SystemExit) as exc_info:
        main(["--no-such-flag"])
    assert exc_info.value.code != 0


# --- default game-id and output-dir --------------------------------------


def test_default_output_dir_is_under_games(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When `--output-dir` is omitted, the CLI writes under `./games/<game-id>/`.

    A predictable default keeps users from having to invent paths for
    casual runs. The convention is documented in `--help`.
    """
    monkeypatch.chdir(tmp_path)
    _invoke(["--dry-run", "--seed", "1", "--game-id", "default-dir-test"], monkeypatch)

    out_dir = tmp_path / "games" / "default-dir-test"
    assert out_dir.exists()
    assert (out_dir / "events.jsonl").exists()
    assert (out_dir / "trajectories.jsonl").exists()


def test_max_iters_default_is_nine() -> None:
    """`--max-iters` defaults to 9.

    Most real commits land by iter 4, but weak benchmark models need headroom
    to reach a valid commit before the loop degrades to the phase default.
    9 buys that headroom without the dead weight of the legacy 20-cap (a stuck
    loop still bounds its LM calls). It is not 6: 6 was tuned for a fail-loud
    world; now a no-commit degrades instead of aborting, so a slightly higher
    cap trades a little worst-case latency for fewer degraded turns.
    """
    from social_deduction_bench.cli import _build_parser

    args = _build_parser().parse_args([])
    assert args.max_iters == 9


def test_max_tokens_default_is_lowered_to_8000() -> None:
    """`--max-tokens` defaults to 8000, not the legacy 24000.

    `max_tokens` caps *output*, and a reasoning-prone small model loops
    until it hits the cap, so a 24000 default invites 30k-token repetition
    spirals (and a parse-retry that doubles them). It also shares the
    context window with the input, so a smaller cap leaves more room for
    the brief. With reasoning disabled by default, 8000 is generous for a
    single decision and caps any residual runaway.
    """
    from social_deduction_bench.cli import _build_parser

    args = _build_parser().parse_args([])
    assert args.max_tokens == 8000


def test_reasoning_is_disabled_by_default_and_optingin_enables_it() -> None:
    """Reasoning/thinking tokens are off by default; `--reasoning` re-enables.

    The benchmark seats reasoning-prone small models whose hidden thinking
    is billed as output and drives the repetition spirals. Disabling it by
    default collapses output and latency; the flag is the escape hatch for
    a model that legitimately needs (or mandates) reasoning.
    """
    from social_deduction_bench.cli import _build_parser

    assert _build_parser().parse_args([]).reasoning is False
    assert _build_parser().parse_args(["--reasoning"]).reasoning is True


def test_reasoning_extra_body_disables_generation_by_default() -> None:
    """`_reasoning_extra_body` disables OpenRouter reasoning unless opted in.

    `reasoning.enabled=false` stops the model GENERATING reasoning tokens
    (vs `exclude` which still generates them, just hides them). When
    reasoning is enabled we pass no override so the model's default applies.
    """
    from social_deduction_bench.cli import _reasoning_extra_body

    assert _reasoning_extra_body(reasoning=False) == {"reasoning": {"enabled": False}}
    assert _reasoning_extra_body(reasoning=True) is None


def test_help_documents_reasoning_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """`--reasoning` is discoverable from `--help` (it's the escape hatch)."""
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "--reasoning" in capsys.readouterr().out


def test_silence_litellm_logging_worker_closes_enqueued_coroutine_without_scheduling() -> None:
    """The silencer must close an enqueued telemetry coroutine, not schedule it.

    After every async completion litellm enqueues a fire-and-forget
    `async_success_handler` coroutine into ONE process-global LoggingWorker. We
    run a fresh event loop per agent phase across many game threads, so that
    singleton is shared across loops/threads it was never built for and floods
    stderr ("Task was destroyed but it is pending", "cannot reuse already awaited
    coroutine", "task_done() called too many times"), burying real errors during a
    paid sweep. We register no litellm callbacks, so the worker is dead weight.
    After silencing, an enqueued coroutine must be CLOSED (no leaked "never
    awaited" coroutine, no background task) — that is what keeps a concurrent
    sweep's logs readable and the worker's state uncorrupted. Were the shim
    absent, the real enqueue would instead call `asyncio.create_task` (here:
    RuntimeError, no running loop) and leave the coroutine un-awaited.
    """
    import inspect

    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    silence_litellm_logging_worker()

    async def _sample() -> None:
        return None

    coro = _sample()
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CREATED  # fresh, unstarted
    GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(async_coroutine=coro)
    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED  # shim closed it, no task


def test_no_stream_thoughts_flag_round_trips(capsys: pytest.CaptureFixture[str]) -> None:
    """`--no-stream-thoughts` is discoverable from `--help`.

    Operators who pipe game output to grep want the dim wrap-line off so
    every printed line is a tool call. The flag advertised here is wired
    into `PrinterSettings.stream_thoughts=False`; absence of the flag
    keeps the default real-time stream on.
    """
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "--no-stream-thoughts" in out


def test_dry_run_completes_under_one_second(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Async fan-out must not slow down the no-LM dry-run path.

    Concurrency exists to amortize remote LM latency; with scripted
    decisions (no I/O), the asyncio.run + gather overhead per phase
    should be negligible. If this test regresses, async fan-out picked
    up a real serialization cost somewhere.
    """
    import time as _time

    monkeypatch.chdir(tmp_path)
    t0 = _time.monotonic()
    _invoke(["--dry-run", "--seed", "9", "--game-id", "speed"], monkeypatch)
    elapsed = _time.monotonic() - t0
    assert elapsed < 2.0, f"dry-run took {elapsed:.2f}s; async overhead regressed"


# --- per-seat models: faction-split mode ---------------------------------

_FACTION_ROSTER: tuple[tuple[str, str], ...] = (
    ("Alice", Role.WEREWOLF.value),
    ("Bob", Role.WEREWOLF.value),
    ("Carol", Role.SEER.value),
    ("Dave", Role.DOCTOR.value),
    ("Eve", Role.VILLAGER.value),
)


def test_resolve_seat_models_uniform_seats_one_model_everywhere() -> None:
    seat_models = _resolve_seat_models(_FACTION_ROSTER, model="qwen", werewolf_model=None, villager_model=None)
    assert seat_models == {name: "qwen" for name, _ in _FACTION_ROSTER}


def test_resolve_seat_models_faction_split_maps_every_role_by_faction() -> None:
    # Pins all four roles: only the werewolf role takes the wolf model; seer, doctor,
    # and plain villager (all the villager faction) take the village model.
    seat_models = _resolve_seat_models(_FACTION_ROSTER, model="ignored", werewolf_model="wolfM", villager_model="vilM")
    assert seat_models == {
        "Alice": "wolfM",
        "Bob": "wolfM",
        "Carol": "vilM",
        "Dave": "vilM",
        "Eve": "vilM",
    }


def test_faction_split_follows_dealt_roles_across_seeds_not_seat_positions() -> None:
    """The wolf model lands on whichever seats the seed dealt as werewolves.

    Roles rotate across seeds (`_seeded_roster`), so a faction-based assignment must
    move with them — otherwise a model pinned to a seat name would systematically
    always (or never) play wolf, breaking cross-play fairness.
    """
    wolf_seat_sets: set[frozenset[str]] = set()
    for seed in range(6):
        roster = _seeded_roster(seed)
        seat_models = _resolve_seat_models(roster, model="x", werewolf_model="W", villager_model="V")
        wolves = frozenset(name for name, role in roster if role == Role.WEREWOLF.value)
        wolf_seat_sets.add(wolves)
        assert {name for name, m in seat_models.items() if m == "W"} == set(wolves)
        assert {name for name, m in seat_models.items() if m == "V"} == {name for name, _ in roster} - set(wolves)
    assert len(wolf_seat_sets) > 1  # the wolf seats actually move across seeds


def test_resolve_seat_models_requires_both_faction_models() -> None:
    with pytest.raises(ValueError, match="both"):
        _resolve_seat_models(_FACTION_ROSTER, model="x", werewolf_model="W", villager_model=None)
    with pytest.raises(ValueError, match="both"):
        _resolve_seat_models(_FACTION_ROSTER, model="x", werewolf_model=None, villager_model="V")


def test_parser_exposes_faction_model_flags_defaulting_none() -> None:
    args = _build_parser().parse_args([])
    assert args.werewolf_model is None
    assert args.villager_model is None
    args = _build_parser().parse_args(["--werewolf-model", "W", "--villager-model", "V"])
    assert args.werewolf_model == "W"
    assert args.villager_model == "V"


def test_help_documents_faction_model_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "--werewolf-model" in out
    assert "--villager-model" in out


def test_model_arg_summary_reflects_the_mode() -> None:
    assert _model_arg_summary(_build_parser().parse_args(["--model", "qwen"])) == "qwen"
    assert _model_arg_summary(_build_parser().parse_args(["--dry-run"])) == "scripted"
    faction = _model_arg_summary(_build_parser().parse_args(["--werewolf-model", "W", "--villager-model", "V"]))
    assert "W" in faction
    assert "V" in faction


def test_faction_mode_with_only_one_model_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A half-specified faction mode must fail loud with a clean arg-error exit (2)
    # and NEVER silently seat a default model on the missing faction.
    # The key is SET here so a missing-key exit (also code 2) can't be mistaken for
    # the faction error: this proves the faction validation fires first, and the
    # stderr message confirms it is the faction error rather than the key guard.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        main(["--werewolf-model", "x", "--seed", "1"])
    assert exc.value.code == 2
    assert "both --werewolf-model and --villager-model" in capsys.readouterr().err


def test_build_manifest_records_per_faction_seat_models() -> None:
    args = _build_parser().parse_args(["--werewolf-model", "W", "--villager-model", "V"])
    roster = (("Alice", Role.WEREWOLF.value), ("Bob", Role.SEER.value))
    seat_models = _resolve_seat_models(
        roster, model=args.model, werewolf_model=args.werewolf_model, villager_model=args.villager_model
    )
    manifest = _build_manifest(
        args=args, roster=roster, seed=1, game_id="g", winner="villagers", rounds=2, seat_models=seat_models
    )
    assert dict(manifest.models) == {"Alice": "W", "Bob": "V"}
    assert "werewolves=W" in manifest.model_arg
    assert "villagers=V" in manifest.model_arg


def test_dry_run_ignores_faction_flags_and_records_scripted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Dry-run is scripted (no LMs); faction flags must not seat anything, and the
    # manifest must record the scripted sentinel rather than the requested models.
    code = _invoke(
        [
            "--dry-run",
            "--werewolf-model",
            "W",
            "--villager-model",
            "V",
            "--game-id",
            "fac",
            "--output-dir",
            str(tmp_path),
        ],
        monkeypatch,
    )
    assert code == 0
    manifest = read_manifest(tmp_path / "fac" / "manifest.json")
    assert set(dict(manifest.models).values()) == {"scripted"}
    assert manifest.model_arg == "scripted"
