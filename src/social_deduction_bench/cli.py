"""`sdb-werewolf` — run one Werewolf game from the terminal.

Entry point: `poetry run sdb-werewolf` (or `python -m social_deduction_bench`).
Drives `run_game` with either:

- `ReActDecisionSource` seating one LLM (default `qwen/qwen3.5-9b` via
  OpenRouter) at every roster slot, OR
- `ScriptedDecisions` in `--dry-run` mode for offline CLI testing.

Writes both `events.jsonl` and `trajectories.jsonl` to
`<output-dir>/<game-id>/` and streams the game live through the rich
`GamePrinter`.

Per CLAUDE.md "Safety & Permissions": paid-LLM runs require explicit
user authorization. The CLI is that authorization vehicle — without
`OPENROUTER_API_KEY` set, real-mode runs fail loud rather than silently
making API calls.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from social_deduction_bench import settings
from social_deduction_bench.agents.decisions import ReActDecisionSource
from social_deduction_bench.agents.trajectory import (
    Trajectory,
    TrajectoryStream,
)
from social_deduction_bench.agents.trajectory import (
    write_jsonl as write_trajectories_jsonl,
)
from social_deduction_bench.engine import Event, EventStream, GameRNG, GameState, write_jsonl
from social_deduction_bench.games.werewolf.assignment import assign_default_roles
from social_deduction_bench.games.werewolf.config import REACTION_MAX_TOKENS
from social_deduction_bench.games.werewolf.day import DayActions
from social_deduction_bench.games.werewolf.events import ABSTAIN, GAME_OVER, EventDraft
from social_deduction_bench.games.werewolf.loop import DecisionSource, run_game
from social_deduction_bench.games.werewolf.night import NightActions
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.games.werewolf.scripted import ScriptedDecisions
from social_deduction_bench.printer import GamePrinter, PrinterSettings

# Player names must NOT encode role information — that would leak hidden
# state to every agent (invariant #2). The smoke / unit tests use names
# like "Wolf1" intentionally for readability, but the CLI is the real
# benchmark entry point, so its default roster uses neutral first names.
_DEFAULT_NAMES: tuple[str, ...] = ("Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace")

# Fixed-role roster used ONLY by `--dry-run`: its scripted decisions hardcode who
# is a wolf (Alice/Bob), so the dry-run must keep these exact role assignments.
# Real (LLM) runs assign roles from the seed via `_seeded_roster` instead, so the
# same seat plays different roles across games and the wolves cannot systematically
# kill the same role on night 1.
_DEFAULT_ROSTER: tuple[tuple[str, str], ...] = (
    ("Alice", Role.WEREWOLF.value),
    ("Bob", Role.WEREWOLF.value),
    ("Carol", Role.SEER.value),
    ("Dave", Role.DOCTOR.value),
    ("Eve", Role.VILLAGER.value),
    ("Frank", Role.VILLAGER.value),
    ("Grace", Role.VILLAGER.value),
)
_DEFAULT_MODEL = "qwen/qwen3.5-9b"


def _seeded_roster(seed: int) -> tuple[tuple[str, str], ...]:
    """Deal the default roles onto `_DEFAULT_NAMES` from the engine seed.

    Wires in the T10 seeded assignment so the role column varies by seed (same
    seed → identical roster, invariant #4). Uses a `GameRNG(seed)` independent of
    the one `run_game` builds internally, so the assignment draw never perturbs
    the in-game draws; both are reproducible from `seed`.
    """
    return assign_default_roles(_DEFAULT_NAMES, GameRNG(seed))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Parses arguments, validates env, builds the appropriate decision
    source (real LLM or scripted dry-run), runs the game with live
    rendering, and writes both JSONL artifacts. Returns 0 on success;
    `SystemExit` with non-zero code on argument errors or missing
    `OPENROUTER_API_KEY`.
    """
    args = _build_parser().parse_args(argv)

    game_id = args.game_id or _default_game_id()
    output_parent = Path(args.output_dir) if args.output_dir else Path("games")
    output_dir = output_parent / game_id
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    # Real runs deal roles from the seed (so a seat plays different roles across
    # games and the wolves can't always hit the same role on night 1); the
    # dry-run keeps the fixed roster its scripted decisions are written against.
    roster = _DEFAULT_ROSTER if args.dry_run else _seeded_roster(seed)

    # Force line-buffered stdout so each Rich `print` flushes immediately.
    # Without this, a long-running game holds output in the libc buffer and
    # the operator sees nothing for tens of seconds at a time.
    reconfigure = getattr(console_file := sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)
    console = Console(file=console_file)
    printer_settings = PrinterSettings(
        ascii=args.ascii,
        quiet=args.quiet,
        stream_iterations=not args.quiet,
        stream_thoughts=args.stream_thoughts and not args.quiet,
    )
    printer = GamePrinter(console, settings=printer_settings)

    printer.print_header(seed=seed, model=args.model, roster=roster)

    if args.dry_run:
        source: DecisionSource = _build_scripted_source()
    else:
        cfg = settings.load()
        if cfg.openrouter_api_key is None:
            print(
                "error: OPENROUTER_API_KEY environment variable is not set.\n"
                "  Set it (e.g. `export OPENROUTER_API_KEY=sk-...`) or pass `--dry-run` to skip API calls.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        source = _build_react_source(
            roster=roster,
            model=args.model,
            api_key=cfg.openrouter_api_key,
            max_iters=args.max_iters,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning=args.reasoning,
            on_trajectory=printer.on_trajectory,
            on_decision_start=printer.on_decision_start,
            on_step=printer.on_react_step,
            on_thought_chunk=printer.on_thought_chunk if printer_settings.stream_thoughts else None,
        )

    wrapped = _PrintingDecisionSource(inner=source, printer=printer)

    t_start = time.monotonic()
    try:
        stream = run_game(
            roster=roster,
            seed=seed,
            decisions=wrapped,
            game_id=game_id,
            max_rounds=args.max_rounds,
        )
    except Exception as err:
        elapsed = time.monotonic() - t_start
        return _handle_run_failure(err, source=source, args=args, elapsed=elapsed)
    elapsed = time.monotonic() - t_start

    trajectories = _trajectories_from(source)
    memories_dump = _memories_from(source)
    events_path, trajectories_path, memories_path = _write_outputs(
        stream, trajectories, memories_dump, output_dir, game_id
    )

    winner_event = next((e for e in stream.log.events if e.type == GAME_OVER), None)
    winner_payload = winner_event.payload.get("winner") if winner_event is not None else "?"
    winner = str(winner_payload)
    final_round, final_phase = _final_position(stream)

    printer.print_summary(
        winner=winner,
        final_round=final_round,
        final_phase=final_phase,
        rounds_played=final_round,
        events_path=str(events_path),
        trajectories_path=str(trajectories_path),
        memories_path=str(memories_path),
        elapsed_seconds=elapsed,
    )
    printer.print_memories(memories=memories_dump, roster=roster)
    return 0


# --- arg parsing ---------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdb-werewolf",
        description=(
            "Run one Werewolf game and write the engine event stream + agent "
            "trajectory sidecar to disk, streaming the game live to the terminal."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Engine seed for replayable runs. Default: random in [0, 2**31).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_default_model(),
        help=f"Uniform model for all 7 seats (default: {_DEFAULT_MODEL}, override via $SDB_SMOKE_MODEL).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=6,
        help=(
            "Max ReAct iterations per decision-point loop (default: 6). Most "
            "successful commits land by iter 4; raise toward 10-20 only if a "
            "weak model legitimately needs more chain steps."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help=(
            "Max completion tokens per LM call (default: 8000). Caps output, not "
            "input; with reasoning off this is generous for one decision. Raise it "
            "(and try --reasoning) only if a model legitimately truncates."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help=(
            "Sampling temperature for every LM (default: 0.7). Raise toward 1.0 if the "
            "model gets stuck in repetition loops; lower toward 0.0 for more determinism "
            "(may worsen small-model loops)."
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="Game-round cap before run_game raises (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: ./games/<game-id>/.",
    )
    parser.add_argument(
        "--game-id",
        type=str,
        default=None,
        help="Game identifier (and output subdirectory name). Default: <UTC-timestamp>-werewolf.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use scripted decisions (no API calls, no API key needed) for CLI plumbing tests.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="ASCII-only glyphs for terminals that can't render emoji.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trajectory rendering; only phase banners and resolutions print.",
    )
    parser.add_argument(
        "--no-stream-thoughts",
        dest="stream_thoughts",
        action="store_false",
        help=(
            "Disable the dim per-iter thought wrap-line. Keeps parallelism + "
            "sub-bullets but produces grep-friendly logs (one printed line per tool call)."
        ),
    )
    parser.set_defaults(stream_thoughts=True)
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help=(
            "Enable the model's reasoning/thinking tokens (default: disabled). Off "
            "by default because hidden thinking is billed as output and drives the "
            "repetition spirals small models fall into. Pass this for a model that "
            "needs — or mandates — reasoning."
        ),
    )
    return parser


def _default_model() -> str:
    cfg = settings.load()
    return cfg.smoke_model or _DEFAULT_MODEL


def _default_game_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-werewolf"


# --- decision sources ----------------------------------------------------


def _reasoning_extra_body(*, reasoning: bool) -> dict[str, object] | None:
    """Return the litellm `extra_body` that disables OpenRouter reasoning, or `None`.

    OpenRouter's documented control for hybrid models (e.g. qwen3.5) is the
    `reasoning` object: `enabled=false` stops the model GENERATING reasoning
    tokens, whereas `exclude=true` would still generate them and merely hide
    them from the response. We want generation off, so we send
    `{"reasoning": {"enabled": False}}` when reasoning is disabled. When it is
    enabled we send no override so the model's own default applies. Pure +
    side-effect-free so it is unit-testable without constructing an LM.

    Caveat: a model that *mandates* reasoning rejects `enabled=false`; the
    operator re-enables with `--reasoning` in that case.
    """
    if reasoning:
        return None
    return {"reasoning": {"enabled": False}}


def _build_react_source(
    *,
    roster: Sequence[tuple[str, str]],
    model: str,
    api_key: str,
    max_iters: int,
    max_tokens: int,
    temperature: float,
    reasoning: bool,
    on_trajectory: object,
    on_decision_start: object,
    on_step: object,
    on_thought_chunk: object,
) -> ReActDecisionSource:
    """Build a `ReActDecisionSource` with one OpenRouter LM per seat.

    `cache=False` mirrors the existing smoke test (T23) so a re-run
    actually exercises the model rather than replaying a cached response.
    `max_tokens` and `temperature` are exposed via the CLI so the
    operator can match the per-decision budget and sampling to the
    chosen model. `reasoning` is off by default (see `_reasoning_extra_body`):
    a reasoning-prone small model's hidden thinking is billed as output and
    drives the repetition spirals, so we disable generation unless asked.
    """
    import dspy  # imported lazily so dry-run / --help don't pay the import cost

    # `extra_body=None` is litellm's default (no override), so passing it
    # unconditionally is equivalent to omitting it when reasoning is enabled.
    extra_body = _reasoning_extra_body(reasoning=reasoning)
    lms = {
        name: dspy.LM(
            f"openrouter/{model}",
            api_key=api_key,
            temperature=temperature,
            cache=False,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        for name, _ in roster
    }
    # The day reaction round runs a short-capped clone of each seat's LM so the
    # extra per-player call stays terse (config.REACTION_MAX_TOKENS). `lm.copy`
    # only updates the `max_tokens` kwarg; api_key / temperature / reasoning carry
    # over.
    reaction_lms = {name: lm.copy(max_tokens=REACTION_MAX_TOKENS) for name, lm in lms.items()}
    return ReActDecisionSource(
        roster=tuple(roster),
        lms=lms,
        reaction_lms=reaction_lms,
        max_iters=max_iters,
        on_trajectory=on_trajectory,  # type: ignore[arg-type]
        on_decision_start=on_decision_start,  # type: ignore[arg-type]
        on_step=on_step,  # type: ignore[arg-type]
        on_thought_chunk=on_thought_chunk,  # type: ignore[arg-type]
    )


def _build_scripted_source() -> ScriptedDecisions:
    """Build a deterministic scripted source ending in a 3-round werewolves win.

    The 7-seat roster loses Eve → Frank → Carol over three nights;
    days are all-abstain so the wolves reach parity. Used only by
    `--dry-run` for CLI plumbing tests.
    """
    return ScriptedDecisions(
        nights=[
            NightActions(
                kill_votes={"Alice": "Eve", "Bob": "Eve"},
                seer_inspect="Alice",
                doctor_protect="Frank",
            ),
            NightActions(
                kill_votes={"Alice": "Frank", "Bob": "Frank"},
                seer_inspect="Alice",
                doctor_protect="Grace",
            ),
            NightActions(
                kill_votes={"Alice": "Carol", "Bob": "Carol"},
                seer_inspect="Bob",
                doctor_protect="Grace",
            ),
        ],
        days=[
            DayActions(exile_votes={n: ABSTAIN for n in ("Alice", "Bob", "Carol", "Dave", "Frank", "Grace")}),
            DayActions(exile_votes={n: ABSTAIN for n in ("Alice", "Bob", "Carol", "Dave", "Grace")}),
        ],
    )


# --- wrappers + outputs --------------------------------------------------


class _PrintingDecisionSource:
    """Thin wrapper that delegates every Protocol method to `inner`.

    Adds one side effect: in `observe`, every newly-appended engine
    event is fed to the printer (the printer filters redundant ones).
    Trajectories are surfaced through the `ReActDecisionSource`'s
    `on_trajectory` callback, not here — that fires DURING a phase,
    while `observe` fires only after a phase resolves.
    """

    def __init__(self, *, inner: DecisionSource, printer: GamePrinter) -> None:
        self._inner = inner
        self._printer = printer

    def night_chat(self, state: GameState, /) -> None:
        self._inner.night_chat(state)

    def night_actions(self, state: GameState, /) -> NightActions:
        return self._inner.night_actions(state)

    def day_actions(self, state: GameState, /) -> DayActions:
        return self._inner.day_actions(state)

    def bids(self, state: GameState, /) -> dict[str, int]:
        return self._inner.bids(state)

    def next_speech(self, state: GameState, speaker: str, /) -> str:
        return self._inner.next_speech(state, speaker)

    def next_reaction(self, state: GameState, reactor: str, /) -> None:
        self._inner.next_reaction(state, reactor)

    def drain_drafts(self) -> tuple[EventDraft, ...]:
        return self._inner.drain_drafts()

    def observe(self, state: GameState, new_events: tuple[Event, ...], /) -> None:
        for event in new_events:
            self._printer.on_event(event)
        self._inner.observe(state, new_events)


def _trajectories_from(source: DecisionSource) -> tuple[Trajectory, ...]:
    """Pull the trajectory snapshot from `source` if it exposes one (`ReActDecisionSource`).

    `ScriptedDecisions` has no trajectories; an empty tuple is fine —
    the sidecar file still gets written with a header line so the
    output convention is uniform.
    """
    return getattr(source, "trajectories", ())


def _memories_from(source: DecisionSource) -> dict[str, dict[str, object]]:
    """Pull the per-agent memory snapshot from `source` if it exposes one.

    Returns a mapping of `player_name -> {plan, beliefs, notes}` via
    `GameMemory.to_json_dict()`. `ScriptedDecisions` has no memories;
    an empty dict is fine — the dump file still gets written so the
    output convention is uniform.
    """
    raw = getattr(source, "memories", {})
    out: dict[str, dict[str, object]] = {}
    for name, memory in raw.items():
        to_json = getattr(memory, "to_json_dict", None)
        if not callable(to_json):
            continue
        dumped = to_json()
        if isinstance(dumped, dict):
            out[name] = dumped
    return out


def _write_outputs(
    stream: EventStream,
    trajectories: tuple[Trajectory, ...],
    memories: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    game_id: str,
) -> tuple[Path, Path, Path]:
    """Write `events.jsonl`, `trajectories.jsonl`, and `memories.json` to `output_dir`.

    `output_dir` is created (with parents) if missing. All three files
    share the same game identity (`stream.header.game_id`) so a downstream
    reader can confirm the pairing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    trajectories_path = output_dir / "trajectories.jsonl"
    memories_path = output_dir / "memories.json"
    write_jsonl(stream, events_path)
    write_trajectories_jsonl(
        TrajectoryStream(header=stream.header, trajectories=trajectories),
        trajectories_path,
    )
    memories_path.write_text(
        json.dumps(
            {"game_id": stream.header.game_id, "memories": memories},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return events_path, trajectories_path, memories_path


def _handle_run_failure(
    err: BaseException,
    *,
    source: DecisionSource,
    args: argparse.Namespace,
    elapsed: float,
) -> int:
    """Convert a mid-game exception into a clean exit with actionable guidance.

    Two failure modes are common when seating a small or weak LLM:

    - `dspy.utils.exceptions.AdapterParseError`: the model returned output
      that DSPy could not parse into the ReAct schema. Usually a
      repetition loop at low temperature.
    - `RuntimeError("agent ... finished without a committed game
      action")`: the loop hit `max_iters` without the model ever calling
      a terminal tool.

    Anything else re-raises so a real bug doesn't get swallowed.
    """
    from dspy.utils.exceptions import AdapterParseError

    trajectories_committed = len(_trajectories_from(source))
    if isinstance(err, AdapterParseError):
        print(
            "error: the LM returned a response DSPy could not parse into the ReAct schema.\n"
            f"  Detail: {err}\n"
            f"  Committed decisions before failure: {trajectories_committed}\n"
            f"  Elapsed: {elapsed:.1f}s\n"
            "  This is usually a small model stuck in a repetition loop. Try:\n"
            f"    --temperature 1.0   (currently {args.temperature})\n"
            f"    --max-tokens {max(args.max_tokens, 32000)}\n"
            f"    --reasoning   (currently {'on' if args.reasoning else 'off'}; off keeps output small)\n"
            "    or a stronger --model (e.g. anthropic/claude-haiku-4.5).",
            file=sys.stderr,
        )
        return 3
    if isinstance(err, RuntimeError) and "without a committed game action" in str(err):
        print(
            "error: an agent ran out of ReAct iterations without committing a terminal tool.\n"
            f"  Detail: {err}\n"
            f"  Committed decisions before failure: {trajectories_committed}\n"
            f"  Elapsed: {elapsed:.1f}s\n"
            "  The model used cognitive tools but never called the terminal one. Try:\n"
            f"    --max-iters {args.max_iters * 2}   (currently {args.max_iters})\n"
            "    or a stronger --model.",
            file=sys.stderr,
        )
        return 3
    raise err


def _final_position(stream: EventStream) -> tuple[int, str]:
    """Return the `(round, phase.value)` of the last logged event.

    Used by the summary panel to say "ended round 3 day". The last
    event is always `GAME_OVER` in a successful run.
    """
    last = stream.log.events[-1]
    return last.round, last.phase.value


if __name__ == "__main__":
    main()
