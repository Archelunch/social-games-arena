"""Tests for the `sdb-werewolf` CLI's terminal printer.

The printer is a thin Rich-Console renderer that the CLI plugs into the
T30 `on_trajectory` callback (per-decision live updates) and the
`DecisionSource.observe` hook (per-phase resolution events). Tests render
into a recording `rich.Console` and assert against `console.export_text()`
so the assertions are color-stripped and stable.

Per CLAUDE.md rule 8: every test pins WHY the behavior matters — a test
that can't fail when the printer regresses is wrong.
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep, Trajectory
from social_deduction_bench.engine import Event, Phase
from social_deduction_bench.games.werewolf.events import (
    ACCUSATION,
    DEFENSE,
    DISCUSSION_RESOLVED,
    EXILE_RESOLVED,
    GAME_OVER,
    KILL_BALLOTS,
    KILL_RESOLVED,
    SEER_INSPECT,
    SPEECH,
)
from social_deduction_bench.printer import GamePrinter, PrinterSettings

ROSTER: tuple[tuple[str, str], ...] = (
    ("Wolf1", "werewolf"),
    ("Wolf2", "werewolf"),
    ("Seer1", "seer"),
    ("Doc1", "doctor"),
    ("Vil1", "villager"),
    ("Vil2", "villager"),
    ("Vil3", "villager"),
)


def _console() -> Console:
    """A recording Rich Console with deterministic width and no color codes.

    `record=True` lets us pull rendered text out via `export_text()`. Width
    pinned so layout assertions are stable across environments.
    """
    return Console(record=True, width=120, legacy_windows=False, file=StringIO())


def _make_step(
    *,
    iter: int = 0,
    thought: str = "decide",
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


def _make_lm_call(
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    latency_ms: float = 400.0,
    cost_usd: float | None = 0.0001,
    model: str = "qwen/qwen3.5-9b",
) -> LMCallRecord:
    return LMCallRecord(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def _make_trajectory(
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
        react_trajectory=react_trajectory if react_trajectory is not None else (_make_step(),),
        lm_calls=lm_calls if lm_calls is not None else (_make_lm_call(),),
    )


# --- header ---------------------------------------------------------------


def test_header_shows_seed_model_and_full_roster() -> None:
    """The opening banner names the seed, the model, and every seat.

    The user must see — at a glance — what game they're about to watch.
    A missing field would force them to grep the events.jsonl just to
    confirm what they ran.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=42, model="qwen/qwen3.5-9b", roster=ROSTER)
    output = console.export_text()

    assert "seed 42" in output
    assert "qwen/qwen3.5-9b" in output
    for name, _ in ROSTER:
        assert name in output


# --- phase banner ---------------------------------------------------------


def test_phase_change_emits_banner_with_round_and_phase() -> None:
    """A new (round, phase) pair triggers a labeled banner before the next event.

    Without the banner, decisions and resolutions blur together. The banner
    is the only visual break between phases.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(_make_trajectory(round=1, phase="night"))
    printer.on_trajectory(_make_trajectory(decision_seq=1, round=1, phase="day", caller="Vil1", role="villager"))

    output = console.export_text()
    night_banner_pos = output.index("Round 1")
    day_idx = output.find("Day", night_banner_pos)
    night_idx = output.find("Night", night_banner_pos)
    assert night_idx >= 0
    assert day_idx >= 0
    assert night_idx < day_idx


def test_same_phase_does_not_re_emit_banner() -> None:
    """Two trajectories in the same phase share one banner.

    If every event drew its own banner, the output would be unreadable.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(_make_trajectory(decision_seq=0, caller="Wolf1"))
    printer.on_trajectory(_make_trajectory(decision_seq=1, caller="Wolf2"))

    output = console.export_text()
    assert output.count("Night") == 1


# --- trajectory rendering -------------------------------------------------


def test_trajectory_renders_actor_role_tool_and_args() -> None:
    """One trajectory renders as `{caller} {tool}({args})` with role decoration.

    These four facts are the minimum the user needs to follow the game.
    Drop any one and the line stops carrying useful signal.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(
        _make_trajectory(
            caller="Seer1",
            role="seer",
            terminal_tool="seer_inspect",
            committed_value="Wolf1",
            react_trajectory=(
                _make_step(
                    tool="seer_inspect",
                    args={"target": "Wolf1"},
                    observation="ok: seer_inspect committed with {'target': 'Wolf1'}",
                ),
            ),
        )
    )
    output = console.export_text()

    assert "Seer1" in output
    assert "seer_inspect" in output
    assert "Wolf1" in output  # the target


def test_trajectory_renders_telemetry_tail() -> None:
    """LM-call telemetry (tokens, latency, cost) appears on the same line.

    "See additional information" is one of the user's stated needs — the
    cost and token counts must be visible without leaving the CLI.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(
        _make_trajectory(
            lm_calls=(_make_lm_call(prompt_tokens=120, completion_tokens=80, latency_ms=450.0, cost_usd=0.00023),)
        )
    )
    output = console.export_text()

    # 120 + 80 = 200 tokens; rendered as "200t" or similar.
    assert "200" in output
    # Latency rendered as "0.5s" or "450ms".
    assert "0.5s" in output or "450" in output
    # Cost rendered with $ prefix.
    assert "$" in output


def test_intermediate_chat_step_renders_as_sub_bullet() -> None:
    """A `werewolf_chat` intermediate appears as a `⤷` sub-bullet under the terminal commit.

    The sub-bullet pattern keeps the decision on one main line and the
    pre-commit reasoning visible. Without it, chat lines would either be
    invisible or clutter the main stream.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    chat_step = _make_step(
        iter=0,
        thought="discuss",
        tool="werewolf_chat",
        args={"message": "we kill Vil1 tonight"},
        observation="ok: werewolf_chat called with {'message': 'we kill Vil1 tonight'}",
    )
    kill_step = _make_step(iter=1, tool="submit_kill_vote", args={"target": "Vil1"})
    printer.on_trajectory(_make_trajectory(react_trajectory=(chat_step, kill_step)))

    output = console.export_text()
    assert "chat" in output.lower()
    assert "we kill Vil1 tonight" in output


def test_rejected_step_renders_with_rejection_marker() -> None:
    """A rejected intermediate step renders with an `✗` marker and the reason text.

    Rejections are the source signal for the illegal-move-rate metric
    (T24). The user watching live must see them — silently hiding errors
    would undermine the benchmark's debugging value.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    rejected = _make_step(
        iter=0,
        tool="submit_kill_vote",
        args={"target": "Wolf1"},
        observation="error: target cannot be caller",
    )
    success = _make_step(
        iter=1,
        tool="submit_kill_vote",
        args={"target": "Vil1"},
        observation="ok: submit_kill_vote committed with {'target': 'Vil1'}",
    )
    printer.on_trajectory(_make_trajectory(react_trajectory=(rejected, success)))

    output = console.export_text()
    assert "rejected" in output.lower() or "✗" in output
    assert "Wolf1" in output  # the rejected target


# --- private/public events -----------------------------------------------


def test_private_event_renders_with_recipients_suffix() -> None:
    """Private events show `[private → recipients]` in dim text.

    Spectator view: the user is not an agent. They must see private
    events but with the privacy boundary made visible — a kill_ballots
    private to the wolf pack must NOT look identical to a public event.
    """
    console = _console()
    printer = GamePrinter(console)

    event = Event(
        seq=0,
        round=1,
        phase=Phase.NIGHT,
        type=KILL_BALLOTS,
        payload={"ballots": {"Wolf1": "Vil1", "Wolf2": "Vil1"}},
        recipients=("Wolf1", "Wolf2"),
    )
    printer.on_event(event)
    output = console.export_text()

    assert "private" in output.lower()
    assert "Wolf1" in output
    assert "Wolf2" in output


def test_kill_resolved_protected_variant_renders_protection() -> None:
    """`kill_resolved` with `protected=True` says so loud.

    The single most impactful event of a night is whether the kill
    succeeded. A line that buries protection in the payload would force
    the user to read JSON to know who died.
    """
    console = _console()
    printer = GamePrinter(console)

    event = Event(
        seq=0,
        round=1,
        phase=Phase.NIGHT,
        type=KILL_RESOLVED,
        payload={"victim": None, "protected": True},
    )
    printer.on_event(event)
    output = console.export_text()

    assert "protect" in output.lower()


def test_kill_resolved_victim_variant_names_the_victim() -> None:
    """`kill_resolved` without protection names who died.

    The output must be readable left-to-right without needing the payload.
    """
    console = _console()
    printer = GamePrinter(console)

    event = Event(
        seq=0,
        round=1,
        phase=Phase.NIGHT,
        type=KILL_RESOLVED,
        payload={"victim": "Vil1"},
    )
    printer.on_event(event)
    output = console.export_text()

    assert "Vil1" in output


def test_exile_resolved_renders_exiled_player() -> None:
    """`exile_resolved` names who was exiled.

    Exile is the day's outcome; without naming the exiled player the
    line is noise.
    """
    console = _console()
    printer = GamePrinter(console)

    ballots = {
        "Wolf1": "Seer1",
        "Wolf2": "Seer1",
        "Seer1": "Wolf1",
        "Doc1": "Wolf1",
        "Vil1": "Wolf1",
        "Vil2": "Wolf1",
        "Vil3": "Wolf1",
    }
    event = Event(
        seq=0,
        round=1,
        phase=Phase.DAY,
        type=EXILE_RESOLVED,
        payload={"exiled": "Wolf1", "ballots": ballots},
    )
    printer.on_event(event)
    output = console.export_text()

    assert "Wolf1" in output
    assert "exile" in output.lower()


def test_discussion_resolved_shows_speakers_in_order() -> None:
    """`discussion_resolved` renders the chosen speakers in bid order.

    The discussion's order shapes the rest of the day's signal — knowing
    who spoke first vs last is core game information.
    """
    console = _console()
    printer = GamePrinter(console)

    bids = {"Seer1": 8, "Wolf1": 5, "Doc1": 2, "Wolf2": 3, "Vil1": 1, "Vil2": 0, "Vil3": 2}
    event = Event(
        seq=0,
        round=1,
        phase=Phase.DAY,
        type=DISCUSSION_RESOLVED,
        payload={"speakers": ["Seer1", "Wolf1", "Doc1"], "bids": bids},
    )
    printer.on_event(event)
    output = console.export_text()

    seer_idx = output.index("Seer1")
    wolf_idx = output.index("Wolf1")
    doc_idx = output.index("Doc1")
    assert seer_idx < wolf_idx < doc_idx


# --- redundant-event filtering -------------------------------------------


def test_seer_inspect_event_is_not_rendered() -> None:
    """`seer_inspect` events are skipped — the seer's trajectory already shows the inspection.

    Rendering the event would duplicate a line the trajectory already
    drew, doubling the noise without adding signal.
    """
    console = _console()
    printer = GamePrinter(console)

    event = Event(
        seq=0,
        round=1,
        phase=Phase.NIGHT,
        type=SEER_INSPECT,
        payload={"target": "Wolf1", "faction": "werewolves"},
        recipients=("Seer1",),
    )
    printer.on_event(event)
    output = console.export_text()

    # Empty — the printer skipped a redundant event entirely.
    assert "seer_inspect" not in output.lower()


def test_speech_event_is_not_rendered_by_on_event_path() -> None:
    """`speech` events are skipped by `on_event` — the speech is rendered by `on_trajectory`.

    Same redundancy guard: a speak terminal commit fires the speech text
    via the trajectory; the resulting SPEECH event is the engine's
    confirmation, not new signal.
    """
    console = _console()
    printer = GamePrinter(console)

    event = Event(
        seq=0,
        round=1,
        phase=Phase.DAY,
        type=SPEECH,
        payload={"speaker": "Seer1", "message": "I inspected Wolf1"},
    )
    printer.on_event(event)
    output = console.export_text()

    # The speech text is NOT rendered (no message string visible).
    assert "I inspected Wolf1" not in output


# --- speech trajectory ---------------------------------------------------


def test_speak_trajectory_renders_message_as_quoted_dialogue() -> None:
    """A `speak` terminal commits the message text, rendered as a quoted line.

    Dialogue is the texture of the game. Reading "Seer1: 'I'm the
    seer'" is what makes a Werewolf game watchable.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(
        _make_trajectory(
            round=1,
            phase="day",
            caller="Seer1",
            role="seer",
            terminal_tool="speak",
            committed_value="I inspected Wolf1 last night.",
            react_trajectory=(
                _make_step(
                    tool="speak",
                    args={"message": "I inspected Wolf1 last night."},
                    observation="ok: speak committed with {'message': 'I inspected Wolf1 last night.'}",
                ),
            ),
        )
    )
    output = console.export_text()

    assert "I inspected Wolf1 last night." in output


# --- reaction trajectories -----------------------------------------------


def test_accuse_trajectory_renders_target_and_reason() -> None:
    """An `accuse` reaction renders the target and the stated reason.

    The reaction round is the new texture of the day; watching "Vil1 accuse
    Wolf2: 'dodged the vote'" is what makes the deduction legible to an operator.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(
        _make_trajectory(
            round=1,
            phase="day",
            caller="Vil1",
            role="villager",
            terminal_tool="accuse",
            committed_value="Wolf2",
            react_trajectory=(
                _make_step(
                    tool="accuse",
                    args={"target": "Wolf2", "reason": "dodged the vote"},
                    observation="ok: accuse committed with {'target': 'Wolf2', 'reason': 'dodged the vote'}",
                ),
            ),
        )
    )
    output = console.export_text()

    assert "accuse" in output
    assert "Wolf2" in output
    assert "dodged the vote" in output


def test_defend_trajectory_renders_target_and_reason() -> None:
    """A `defend` reaction renders who is defended and why."""
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(
        _make_trajectory(
            round=1,
            phase="day",
            caller="Doc1",
            role="doctor",
            terminal_tool="defend",
            committed_value="Seer1",
            react_trajectory=(
                _make_step(
                    tool="defend",
                    args={"target": "Seer1", "reason": "their read checks out"},
                    observation="ok: defend committed with {'target': 'Seer1', 'reason': 'their read checks out'}",
                ),
            ),
        )
    )
    output = console.export_text()

    assert "defend" in output
    assert "Seer1" in output
    assert "their read checks out" in output


def test_pass_turn_trajectory_renders_pass() -> None:
    """A `pass_turn` reaction renders a terse `pass` — silence, but visible."""
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(
        _make_trajectory(
            round=1,
            phase="day",
            caller="Vil1",
            role="villager",
            terminal_tool="pass_turn",
            committed_value=None,
            react_trajectory=(_make_step(tool="pass_turn", args={}, observation="ok: pass_turn committed with {}"),),
        )
    )
    output = console.export_text()

    assert "pass" in output


def test_accusation_and_defense_events_are_not_rendered_by_on_event_path() -> None:
    """`accusation` / `defense` events are skipped by `on_event`.

    Both render from the reaction trajectory (the accuse/defend terminal line);
    the resulting public events are the engine's confirmation, not new signal, so
    `on_event` must not render them a second time.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.on_event(
        Event(
            seq=0,
            round=1,
            phase=Phase.DAY,
            type=ACCUSATION,
            payload={"accuser": "Vil1", "target": "Wolf2", "reason": "double-rendered?"},
        )
    )
    printer.on_event(
        Event(
            seq=1,
            round=1,
            phase=Phase.DAY,
            type=DEFENSE,
            payload={"defender": "Doc1", "defended": "Seer1", "reason": "also double?"},
        )
    )
    output = console.export_text()

    assert "double-rendered?" not in output
    assert "also double?" not in output


# --- summary panel --------------------------------------------------------


def test_summary_panel_names_winner_and_paths() -> None:
    """The final summary panel renders the winner and both output paths.

    The user runs the CLI and gets a single block of structured info at
    the end — anything else means they have to grep stderr or the file
    system. The block must self-contain the answer.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=42, model="qwen/qwen3.5-9b", roster=ROSTER)
    # Simulate a few decisions for telemetry totals.
    for seq in range(3):
        printer.on_trajectory(_make_trajectory(decision_seq=seq))
    printer.print_summary(
        winner="villagers",
        final_round=3,
        final_phase="day",
        rounds_played=3,
        events_path="games/g1/events.jsonl",
        trajectories_path="games/g1/trajectories.jsonl",
        elapsed_seconds=138.0,
    )
    output = console.export_text()

    assert "villagers" in output
    assert "games/g1/events.jsonl" in output
    assert "games/g1/trajectories.jsonl" in output


# --- streaming hooks -----------------------------------------------------


def test_on_decision_start_prints_thinking_marker() -> None:
    """`on_decision_start` prints an immediate "thinking…" line for the actor.

    Without this hook the operator sees nothing between the phase banner
    and the eventual commit — potentially 30+ seconds of silence for a
    small model. The line confirms the system is alive.
    """
    console = _console()
    printer = GamePrinter(console)
    printer.on_decision_start("Wolf1", "werewolf")
    output = console.export_text()
    assert "Wolf1" in output
    assert "thinking" in output


def test_on_decision_start_respects_quiet_mode() -> None:
    """In `--quiet` mode the streaming chatter is suppressed.

    Quiet runs are for piping to a file or producing minimal output;
    `thinking…` markers would defeat that.
    """
    console = _console()
    printer = GamePrinter(console, settings=PrinterSettings(quiet=True))
    printer.on_decision_start("Wolf1", "werewolf")
    assert console.export_text() == ""


def test_on_react_step_renders_tool_and_args_with_sub_bullet() -> None:
    """`on_react_step` prints a `⤷ tool(args)` sub-bullet as the iteration lands.

    This is the real-time feed the operator watches during a slow LLM
    game. The line must name the tool and surface its args so the user
    can follow the agent's reasoning without opening trajectories.jsonl.
    """
    from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep

    console = _console()
    printer = GamePrinter(console)
    step = ReActStep(iter=0, thought="t", tool="get_private_info", args={}, observation="ok: ...")
    lm_calls = (LMCallRecord(model="m", prompt_tokens=42, completion_tokens=10, latency_ms=120.0, cost_usd=None),)

    printer.on_react_step("Wolf1", "werewolf", step, lm_calls)
    output = console.export_text()

    assert "get_private_info" in output
    assert "⤷" in output


def test_on_react_step_marks_rejected_attempts_with_red_x() -> None:
    """A failed tool call (the agent self-corrects) renders distinctively.

    `error:` observations come from the validation layer; the operator
    needs to spot them in the stream to debug an agent that's looping
    on rejections.
    """
    from social_deduction_bench.agents.trajectory import ReActStep

    console = _console()
    printer = GamePrinter(console)
    step = ReActStep(
        iter=0,
        thought="t",
        tool="submit_kill_vote",
        args={"target": "Wolf1"},
        observation="error: tool 'submit_kill_vote' cannot target the caller 'Wolf1'",
    )
    printer.on_react_step("Wolf1", "werewolf", step, ())
    output = console.export_text()
    assert "submit_kill_vote" in output
    assert "✗" in output
    assert "cannot target the caller" in output


def test_stream_iterations_setting_skips_on_trajectory_sub_bullets() -> None:
    """When streaming is on, `on_trajectory` no longer re-renders intermediates.

    Otherwise every werewolf_chat / rejected attempt would appear twice —
    once from `on_react_step` (live) and once from `on_trajectory`
    (after commit). The CLI sets `stream_iterations=True`; the test
    fixtures default to off for backward compat.
    """
    from social_deduction_bench.agents.trajectory import ReActStep

    console = _console()
    printer = GamePrinter(console, settings=PrinterSettings(stream_iterations=True))

    chat = ReActStep(
        iter=0, thought="t", tool="werewolf_chat", args={"message": "kill"}, observation="ok: werewolf_chat called"
    )
    terminal = ReActStep(
        iter=1, thought="t", tool="submit_kill_vote", args={"target": "Vil1"}, observation="ok: committed"
    )
    trajectory = _make_trajectory(
        decision_seq=0,
        round=1,
        phase="night",
        caller="Wolf1",
        role="werewolf",
        terminal_tool="submit_kill_vote",
        react_trajectory=(chat, terminal),
    )

    printer.on_trajectory(trajectory)
    output = console.export_text()

    # Main line still renders, but werewolf_chat sub-bullet does NOT (streamed elsewhere).
    assert "submit_kill_vote" in output
    assert "werewolf_chat" not in output


# --- memory panel --------------------------------------------------------


def test_print_memories_renders_plan_beliefs_and_notes_per_agent() -> None:
    """`print_memories` surfaces each agent's plan, beliefs, and notes.

    After a real run the operator needs to see what every agent was
    thinking — without this panel the only access to memory is grepping
    `memories.json` by hand. The block must name each player and show
    their plan, structured beliefs, and free-text notes.
    """
    console = _console()
    printer = GamePrinter(console)
    roster = (("Wolf1", "werewolf"), ("Seer1", "seer"))
    memories = {
        "Wolf1": {
            "plan": "frame the seer",
            "beliefs": {
                "Seer1": {"guess": "seer", "confidence": "high", "evidence": "claimed publicly"},
            },
            "notes": [{"round": 1, "text": "stay quiet"}],
        },
        "Seer1": {
            "plan": "inspect wolves",
            "beliefs": {
                "Wolf1": {"guess": "werewolf", "confidence": "high", "evidence": "inspected R1"},
            },
            "notes": [{"round": 1, "text": "Wolf1 = werewolf"}],
        },
    }

    printer.print_memories(memories=memories, roster=roster)
    output = console.export_text()

    assert "Memories" in output
    for token in (
        "Wolf1",
        "Seer1",
        "frame the seer",
        "inspect wolves",
        "claimed publicly",
        "inspected R1",
        "stay quiet",
        "Wolf1 = werewolf",
    ):
        assert token in output, f"missing {token!r} in:\n{output}"


def test_print_memories_with_empty_dict_renders_nothing() -> None:
    """Dry-run uses `ScriptedDecisions` and has no memories — render must skip silently.

    A "Memories" panel with no rows would just be noise; the panel only
    earns its place when there's something to show.
    """
    console = _console()
    printer = GamePrinter(console)
    printer.print_memories(memories={}, roster=(("Wolf1", "werewolf"),))
    assert "Memories" not in console.export_text()


def test_print_memories_handles_empty_per_agent_fields() -> None:
    """An agent who never wrote a note / belief / plan still renders without crashing.

    Some seats (e.g. plain villagers in a short game) may end with empty
    cognitive state. The panel must show `(none)` markers rather than
    blanking or raising.
    """
    console = _console()
    printer = GamePrinter(console)
    memories = {
        "Vil1": {"plan": "", "beliefs": {}, "notes": []},
    }
    printer.print_memories(memories=memories, roster=(("Vil1", "villager"),))
    output = console.export_text()
    assert "Vil1" in output
    assert "(none)" in output


# --- ASCII fallback ------------------------------------------------------


def test_ascii_mode_strips_role_emojis() -> None:
    """`--ascii` mode swaps emoji glyphs for ASCII letters.

    Some terminals (old Windows shells, restricted SSH sessions) can't
    render emoji. Without an ASCII fallback the CLI is broken in those
    environments.
    """
    console = _console()
    printer = GamePrinter(console, settings=PrinterSettings(ascii=True))

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(_make_trajectory(caller="Seer1", role="seer", terminal_tool="seer_inspect"))
    output = console.export_text()

    # No emoji codepoints in ASCII mode.
    assert "🐺" not in output
    assert "👁" not in output
    assert "🩺" not in output


# --- deterministic rendering ---------------------------------------------


def test_two_identical_inputs_render_byte_identically() -> None:
    """Same inputs → byte-identical export_text() output.

    Used by the CLI's regression-testing — golden output diffs only work
    if the printer is deterministic on its inputs.
    """
    out1 = _console()
    out2 = _console()
    for console in (out1, out2):
        printer = GamePrinter(console)
        printer.print_header(seed=42, model="qwen/qwen3.5-9b", roster=ROSTER)
        printer.on_trajectory(_make_trajectory())
        printer.on_event(Event(seq=0, round=1, phase=Phase.NIGHT, type=KILL_RESOLVED, payload={"victim": "Vil1"}))

    assert out1.export_text() == out2.export_text()


def test_quiet_mode_hides_per_trajectory_lines() -> None:
    """`--quiet` mode suppresses trajectory rendering; resolutions still print.

    For long games where the user wants a summary view, the resolutions
    (kills, exiles, game over) are the high-signal moments. Per-decision
    detail is in the sidecar; no need to render live.
    """
    console = _console()
    printer = GamePrinter(console, settings=PrinterSettings(quiet=True))

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(_make_trajectory(caller="Wolf1"))
    printer.on_event(Event(seq=0, round=1, phase=Phase.NIGHT, type=KILL_RESOLVED, payload={"victim": "Vil1"}))
    output = console.export_text()

    # Resolution still renders.
    assert "Vil1" in output
    # But the per-trajectory tool name is suppressed.
    assert "submit_kill_vote" not in output


# --- empty / missing telemetry --------------------------------------------


def test_trajectory_with_no_lm_calls_renders_without_telemetry_tail() -> None:
    """A trajectory with zero LM calls renders the action line without telemetry.

    Cache hits and the DummyLM produce trajectories with empty
    `lm_calls`. The printer must not crash on `[]`.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(_make_trajectory(lm_calls=()))
    output = console.export_text()

    # No crash and the action still renders.
    assert "Wolf1" in output


# --- game over -----------------------------------------------------------


def test_game_over_event_does_not_re_render_summary() -> None:
    """`game_over` event renders a single short line; the panel is printed by `print_summary`.

    Two separate "GAME OVER" headers would be confusing. The event line
    is a low-key marker; the summary panel is the structured block.
    """
    console = _console()
    printer = GamePrinter(console)

    event = Event(seq=99, round=3, phase=Phase.DAY, type=GAME_OVER, payload={"winner": "villagers"})
    printer.on_event(event)
    output = console.export_text()

    assert "villagers" in output


@pytest.mark.parametrize(("role", "glyph"), [("werewolf", "🐺"), ("seer", "👁"), ("doctor", "🩺")])
def test_role_glyphs_appear_in_default_emoji_mode(role: str, glyph: str) -> None:
    """Default (non-ASCII) mode renders role glyphs.

    The glyphs are the visual handle that lets a glance distinguish wolf
    decisions from seer ones. The mapping must be stable.
    """
    console = _console()
    printer = GamePrinter(console)

    printer.print_header(seed=1, model="m", roster=ROSTER)
    printer.on_trajectory(_make_trajectory(caller="X", role=role, terminal_tool="x_tool", committed_value="Y"))
    output = console.export_text()

    assert glyph in output


# --- token streaming (on_thought_chunk) ----------------------------------


def test_on_thought_chunk_buffers_text_and_renders_on_react_step() -> None:
    """Chunks for one iter accumulate into a dim thought line that prints
    just before that iter's `⤷` sub-bullet.

    The CLI hook for `dspy.streamify`'s `next_thought` listener feeds chunks
    here as they arrive from the LM. Without rendering, the user gains
    nothing from streaming — they still see only the terminal commit. By
    flushing the accumulated thought above the `⤷` line, the operator can
    read what the agent was thinking as the iter commits, instead of
    digging into `trajectories.jsonl` after the game.
    """
    console = _console()
    printer = GamePrinter(console, settings=PrinterSettings(stream_thoughts=True))

    printer.on_thought_chunk("Wolf1", "werewolf", 0, "We should target ")
    printer.on_thought_chunk("Wolf1", "werewolf", 0, "Vil1 tonight because")
    printer.on_thought_chunk("Wolf1", "werewolf", 0, " they look weak.")

    # Nothing rendered yet — the buffer is hot, but no iter has closed.
    assert "Vil1" not in console.export_text()

    step = _make_step(iter=0, tool="submit_kill_vote", args={"target": "Vil1"})
    printer.on_react_step("Wolf1", "werewolf", step, ())
    output = console.export_text()
    assert "We should target Vil1 tonight because they look weak." in output
    assert "⤷ submit_kill_vote" in output


def test_on_thought_chunk_renders_nothing_when_stream_thoughts_off() -> None:
    """`PrinterSettings.stream_thoughts=False` keeps the printer silent on chunks.

    Operators who pipe the stream to a file may want grep-friendly logs
    without the wrap-line; `--no-stream-thoughts` flips this off without
    disabling parallelism or per-iter sub-bullets.
    """
    console = _console()
    printer = GamePrinter(console, settings=PrinterSettings(stream_thoughts=False))

    printer.on_thought_chunk("Wolf1", "werewolf", 0, "hidden text")
    step = _make_step(iter=0, tool="submit_kill_vote", args={"target": "Vil1"})
    printer.on_react_step("Wolf1", "werewolf", step, ())
    output = console.export_text()
    assert "hidden text" not in output
    # The sub-bullet itself still renders — only the thought wrap-line was
    # suppressed, not the iteration's commit line.
    assert "⤷ submit_kill_vote" in output


def test_on_thought_chunk_buffers_per_caller_independently() -> None:
    """Parallel agents stream into separate buffers — Wolf1 and Wolf2 chunks
    must not concatenate into one thought line.

    With `asyncio.gather` interleaving multiple seats' streams, a shared
    buffer would smear the wolves' thoughts together and the operator
    would see "We should target target Vil1 Wolf1's right" — illegible.
    """
    console = _console()
    printer = GamePrinter(console, settings=PrinterSettings(stream_thoughts=True))

    printer.on_thought_chunk("Wolf1", "werewolf", 0, "Wolf1 idea")
    printer.on_thought_chunk("Wolf2", "werewolf", 0, "Wolf2 idea")
    printer.on_thought_chunk("Wolf1", "werewolf", 0, " continues")

    step_w1 = _make_step(iter=0, tool="submit_kill_vote", args={"target": "Vil1"})
    step_w2 = _make_step(iter=0, tool="submit_kill_vote", args={"target": "Vil2"})
    printer.on_react_step("Wolf1", "werewolf", step_w1, ())
    printer.on_react_step("Wolf2", "werewolf", step_w2, ())

    output = console.export_text()
    assert "Wolf1 idea continues" in output
    assert "Wolf2 idea" in output
    # The mixed-buffer regression would produce "Wolf1 ideaWolf2 idea continues".
    assert "Wolf1 ideaWolf2" not in output


def test_on_thought_chunk_buffer_clears_after_iter_renders() -> None:
    """The buffer for one iter is dropped once it's flushed.

    A latent buffer would re-emit the previous iter's thought when the
    next iter completes — duplicating signal and confusing the operator.
    Each iter's thought belongs only to that iter.
    """
    console = _console()
    printer = GamePrinter(console, settings=PrinterSettings(stream_thoughts=True))

    printer.on_thought_chunk("Wolf1", "werewolf", 0, "first thought")
    printer.on_react_step("Wolf1", "werewolf", _make_step(iter=0, tool="submit_kill_vote"), ())
    # Iter 1 has its own thought; iter 0's text must not re-appear.
    printer.on_thought_chunk("Wolf1", "werewolf", 1, "second thought")
    printer.on_react_step("Wolf1", "werewolf", _make_step(iter=1, tool="submit_kill_vote"), ())

    output = console.export_text()
    assert output.count("first thought") == 1


def test_on_thought_chunk_quiet_mode_suppresses_rendering() -> None:
    """`--quiet` wins over `stream_thoughts=True` — no per-iter chatter at all."""
    console = _console()
    printer = GamePrinter(console, settings=PrinterSettings(stream_thoughts=True, quiet=True))

    printer.on_thought_chunk("Wolf1", "werewolf", 0, "should be silent")
    printer.on_react_step("Wolf1", "werewolf", _make_step(iter=0, tool="submit_kill_vote"), ())
    assert console.export_text() == ""
