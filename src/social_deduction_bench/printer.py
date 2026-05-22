"""Rich-Console renderer for the `sdb-werewolf` CLI.

Streams a Werewolf game to the terminal in real time. Two ingress points:

- `on_trajectory(trajectory)`: fires once per committed decision (T30
  sidecar), via the `ReActDecisionSource.on_trajectory` callback. Renders
  one main line per decision with role icon, terminal tool, args, and a
  dim telemetry tail. Intermediate steps (werewolf chats, rejected
  attempts) render as `⤷` sub-bullets.
- `on_event(event)`: fires for resolver events through the CLI's
  `_PrintingDecisionSource.observe` hook. Renders only resolutions
  (`kill_resolved`, `exile_resolved`, `discussion_resolved`,
  `kill_ballots`, `game_over`) — decision-events (`bid`, `speech`,
  `werewolf_chat`, `seer_inspect`, `doctor_protect`, `tool_rejected`)
  are redundant with what trajectories already rendered.

Design choices: streaming output (scrollable, copy-paste-friendly),
spectator view (private events visible with a `[private → recipients]`
suffix), and decision-line trace depth (full per-iteration thoughts stay
in `trajectories.jsonl`). See `~/.claude/plans/t30-silly-creek.md` for
the locked-in shape brief.

Tests live in `tests/test_printer.py` and render against a recording
`Console(record=True)` so assertions read `export_text()` (color-stripped,
stable across environments).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep, Trajectory
from social_deduction_bench.engine import Event
from social_deduction_bench.games.werewolf.events import (
    ACCUSATION,
    BID,
    DEFENSE,
    DISCUSSION_RESOLVED,
    EXILE_RESOLVED,
    GAME_OVER,
    KILL_BALLOTS,
    KILL_RESOLVED,
    SEER_INSPECT,
    SPEECH,
    TOOL_REJECTED,
    WEREWOLF_CHAT,
)

_ROLE_GLYPHS_EMOJI: Mapping[str, str] = {
    "werewolf": "🐺",
    "seer": "👁 ",
    "doctor": "🩺",
    "villager": "  ",
}
_ROLE_GLYPHS_ASCII: Mapping[str, str] = {
    "werewolf": "W ",
    "seer": "S ",
    "doctor": "D ",
    "villager": "V ",
}
_ROLE_COLORS: Mapping[str, str] = {
    "werewolf": "red",
    "seer": "magenta",
    "doctor": "green",
    "villager": "cyan",
}

# Decision-events the trajectory layer already renders; skip them on `on_event`.
# Accusations and defenses render from the reaction trajectory (the accuse/defend
# terminal line), so they join the redundant set rather than rendering twice.
_REDUNDANT_EVENT_TYPES = frozenset(
    {SEER_INSPECT, BID, SPEECH, WEREWOLF_CHAT, TOOL_REJECTED, ACCUSATION, DEFENSE, "doctor_protect"}
)


@dataclass(frozen=True, slots=True)
class PrinterSettings:
    """Knobs for the CLI's printer.

    `ascii`: render ASCII glyphs only — useful in terminals that can't
    show emoji. `quiet`: suppress per-trajectory rendering; phase
    banners and resolutions still print. `stream_iterations`: render
    each ReAct iteration as it lands (via `on_react_step`) instead of
    waiting for the trajectory commit; when on, `on_trajectory` skips
    its own sub-bullet rendering to avoid duplicates.
    `stream_thoughts`: when on, `on_thought_chunk` buffers token chunks
    from the LM's `next_thought` stream per (caller, iter) and flushes
    them as a dim `💭 thought` wrap-line just above each iter's
    `⤷ tool(args)` sub-bullet. When off, the CLI keeps parallelism +
    sub-bullets but skips the thought wrap — used by `--no-stream-thoughts`
    for grep-friendly logs.
    """

    ascii: bool = False
    quiet: bool = False
    stream_iterations: bool = False
    stream_thoughts: bool = True


@dataclass
class _Totals:
    """Running cost / token / call counters for the closing summary panel."""

    decisions: int = 0
    lm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class GamePrinter:
    """Stateful Rich-Console renderer driven by the CLI.

    The CLI wires `on_trajectory` and `on_event` to the matching adapter
    hooks. The printer tracks the current `(round, phase)` so phase
    banners fire exactly once per transition.
    """

    console: Console
    settings: PrinterSettings = field(default_factory=PrinterSettings)
    _current_phase: tuple[int, str] | None = field(default=None, init=False)
    _totals: _Totals = field(default_factory=_Totals, init=False)
    _thought_buffers: dict[tuple[str, int], list[str]] = field(default_factory=dict, init=False)

    # --- public API ------------------------------------------------------

    def print_header(self, *, seed: int, model: str, roster: Sequence[tuple[str, str]]) -> None:
        """Render the opening banner: seed, model, and the full roster."""
        title = Text(f"Werewolf · seed {seed} · model {model} · {len(roster)} seats", style="bold")
        seats = Text()
        for idx, (name, role) in enumerate(roster):
            if idx:
                seats.append("  ")
            seats.append(self._role_chip(name, role))
        self.console.print(Panel.fit(Text.assemble(title, "\n", seats), border_style="dim", padding=(0, 2)))
        self.console.print()

    def on_trajectory(self, trajectory: Trajectory) -> None:
        """Render one decision: actor, intermediates (sub-bullets), terminal, telemetry."""
        self._totals.decisions += 1
        for call in trajectory.lm_calls:
            self._totals.lm_calls += 1
            self._totals.prompt_tokens += call.prompt_tokens
            self._totals.completion_tokens += call.completion_tokens
            if call.cost_usd is not None:
                self._totals.cost_usd += call.cost_usd

        if self.settings.quiet:
            return

        self._maybe_phase_banner(trajectory.round, trajectory.phase)

        chip = self._role_chip(trajectory.caller, trajectory.role)
        if not self.settings.stream_iterations:
            # Sub-bullets for intermediates (werewolf_chat) and rejected attempts.
            # Skipped when `stream_iterations` is on — `on_react_step` already
            # rendered each iteration in real time, so re-rendering here would
            # duplicate lines.
            for step in trajectory.react_trajectory:
                if step.tool == trajectory.terminal_tool and step.observation.startswith("ok:"):
                    continue  # the terminal commit renders as the main line below
                if step.observation.startswith("error:"):
                    self._render_rejection_sub(step)
                elif step.tool == "werewolf_chat":
                    self._render_chat_sub(step)
                # silently skip cognitive tool calls — they're audit-only signal

        main_line = self._compose_decision_line(chip, trajectory)
        self.console.print(main_line)

    def on_decision_start(self, caller: str, role: str) -> None:
        """Announce a player's turn before the first LM call goes out.

        Without this hook the user sees nothing between the phase banner
        and the eventual commit — potentially 30+ seconds of silence for
        a small model. The dim "thinking…" line is the activity signal.
        """
        if self.settings.quiet:
            return
        chip = self._role_chip(caller, role)
        line = Text("  ")
        line.append(chip)
        line.append("  ")
        line.append("thinking…", style="dim italic")
        self.console.print(line)

    def on_thought_chunk(self, caller: str, role: str, iter_idx: int, text: str) -> None:
        """Accept one streamed chunk of `next_thought` for `(caller, iter_idx)`.

        Buffered per (caller, iter_idx) so concurrent agents don't smear
        their thoughts together. Flushed by `on_react_step` as a dim
        `💭 thought` line just above that iter's `⤷` sub-bullet. The
        buffer is dropped after flush — re-emitting a previous iter's
        thought on a later iter would duplicate signal.

        No-op when `quiet` or `stream_thoughts=False`; the chunks land in
        the trajectory sidecar regardless of this hook.
        """
        if self.settings.quiet or not self.settings.stream_thoughts:
            return
        _ = role  # the role chip is rendered when the buffer flushes, not now
        self._thought_buffers.setdefault((caller, iter_idx), []).append(text)

    def on_react_step(
        self,
        caller: str,
        role: str,
        step: ReActStep,
        lm_calls: tuple[LMCallRecord, ...],
    ) -> None:
        """Render one ReAct iteration as soon as it lands (real-time stream).

        Each iteration is one LM call: the committing terminal, or an
        intermediate (cognitive / werewolf_chat) or rejected attempt — all
        get a `⤷` sub-bullet. (There is no `finish` step: a terminal commit
        ends the loop.)

        If `on_thought_chunk` was called for this `(caller, iter)`, the
        accumulated thought text flushes as a dim wrap-line above the
        sub-bullet, then the buffer is cleared.
        """
        if self.settings.quiet:
            # Drain any leftover chunk buffer so it doesn't survive into a
            # later non-quiet phase (quiet mode could be toggled in theory).
            self._thought_buffers.pop((caller, step.iter), None)
            return
        self._flush_thought(caller, role, step.iter)
        is_error = step.observation.startswith("error:")
        sub = Text("    ⤷ ", style="dim")
        if is_error:
            sub.append("✗ ", style="red")
        sub.append(step.tool, style=_ROLE_COLORS.get(role, "default") if not is_error else "red")
        args_text = _format_args(step.args)
        if args_text:
            sub.append("(")
            sub.append(args_text, style="dim")
            sub.append(")")
        else:
            sub.append("()")
        tail = _iter_telemetry_tail(lm_calls)
        if tail.plain:
            sub.append("  ")
            sub.append(tail)
        self.console.print(sub)
        if is_error and step.observation:
            # Surface the rejection reason on its own dim wrapped line — that's
            # the signal the agent will read to self-correct, so the operator
            # should see it too.
            why = Text("        ", style="dim")
            why.append(step.observation, style="red dim")
            self.console.print(why)

    def on_event(self, event: Event) -> None:
        """Render resolver events; skip those redundant with trajectories."""
        if event.type in _REDUNDANT_EVENT_TYPES:
            return

        self._maybe_phase_banner(event.round, event.phase.value)

        if event.type == KILL_RESOLVED:
            self._render_kill_resolved(event)
        elif event.type == KILL_BALLOTS:
            self._render_kill_ballots(event)
        elif event.type == EXILE_RESOLVED:
            self._render_exile_resolved(event)
        elif event.type == DISCUSSION_RESOLVED:
            self._render_discussion_resolved(event)
        elif event.type == GAME_OVER:
            self._render_game_over(event)

    def print_summary(
        self,
        *,
        winner: str,
        final_round: int,
        final_phase: str,
        rounds_played: int,
        events_path: str,
        trajectories_path: str,
        memories_path: str | None = None,
        elapsed_seconds: float,
    ) -> None:
        """Render the closing summary panel with the structured stats."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", no_wrap=True)
        table.add_column()
        table.add_row("winner", Text(winner, style="bold green" if winner == "villagers" else "bold red"))
        table.add_row("ended", f"round {final_round} {final_phase}")
        table.add_row("rounds", str(rounds_played))
        table.add_row("decisions", str(self._totals.decisions))
        table.add_row("LM calls", str(self._totals.lm_calls))
        tokens = self._totals.prompt_tokens + self._totals.completion_tokens
        prompt = _fmt_tokens(self._totals.prompt_tokens)
        completion = _fmt_tokens(self._totals.completion_tokens)
        table.add_row("tokens", f"{_fmt_tokens(tokens)}  (in {prompt} · out {completion})")
        table.add_row("cost", f"${self._totals.cost_usd:.4f}")
        table.add_row("elapsed", _fmt_duration(elapsed_seconds))
        table.add_row("", "")
        table.add_row("events.jsonl", events_path)
        table.add_row("trajectories.jsonl", trajectories_path)
        if memories_path is not None:
            table.add_row("memories.json", memories_path)

        self.console.print()
        self.console.print(Panel(table, title="GAME OVER", border_style="bold", padding=(1, 2)))

    def print_memories(
        self,
        *,
        memories: Mapping[str, Mapping[str, object]],
        roster: Sequence[tuple[str, str]],
    ) -> None:
        """Render each agent's private memory (plan + beliefs + notes) after the game.

        `memories` is the per-agent slice from `GameMemory.to_json_dict()` —
        events are intentionally excluded (they live in `events.jsonl`).
        Skips silently when `memories` is empty (e.g. dry-run with
        `ScriptedDecisions`, which doesn't carry a memory).
        """
        if not memories:
            return
        role_by_name = {name: role for name, role in roster}

        body = Table.grid(padding=(0, 0))
        body.add_column()
        first = True
        for name, _role in roster:
            if name not in memories:
                continue
            if not first:
                body.add_row(Text(""))
            first = False
            body.add_row(self._compose_memory_block(name, role_by_name.get(name, ""), memories[name]))

        self.console.print()
        self.console.print(Panel(body, title="Memories", border_style="dim", padding=(1, 2)))

    # --- internal renderers ---------------------------------------------

    def _maybe_phase_banner(self, round_: int, phase: str) -> None:
        key = (round_, phase)
        if self._current_phase == key:
            return
        self._current_phase = key
        label = f"Round {round_} — {phase.capitalize()}"
        style = "yellow" if phase == "day" else "blue"
        self.console.print()
        self.console.print(Rule(title=Text(label, style=f"bold {style}"), align="left", style="dim"))
        self.console.print()

    def _compose_decision_line(self, chip: Text, trajectory: Trajectory) -> Text:
        """Build the main per-decision line: `{chip}  {tool}({args})  {telemetry}`."""
        line = Text("  ")  # base 2-space indent
        line.append(chip)
        line.append("  ")
        args_str = self._format_terminal_call(trajectory)
        line.append(args_str)
        tail = self._telemetry_tail(trajectory.lm_calls)
        if tail.plain:
            # Pad to a soft column for readability when width allows.
            pad = max(0, 56 - line.cell_len)
            line.append(" " * pad)
            line.append(tail)
        return line

    def _format_terminal_call(self, trajectory: Trajectory) -> Text:
        """Render the committed action: `tool(args) → result` for richer terminals."""
        tool = trajectory.terminal_tool
        commit_value = trajectory.committed_value

        if tool == "speak":
            quote = Text()
            quote.append("speak: ", style="cyan")
            quote.append(f"“{commit_value}”")
            return quote

        if tool in ("accuse", "defend", "pass_turn"):
            return self._format_reaction_call(trajectory)

        # Pull args from the matching ok-step in the trajectory; fall back to commit value.
        args_text = self._args_from_terminal_step(trajectory)
        out = Text()
        out.append(tool, style="bold")
        out.append("(")
        out.append(args_text)
        out.append(")")

        # For seer_inspect, append the inspection result if surfaced in the observation.
        if tool == "seer_inspect":
            faction = self._extract_faction_from_step(trajectory)
            if faction:
                out.append("  →  ")
                out.append(faction, style="bold magenta")
        return out

    def _args_from_terminal_step(self, trajectory: Trajectory) -> Text:
        for step in trajectory.react_trajectory:
            if step.tool == trajectory.terminal_tool and step.observation.startswith("ok:"):
                return Text(", ".join(f"{k}={_format_arg_value(v)}" for k, v in step.args.items()))
        # No matching step (defensive); render the commit value alone.
        return Text(str(trajectory.committed_value))

    def _format_reaction_call(self, trajectory: Trajectory) -> Text:
        """Render a day reaction: `accuse Bob: "…"`, `defend Carol: "…"`, or `pass`.

        The accuse/defend target + reason live in the committing step's args (the
        `committed_value` is a sanitized `Reaction` repr, not the structured
        fields), so pull them from the matching ok-step.
        """
        tool = trajectory.terminal_tool
        out = Text()
        if tool == "pass_turn":
            out.append("pass", style="dim")
            return out
        args = self._terminal_step_args(trajectory)
        target = str(args.get("target", "?"))
        reason = str(args.get("reason", ""))
        out.append(f"{tool} ", style="red" if tool == "accuse" else "green")
        out.append(target, style="bold")
        if reason:
            out.append(f": “{reason}”")
        return out

    def _terminal_step_args(self, trajectory: Trajectory) -> Mapping[str, object]:
        """Return the committing terminal step's args mapping, or `{}` if none matched."""
        for step in trajectory.react_trajectory:
            if step.tool == trajectory.terminal_tool and step.observation.startswith("ok:"):
                return step.args
        return {}

    def _extract_faction_from_step(self, trajectory: Trajectory) -> str | None:
        """Pull the inspect-result faction out of the committed step's args/observation."""
        # The engine emits the SEER_INSPECT event payload with the faction; the
        # trajectory step's args carry only the target. We don't have the
        # faction here unless it's surfaced — return None and let the user
        # read the sidecar. (Future: pass through engine result via trace_sink.)
        return None

    def _telemetry_tail(self, lm_calls: tuple[LMCallRecord, ...]) -> Text:
        if not lm_calls:
            return Text("")
        tokens = sum(c.prompt_tokens + c.completion_tokens for c in lm_calls)
        latency_ms = sum(c.latency_ms for c in lm_calls)
        cost_parts = [c.cost_usd for c in lm_calls if c.cost_usd is not None]
        cost = sum(cost_parts) if cost_parts else None
        out = Text(style="dim")
        out.append(f"{tokens}t")
        out.append(" · ")
        out.append(_fmt_latency(latency_ms))
        if cost is not None:
            out.append(" · ")
            out.append(f"${cost:.4f}")
        return out

    def _flush_thought(self, caller: str, role: str, iter_idx: int) -> None:
        """Pop the buffered thought for `(caller, iter_idx)` and print it.

        Renders nothing if no chunks were buffered (e.g. streaming was off
        or the LM didn't emit a `next_thought` field). Clears the entry
        in either case so a stale buffer can't leak into a later iter.
        """
        chunks = self._thought_buffers.pop((caller, iter_idx), None)
        if not chunks or not self.settings.stream_thoughts:
            return
        text = "".join(chunks).strip()
        if not text:
            return
        color = _ROLE_COLORS.get(role, "default")
        line = Text("    💭 ", style=f"dim {color}" if not self.settings.ascii else "dim")
        line.append(f"{caller}: ", style="dim")
        line.append(text, style="dim italic")
        self.console.print(line)

    def _render_chat_sub(self, step: ReActStep) -> None:
        message = step.args.get("message", "")
        line = Text("     ⤷ ", style="dim")
        line.append("chat: ", style="dim magenta")
        line.append(f"“{message}”")
        self.console.print(line)

    def _render_rejection_sub(self, step: ReActStep) -> None:
        reason = step.observation.removeprefix("error:").strip()
        target = step.args.get("target", step.args)
        line = Text("     ⤷ ", style="dim")
        line.append("✗ ", style="red")
        line.append(f"{step.tool}({target}) rejected: {reason}", style="dim red")
        self.console.print(line)

    def _render_kill_resolved(self, event: Event) -> None:
        protected = bool(event.payload.get("protected"))
        victim = event.payload.get("victim")
        line = Text("  ")
        line.append("✔ ", style="bold green" if protected else "bold red")
        line.append("kill_resolved", style="bold")
        line.append(" · ")
        if protected or victim is None:
            line.append(f"{victim or '(unknown)'} protected — no death", style="green")
        else:
            line.append(f"{victim} killed", style="red")
        self.console.print(line)

    def _render_kill_ballots(self, event: Event) -> None:
        ballots = event.payload.get("ballots", {})
        if isinstance(ballots, Mapping):
            ballot_text = ", ".join(f"{v}→{t}" for v, t in sorted(ballots.items()))
        else:
            ballot_text = str(ballots)
        line = Text("  ")
        line.append("· ", style="dim")
        line.append("kill_ballots", style="dim")
        line.append(" · ")
        line.append(ballot_text, style="dim red")
        line.append("  ")
        line.append(self._privacy_suffix(event), style="dim")
        self.console.print(line)

    def _render_exile_resolved(self, event: Event) -> None:
        exiled = event.payload.get("exiled")
        ballots = event.payload.get("ballots", {})
        line = Text("  ")
        line.append("⚔ ", style="bold yellow")
        line.append("exile_resolved", style="bold")
        line.append(" · ")
        if exiled is None or exiled == "abstain":
            line.append("no exile (tie or abstain)", style="dim yellow")
        else:
            line.append(f"{exiled} exiled", style="yellow")
        if isinstance(ballots, Mapping) and ballots:
            tally: dict[str, int] = {}
            for target in ballots.values():
                if isinstance(target, str):
                    tally[target] = tally.get(target, 0) + 1
            tally_text = ", ".join(f"{t}={n}" for t, n in sorted(tally.items(), key=lambda kv: -kv[1]))
            line.append("  ")
            line.append(f"({tally_text})", style="dim")
        self.console.print(line)

    def _render_discussion_resolved(self, event: Event) -> None:
        speakers = event.payload.get("speakers", [])
        bids = event.payload.get("bids", {}) or {}
        if not isinstance(speakers, list) or not speakers:
            return  # no chosen speakers — skip the empty line entirely
        parts: list[str] = []
        for s in speakers:
            if not isinstance(s, str):
                continue
            bid = bids.get(s) if isinstance(bids, Mapping) else None
            parts.append(f"{s}({bid})" if bid is not None else s)
        line = Text("  ")
        line.append("💬 ", style="cyan")
        line.append("discussion", style="bold")
        line.append(" · ")
        line.append(" → ".join(parts), style="cyan")
        self.console.print(line)

    def _render_game_over(self, event: Event) -> None:
        winner = event.payload.get("winner", "?")
        line = Text("  ")
        line.append("◆ ", style="bold")
        line.append("game_over", style="bold")
        line.append(" · winner ")
        style = "bold green" if winner == "villagers" else "bold red"
        line.append(str(winner), style=style)
        self.console.print(line)

    # --- helpers --------------------------------------------------------

    def _compose_memory_block(self, name: str, role: str, memory: Mapping[str, object]) -> Text:
        """Render one agent's memory block: header chip + plan + beliefs + notes."""
        out = Text()
        out.append(self._role_chip(name, role))
        out.append("\n")

        plan = memory.get("plan") or ""
        if plan:
            out.append("  plan: ", style="dim")
            out.append(str(plan))
            out.append("\n")
        else:
            out.append("  plan: ", style="dim")
            out.append("(none)", style="dim")
            out.append("\n")

        beliefs = memory.get("beliefs") or {}
        if isinstance(beliefs, Mapping) and beliefs:
            out.append("  beliefs:\n", style="dim")
            for player in sorted(beliefs):
                row = beliefs[player]
                if not isinstance(row, Mapping):
                    continue
                guess = str(row.get("guess", ""))
                confidence = str(row.get("confidence", ""))
                evidence = str(row.get("evidence", "")) or "(no evidence)"
                out.append(f"    {player}: ")
                out.append(guess, style="bold")
                out.append(f" ({confidence}) — {evidence}\n")
        else:
            out.append("  beliefs: ", style="dim")
            out.append("(none)", style="dim")
            out.append("\n")

        notes = memory.get("notes") or []
        if isinstance(notes, list) and notes:
            out.append("  notes:\n", style="dim")
            for note in notes:
                if not isinstance(note, Mapping):
                    continue
                round_ = note.get("round")
                text = str(note.get("text", ""))
                out.append(f"    [R{round_}] {text}\n")
        else:
            out.append("  notes: ", style="dim")
            out.append("(none)", style="dim")
            out.append("\n")

        return out

    def _role_chip(self, name: str, role: str) -> Text:
        glyph_map = _ROLE_GLYPHS_ASCII if self.settings.ascii else _ROLE_GLYPHS_EMOJI
        glyph = glyph_map.get(role, "  ")
        color = _ROLE_COLORS.get(role, "default")
        chip = Text()
        chip.append(f"{name:<5}", style=f"bold {color}")
        chip.append(" ")
        chip.append(glyph)
        return chip

    def _privacy_suffix(self, event: Event) -> str:
        if not event.recipients:
            return ""
        return f"[private → {','.join(event.recipients)}]"


# --- module-level formatters ---------------------------------------------


def _format_arg_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _format_args(args: Mapping[str, object]) -> str:
    """Render a tool's kwargs as `key=value, key=value` for the streaming sub-bullet."""
    if not args:
        return ""
    parts: list[str] = []
    for key, value in args.items():
        rendered = _format_arg_value(value)
        if isinstance(value, str) and len(rendered) > 60:
            rendered = rendered[:57] + "…"
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _iter_telemetry_tail(lm_calls: tuple[LMCallRecord, ...]) -> Text:
    """Build the dim telemetry suffix for one streamed iteration."""
    if not lm_calls:
        return Text("")
    total_prompt = sum(c.prompt_tokens for c in lm_calls)
    total_completion = sum(c.completion_tokens for c in lm_calls)
    total_tokens = total_prompt + total_completion
    latency_ms = sum(c.latency_ms for c in lm_calls)
    cost = sum(c.cost_usd for c in lm_calls if c.cost_usd is not None)
    tail = Text(style="dim")
    tail.append(f"{_fmt_tokens(total_tokens)} · {_fmt_latency(latency_ms)}")
    if cost > 0:
        tail.append(f" · ${cost:.4f}")
    return tail


def _fmt_tokens(n: int) -> str:
    if n < 1000:
        return f"{n}t"
    if n < 1_000_000:
        return f"{n / 1000:.1f}kt"
    return f"{n / 1_000_000:.2f}Mt"


def _fmt_latency(ms: float) -> str:
    if ms < 1000:
        return f"{round(ms)}ms"
    return f"{ms / 1000:.1f}s"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m {remainder:04.1f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {remainder:04.1f}s"
