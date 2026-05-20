"""Per-agent in-process memory (Tier 0).

Upholds the consumer side of benchmark invariants #2 and #4: agents
receive only events the engine routes to them (T05), and the recorded
history is deterministic and replayable. `GameMemory` is what each agent
stores — it must add zero nondeterminism (no `time`, no `uuid`, no
randomness, no LLM / embedding call) and must keep per-agent state
isolated (one instance per player, no shared state).

WEREWOLF_DESIGN.md §7 declines a vector store: a 7-player game's
transcript is a few thousand tokens, and an LLM-driven retrieval layer
would turn replay nondeterministic. Tier 0 is therefore the full event
log + agent notes, optionally filtered to the last N rounds — no keyword
filter (Tier 1), no embeddings (Tier 2).

§8 makes beliefs separate from notes: a structured one-row-per-subject
table (`{guess, confidence, evidence}`) the LLM updates via `set_belief`.
Confidence is an ordinal bucket — `"low"` / `"medium"` / `"high"` — so a
typo or case shift fails loud at write time rather than drifting case by
case. Beliefs are a current view (each call replaces the row), not a
history.

§9 adds one persistent `plan` string per agent — written via `set_plan`,
read via `plan`, with the same overwriting (not historical) semantics as
a belief row. The default `""` is the "no plan yet" state.

`record_event` is the ingestion seam the T21 agent loop calls once per
decision point after `observations_for(events, player)`. The memory does
not filter by recipient; that is the engine routing layer's job (T05).
Re-pushing the same event is allowed (no dedup) so a double-push bug in
the loop is visible to the loop's tests instead of being silently
swallowed here.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from social_deduction_bench.engine import Event

Confidence = Literal["low", "medium", "high"]
_CONFIDENCE_LEVELS: frozenset[str] = frozenset({"low", "medium", "high"})


@dataclass(frozen=True, slots=True)
class Note:
    """One free-text entry the agent writes to its own memory.

    Frozen + slots mirrors `engine.Event` — once recorded, a note's text
    and round cannot be edited so `recall`'s output is a deterministic
    function of the write sequence.
    """

    round: int
    text: str

    def __post_init__(self) -> None:
        """Reject negative rounds and blank text — fail-loud at write time.

        A blank note would render as an empty line in `recall` (a silent
        corruption of the LLM's context); a negative round cannot come
        from a legal game and would invert the `last_n_rounds` window.
        """
        if self.round < 0:
            raise ValueError(f"Note 'round' must be non-negative, got {self.round}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"Note 'text' must be a non-blank str, got {self.text!r}")


@dataclass(frozen=True, slots=True)
class Belief:
    """One row of an agent's structured suspicion table.

    `player` is the *subject* of the belief (whom the agent suspects),
    not the holder; the holder is implicit — every `Belief` in
    `GameMemory.beliefs` is that memory's opinion. `guess` is a
    game-agnostic free `str`; the Werewolf cognitive-tool layer (T16)
    passes faction strings like `"werewolf"` / `"villager"`, but
    `GameMemory` does not enforce a vocabulary — that would couple
    `agents/` to one game.
    """

    player: str
    guess: str
    confidence: Confidence
    evidence: str

    def __post_init__(self) -> None:
        """Reject blank `player` / `guess`, non-str `evidence`, and any
        confidence outside `{"low", "medium", "high"}` — fail-loud at
        write time, so a typo (`"medum"`) or case shift (`"HIGH"`)
        cannot drift the structured-table contract case by case.
        """
        if not isinstance(self.player, str) or not self.player.strip():
            raise ValueError(f"Belief 'player' must be a non-blank str, got {self.player!r}")
        if not isinstance(self.guess, str) or not self.guess.strip():
            raise ValueError(f"Belief 'guess' must be a non-blank str, got {self.guess!r}")
        if self.confidence not in _CONFIDENCE_LEVELS:
            raise ValueError(
                f"Belief 'confidence' must be one of {sorted(_CONFIDENCE_LEVELS)}, got {self.confidence!r}"
            )
        if not isinstance(self.evidence, str):
            raise ValueError(f"Belief 'evidence' must be a str, got {type(self.evidence).__name__}")


class GameMemory:
    """Per-agent in-process memory (Tier 0). No DB, no embeddings, no LLM calls.

    Owns three append-only / overwriting buffers:

      - `_events`: engine-pushed `Event`s the agent has observed.
      - `_notes`:  the agent's own `remember()` entries.
      - `_beliefs`: one `Belief` per subject (overwriting on revision).

    Public accessors return read-only views: tuple snapshots for events
    and notes (matches `EventLog.events`), a `MappingProxyType` for
    beliefs (matches `Event.payload`). Direct mutation through the
    accessors is impossible — every update goes through a validating
    method.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._notes: list[Note] = []
        self._beliefs: dict[str, Belief] = {}
        self._plan: str = ""

    # --- ingest ---------------------------------------------------------

    def record_event(self, event: Event) -> None:
        """Append `event` to this agent's observed history.

        Pushed by the T21 agent loop after `observations_for(events,
        player)` — `GameMemory` trusts that filter and does not re-check
        recipients. Re-pushing the same event is allowed (no dedup); a
        double-push is a loop bug, surfaced by loop tests rather than
        silently absorbed here.
        """
        self._events.append(event)

    def remember(self, note: str, round_: int) -> None:
        """Write a free-text note tagged with the round the agent thought of it.

        Delegates validation to `Note.__post_init__` (blank text and
        negative rounds fail loud there).
        """
        self._notes.append(Note(round=round_, text=note))

    def set_belief(self, player: str, guess: str, confidence: Confidence, evidence: str) -> None:
        """Write or overwrite the suspicion row for `player`.

        Beliefs are a current view, not a history — a second call for
        the same subject replaces the row. Validation lives in
        `Belief.__post_init__`.
        """
        self._beliefs[player] = Belief(player=player, guess=guess, confidence=confidence, evidence=evidence)

    def set_plan(self, text: str) -> None:
        """Write or overwrite the agent's persistent strategic plan (§9).

        Plans are a current view, not a history — a second call replaces
        the previous one. Blanks fail loud (mirrors `remember`); an
        explicit clear is meaningful text the LLM can write, not a blank.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Plan 'text' must be a non-blank str, got {text!r}")
        self._plan = text

    # --- read-only snapshots -------------------------------------------

    @property
    def events(self) -> tuple[Event, ...]:
        """A tuple snapshot of recorded events — does not change when the buffer grows later."""
        return tuple(self._events)

    @property
    def notes(self) -> tuple[Note, ...]:
        """A tuple snapshot of agent notes — same snapshot semantics as `events`."""
        return tuple(self._notes)

    @property
    def beliefs(self) -> Mapping[str, Belief]:
        """A read-only `MappingProxyType` view of the belief table.

        The proxy is live (it grows as `set_belief` is called) but
        immutable — `m.beliefs["X"] = ...` raises `TypeError`. Callers
        wanting an isolated copy can `dict(m.beliefs)`.
        """
        return MappingProxyType(self._beliefs)

    @property
    def plan(self) -> str:
        """The agent's current persistent plan (§9). ``""`` until first `set_plan`."""
        return self._plan

    # --- Tier 0 retrieval ----------------------------------------------

    def recall(self, last_n_rounds: int | None = None) -> str:
        """Return the LLM-facing render of events + notes, optionally
        filtered to the last `N` rounds.

        Per WEREWOLF_DESIGN.md §7 Tier 0: full event log + notes,
        optionally `last_n_rounds`-filtered. No keyword filter (Tier 1).

        Output is one line per item:

          - event: ``[R{round}] {type} {payload-as-sorted-json}``
          - note:  ``[R{round}] note: {text}``

        Items are sorted by ``(round, kind, insertion_index)`` with
        events as kind 0 and notes as kind 1; the stable sort + explicit
        insertion-index tiebreak makes the output a byte-deterministic
        function of the write sequence (invariant #4 at the per-agent
        layer). Empty memory and `last_n_rounds=0` both return ``""``.

        ``last_n_rounds < 0`` raises ``ValueError`` — a negative window
        has no legal reading and silently flipping the sign would mask
        a caller bug.
        """
        if last_n_rounds is not None and last_n_rounds < 0:
            raise ValueError(f"last_n_rounds must be non-negative or None, got {last_n_rounds}")

        items: list[tuple[int, int, int, str]] = []
        for index, event in enumerate(self._events):
            payload_json = json.dumps(dict(event.payload), sort_keys=True)
            items.append((event.round, 0, index, f"[R{event.round}] {event.type} {payload_json}"))
        for index, note in enumerate(self._notes):
            items.append((note.round, 1, index, f"[R{note.round}] note: {note.text}"))

        if not items:
            return ""

        if last_n_rounds is not None:
            latest = max(item[0] for item in items)
            cutoff = latest - last_n_rounds
            items = [item for item in items if item[0] > cutoff]

        items.sort(key=lambda item: (item[0], item[1], item[2]))
        return "\n".join(item[3] for item in items)
