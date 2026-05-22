"""Tests for per-agent `GameMemory` (T19, Tier 0).

These encode the consumer side of benchmark invariants #2 and #4: agents
never read hidden state and the recorded history is deterministic. The
engine routes events to the right agents (T05); `GameMemory` is what each
agent stores. It must:

  - keep per-agent state isolated (no cross-instance leak),
  - add zero nondeterminism (no time, uuid, randomness, or LLM/embedding
    call inside `recall` / `remember` / `set_belief`),
  - serialize `recall` output as a byte-deterministic function of the call
    sequence (the integrity reviewer's per-agent restatement of #4).

Tier 0 retrieval is the WEREWOLF_DESIGN.md §7 table's first row: return
the full event log + notes, optionally filtered by `last_n_rounds`. No
keyword/substring filter (that is Tier 1, parked for a later task).
"""

import re
from types import MappingProxyType

import pytest

from social_deduction_bench.agents import Belief, GameMemory, Note
from social_deduction_bench.engine import Event, Phase


def _event(round_: int, type_: str, payload: dict[str, object] | None = None) -> Event:
    """Build a public Event with a fixed seq=0 — `record_event` does not key on seq."""
    return Event(seq=0, round=round_, phase=Phase.DAY, type=type_, payload=payload or {})


# --- Construction & accessors --------------------------------------------------


def test_new_memory_is_empty_everything_blank() -> None:
    """A fresh `GameMemory` carries no history — accessors are empty and recall is `""`."""
    m = GameMemory()
    assert m.events == ()
    assert m.notes == ()
    assert dict(m.beliefs) == {}
    assert m.plan == ""
    assert m.recall() == ""


def test_events_accessor_returns_tuple_snapshot() -> None:
    """`events` is a snapshot — a tuple taken before a later append must not change.

    Mirrors `EventLog.events` (the engine precedent). A live-view accessor
    would silently invalidate test assertions and cross-thread reads.
    """
    m = GameMemory()
    m.record_event(_event(1, "kill_resolved"))
    snapshot = m.events
    m.record_event(_event(2, "exile_resolved"))
    assert len(snapshot) == 1
    assert len(m.events) == 2


def test_notes_accessor_returns_tuple_snapshot() -> None:
    """`notes` is a snapshot — same contract as `events`, applied to agent notes."""
    m = GameMemory()
    m.remember("first", round_=1)
    snapshot = m.notes
    m.remember("second", round_=2)
    assert len(snapshot) == 1
    assert len(m.notes) == 2


def test_beliefs_accessor_is_read_only_mapping() -> None:
    """`beliefs` is a `MappingProxyType` — direct mutation raises `TypeError`.

    Beliefs are the structured suspicion table (WEREWOLF_DESIGN.md §8); a
    caller mutating the returned mapping in place would silently bypass
    `set_belief`'s validation. Mirrors `Event.payload`'s read-only proxy.
    """
    m = GameMemory()
    m.set_belief("Bob", "werewolf", "high", "voted with the pack")
    assert isinstance(m.beliefs, MappingProxyType)
    with pytest.raises(TypeError):
        m.beliefs["Bob"] = Belief(player="Bob", guess="villager", confidence="low", evidence="")  # type: ignore[index]


# --- record_event --------------------------------------------------------------


def test_record_event_appends_to_events() -> None:
    """`record_event` stores the engine `Event` byte-identical for later retrieval."""
    m = GameMemory()
    e = _event(1, "kill_resolved", {"victim": "Alice"})
    m.record_event(e)
    assert m.events == (e,)


def test_record_event_preserves_insertion_order() -> None:
    """The accessor tuple matches the push order, regardless of `event.seq`.

    The agent loop pushes the player-visible slice of the global log; memory
    is the consumer's record, not the engine's, so it must respect the
    insertion order it was given.
    """
    m = GameMemory()
    first = Event(seq=7, round=2, phase=Phase.DAY, type="b", payload={})
    second = Event(seq=3, round=1, phase=Phase.NIGHT, type="a", payload={})
    m.record_event(first)
    m.record_event(second)
    assert m.events == (first, second)


def test_record_event_does_not_dedupe() -> None:
    """Re-pushing the same `Event` yields length-2 events — dedup is the loop's job.

    Adding a dedup check here would put two authorities on event ordering
    (`EventLog` + `GameMemory`); we keep memory dumb so an accidental
    double-push is visible to the loop's tests, not silently swallowed.
    """
    m = GameMemory()
    e = _event(1, "kill_resolved")
    m.record_event(e)
    m.record_event(e)
    assert len(m.events) == 2


# --- remember ------------------------------------------------------------------


def test_remember_appends_a_note_with_round_and_text() -> None:
    """`remember(text, round_)` stores a `Note(round, text)` in insertion order."""
    m = GameMemory()
    m.remember("Bob looks suspicious", round_=2)
    assert m.notes == (Note(round=2, text="Bob looks suspicious"),)


def test_remember_rejects_blank_text() -> None:
    """Blank notes are a write-side mistake (LLM emitted whitespace) — fail loud.

    Silently dropping or storing blank notes would let `recall` show empty
    lines as if they were real entries.
    """
    m = GameMemory()
    with pytest.raises(ValueError, match="text"):
        m.remember("   ", round_=1)


def test_remember_rejects_negative_round() -> None:
    """A negative `round_` cannot come from a legal game — fail loud at write time."""
    m = GameMemory()
    with pytest.raises(ValueError, match="round"):
        m.remember("note", round_=-1)


# --- set_belief ----------------------------------------------------------------


def test_set_belief_inserts_row() -> None:
    """`set_belief(player, ...)` stores a row keyed by the subject's name."""
    m = GameMemory()
    m.set_belief("Bob", "werewolf", "high", "voted with the pack")
    assert m.beliefs["Bob"] == Belief(player="Bob", guess="werewolf", confidence="high", evidence="voted with the pack")


def test_set_belief_overwrites_previous_row() -> None:
    """A second `set_belief` for the same subject replaces the row — beliefs are a current view, not history.

    WEREWOLF_DESIGN.md §8 frames beliefs as a "structured suspicion table"
    with one row per player; revisions replace, they do not stack.
    """
    m = GameMemory()
    m.set_belief("Bob", "werewolf", "high", "voted with the pack")
    m.set_belief("Bob", "villager", "low", "voted alone day 2")
    assert m.beliefs["Bob"].guess == "villager"
    assert m.beliefs["Bob"].confidence == "low"
    assert m.beliefs["Bob"].evidence == "voted alone day 2"


def test_set_belief_accepts_low_medium_high() -> None:
    """The three confidence buckets are the full legal set."""
    m = GameMemory()
    for level in ("low", "medium", "high"):
        m.set_belief("Bob", "werewolf", level, "")
        assert m.beliefs["Bob"].confidence == level


@pytest.mark.parametrize("bad_confidence", ["uncertain", "LOW", "Medium", "extreme", "", "very high"])
def test_set_belief_rejects_other_confidence_values(bad_confidence: str) -> None:
    """Any confidence outside the literal set raises — silently accepting a
    typo'd value (`"medum"`) or a case-shifted one (`"HIGH"`) would let the
    structured-table contract drift case by case.
    """
    m = GameMemory()
    with pytest.raises(ValueError, match="confidence"):
        m.set_belief("Bob", "werewolf", bad_confidence, "")  # type: ignore[arg-type]


def test_set_belief_rejects_blank_player_or_guess() -> None:
    """Blank `player` / `guess` are write-side mistakes — fail loud at write time."""
    m = GameMemory()
    with pytest.raises(ValueError, match="player"):
        m.set_belief("   ", "werewolf", "high", "")
    with pytest.raises(ValueError, match="guess"):
        m.set_belief("Bob", "  ", "high", "")


def test_set_belief_accepts_empty_evidence() -> None:
    """`evidence` may be empty — early game, the agent has no concrete evidence yet."""
    m = GameMemory()
    m.set_belief("Bob", "werewolf", "low", "")
    assert m.beliefs["Bob"].evidence == ""


# --- set_plan ------------------------------------------------------------------


def test_new_memory_plan_is_empty_string() -> None:
    """A fresh memory's `plan` is the empty string (the §9 "no plan yet" state).

    Pins the `str` invariant: callers (T16's `get_plan`) never need to
    handle `None`. The default is a value, not a missing field.
    """
    m = GameMemory()
    assert m.plan == ""
    assert isinstance(m.plan, str)


def test_set_plan_stores_text() -> None:
    """`set_plan(text)` round-trips through the `plan` accessor (§9)."""
    m = GameMemory()
    m.set_plan("reveal seer result on R2 if accused")
    assert m.plan == "reveal seer result on R2 if accused"


def test_set_plan_overwrites_previous_plan() -> None:
    """A second `set_plan` replaces the first — §9 calls plan "one persistent string," not a history.

    Mirrors `set_belief`'s overwrite semantics: an updated strategic
    posture replaces the prior one. A history would require a vocabulary
    the spec does not define.
    """
    m = GameMemory()
    m.set_plan("attack the doctor early")
    m.set_plan("protect the seer instead")
    assert m.plan == "protect the seer instead"


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t"])
def test_set_plan_rejects_blank_text(blank: str) -> None:
    """Blank plan text is a write-side mistake — fail loud at write time.

    Mirrors `Note.text`'s rule (`not s.strip()`): an empty plan is the
    default state, not an explicit assertion. If an agent ever wants to
    declare "no plan," it writes meaningful text like "no plan" — not a
    blank.
    """
    m = GameMemory()
    with pytest.raises(ValueError, match="Plan"):
        m.set_plan(blank)


@pytest.mark.parametrize("bad", [None, 123, 1.5, b"bytes", ["list"], {"k": "v"}])
def test_set_plan_rejects_non_str(bad: object) -> None:
    """`set_plan(non-str)` raises — belt-and-braces vs. Pyrefly.

    Mirrors the `isinstance(self.text, str)` guard in `Note` and the
    `isinstance` guards in `Belief`; surface a runtime contract violation
    at the write seam rather than corrupting a later `get_plan` call.
    `None` is included explicitly so a future "relax to truthy-only"
    refactor cannot silently let `None` through as a meaningful "clear."
    """
    m = GameMemory()
    with pytest.raises(ValueError, match="Plan"):
        m.set_plan(bad)  # type: ignore[arg-type]


def test_plan_is_not_surfaced_through_recall() -> None:
    """`recall()` renders events + notes only (§7 Tier 0) — the plan is a separate read surface.

    §9 says the plan "is part of memory," and a future refactor might be
    tempted to fold it into `recall()` for symmetry. That change would
    silently alter the LLM-facing context shape — pin the contract so it
    has to rewrite this test deliberately. T16's `get_plan` cognitive
    tool reads `memory.plan` directly.
    """
    m = GameMemory()
    m.set_plan("unique-plan-marker-xyzzy")
    m.record_event(_event(1, "kill_resolved"))
    m.remember("a note", round_=1)
    rendered = m.recall()
    assert "unique-plan-marker-xyzzy" not in rendered
    assert "plan" not in rendered  # neither the literal nor a "plan:" prefix


def test_plan_property_is_read_only() -> None:
    """`m.plan` is a property — direct assignment raises `AttributeError`.

    Pins the property/no-setter shape so a future regression (replacing
    the property with a plain attribute) cannot silently let callers
    bypass `set_plan`'s validation.
    """
    m = GameMemory()
    with pytest.raises(AttributeError):
        m.plan = "rebind"  # type: ignore[misc]


# --- recall (Tier 0) -----------------------------------------------------------


def test_recall_on_empty_memory_returns_empty_string() -> None:
    """An empty memory's recall is `""` — must not crash on `max([])`."""
    assert GameMemory().recall() == ""


def test_recall_renders_events_and_notes_together_in_round_order() -> None:
    """Events and notes mix into one stream, sorted by `round`; ties preserve insertion order."""
    m = GameMemory()
    m.record_event(_event(1, "kill_resolved", {"victim": "Alice"}))
    m.record_event(_event(3, "exile_resolved", {"exiled": "Bob"}))
    m.remember("trust Cara", round_=2)
    out = m.recall()
    lines = out.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("[R1] kill_resolved")
    assert lines[1] == "[R2] note: trust Cara"
    assert lines[2].startswith("[R3] exile_resolved")


def test_recall_event_line_format_is_stable() -> None:
    """An event line is `[R{round}] {type} {sorted-json-payload}` — keys sorted for replay.

    `events.py::to_jsonl_lines` uses `sort_keys=True` for byte-identical
    transcripts (invariant #4); `recall`'s LLM-facing render must do the
    same so two replays produce identical context windows. The payload
    keys are inserted in `victim`-then-`actor` order so a removal of
    `sort_keys=True` would flip the rendered output to insertion order
    and fail this assertion.
    """
    m = GameMemory()
    m.record_event(_event(1, "kill_resolved", {"victim": "Alice", "actor": "Bob"}))
    line = m.recall().splitlines()[0]
    assert re.match(r"^\[R\d+\] \w+ \{.*\}$", line)
    assert line == '[R1] kill_resolved {"actor": "Bob", "victim": "Alice"}'


def test_recall_note_line_format_is_stable() -> None:
    """A note line is `[R{round}] note: {text}` — single-line, no JSON wrapping."""
    m = GameMemory()
    m.remember("trust Cara", round_=2)
    assert m.recall() == "[R2] note: trust Cara"


def test_recall_last_n_rounds_returns_recent_window() -> None:
    """`recall(last_n_rounds=1)` keeps only the latest round — strict `round > latest - N`."""
    m = GameMemory()
    m.record_event(_event(1, "a"))
    m.record_event(_event(2, "b"))
    m.record_event(_event(3, "c"))
    lines = m.recall(last_n_rounds=1).splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("[R3] c")


def test_recall_last_n_rounds_zero_returns_empty_string() -> None:
    """`last_n_rounds=0` means "give me zero rounds" → `""`. No off-by-one inversion."""
    m = GameMemory()
    m.record_event(_event(1, "a"))
    m.record_event(_event(2, "b"))
    assert m.recall(last_n_rounds=0) == ""


def test_recall_last_n_rounds_larger_than_history_returns_everything() -> None:
    """A window wider than the history returns the same string as the unfiltered call."""
    m = GameMemory()
    m.record_event(_event(1, "a"))
    m.record_event(_event(2, "b"))
    m.record_event(_event(3, "c"))
    assert m.recall(last_n_rounds=99) == m.recall()


def test_recall_last_n_rounds_with_no_events_returns_empty() -> None:
    """Filtering an empty memory must not crash on `max([])` — returns `""`."""
    assert GameMemory().recall(last_n_rounds=1) == ""


def test_recall_negative_last_n_rounds_raises() -> None:
    """Negative window has no legal reading — fail loud rather than silently flipping sign."""
    with pytest.raises(ValueError, match="last_n_rounds"):
        GameMemory().recall(last_n_rounds=-1)


def test_recall_stable_when_events_share_round() -> None:
    """Same-round events come out in insertion order — stable sort + insertion-index tiebreak.

    Without insertion-order stability, the recall string would depend on
    sort algorithm details and break invariant #4 at the per-agent layer.
    """
    m = GameMemory()
    m.record_event(_event(2, "first"))
    m.record_event(_event(2, "second"))
    m.record_event(_event(2, "third"))
    lines = m.recall().splitlines()
    assert [line.split()[1] for line in lines] == ["first", "second", "third"]


def test_recall_orders_events_before_notes_at_same_round() -> None:
    """Same-round items: every event line comes before every note line.

    The secondary sort key is `kind` (event=0, note=1). Without it,
    same-round mixing would depend on whichever was appended first,
    which would break the rule "the engine acted, then the agent
    reflected" — the LLM-facing chronology must always show observed
    events ahead of the agent's own commentary on that round.
    """
    m = GameMemory()
    m.remember("first thought", round_=1)
    m.record_event(_event(1, "kill_resolved"))
    m.remember("second thought", round_=1)
    lines = m.recall().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("[R1] kill_resolved")
    assert lines[1] == "[R1] note: first thought"
    assert lines[2] == "[R1] note: second thought"


def test_recall_includes_notes_round_in_cutoff() -> None:
    """A lone note at round 5 with `last_n_rounds=1` returns that note.

    Notes contribute to the `latest`-round computation, not just events —
    otherwise an agent who only writes notes would see `recall` return `""`.
    """
    m = GameMemory()
    m.remember("late insight", round_=5)
    assert m.recall(last_n_rounds=1) == "[R5] note: late insight"


# --- Determinism & isolation ---------------------------------------------------


def test_two_memories_independent() -> None:
    """Appends to one instance do not leak into another — per-agent isolation (#2 consumer side).

    The engine routes private events to the right recipient; the consumer
    boundary must not unify two agents' memories through shared state.
    """
    m1 = GameMemory()
    m2 = GameMemory()
    m1.record_event(_event(1, "kill_resolved"))
    m1.remember("note", round_=1)
    m1.set_belief("Bob", "werewolf", "high", "")
    m1.set_plan("attack Bob")
    assert m2.events == ()
    assert m2.notes == ()
    assert dict(m2.beliefs) == {}
    assert m2.plan == ""


def test_identical_operation_sequence_yields_identical_state() -> None:
    """Two memories built from the same op sequence produce byte-identical state.

    Per-agent restatement of invariant #4: `recall` / `remember` /
    `set_belief` / `set_plan` add no nondeterminism (no time, uuid,
    randomness). Two runs of the same seeded game must produce identical
    agent-side context — `recall` rendering, belief table, and plan.
    """

    def build() -> GameMemory:
        m = GameMemory()
        m.record_event(_event(1, "kill_resolved", {"victim": "Alice"}))
        m.remember("trust Cara", round_=2)
        m.record_event(_event(3, "exile_resolved", {"exiled": "Bob"}))
        m.set_belief("Bob", "werewolf", "high", "voted with the pack")
        m.set_plan("alpha")
        m.set_plan("beta")
        return m

    m1, m2 = build(), build()
    assert m1.recall() == m2.recall()
    assert m1.recall(last_n_rounds=2) == m2.recall(last_n_rounds=2)
    assert m1.plan == m2.plan == "beta"


def test_recall_does_not_mutate_internal_state() -> None:
    """`recall(last_n_rounds=1)` then `recall()` returns the full history.

    A `recall` that filtered the internal list in place would silently
    drop history; this pins the read-only contract.
    """
    m = GameMemory()
    m.record_event(_event(1, "a"))
    m.record_event(_event(2, "b"))
    m.record_event(_event(3, "c"))
    _ = m.recall(last_n_rounds=1)
    assert len(m.recall().splitlines()) == 3


# --- to_json_dict --------------------------------------------------------------


def test_to_json_dict_emits_plan_beliefs_and_notes_excluding_events() -> None:
    """`to_json_dict` is the post-game memory dump shape.

    Events are intentionally excluded — they already live in `events.jsonl`.
    The interesting agent-private state is plan + beliefs + notes; the
    CLI's `memories.json` sidecar reads exactly this shape.
    """
    m = GameMemory()
    m.record_event(_event(1, "kill_resolved", {"victim": "Alice"}))
    m.remember("trust Cara", round_=2)
    m.set_belief("Bob", "werewolf", "high", "voted with the pack")
    m.set_plan("frame the seer")

    dump = m.to_json_dict()

    assert dump == {
        "plan": "frame the seer",
        "beliefs": {
            "Bob": {"guess": "werewolf", "confidence": "high", "evidence": "voted with the pack"},
        },
        "notes": [{"round": 2, "text": "trust Cara"}],
    }
    assert "events" not in dump


def test_to_json_dict_round_trips_through_json() -> None:
    """The dump is pure JSON primitives so the sidecar can serialize it.

    A nested non-primitive would crash the writer mid-game-over.
    """
    import json

    m = GameMemory()
    m.remember("watch Wolf1", round_=1)
    m.set_belief("Wolf1", "werewolf", "medium", "")
    m.set_plan("inspect tonight")

    serialized = json.dumps(m.to_json_dict(), sort_keys=True)
    restored = json.loads(serialized)
    assert restored["plan"] == "inspect tonight"
    assert restored["beliefs"]["Wolf1"]["confidence"] == "medium"
    assert restored["notes"][0] == {"round": 1, "text": "watch Wolf1"}
