"""Tests for the engine event stream (T04).

These encode benchmark invariants #2, #4, and #5:

- #5 — every game is an append-only event stream (JSONL): replayable,
  debuggable, the basis for post-hoc metrics. The log must never lose, reorder,
  or retroactively edit a recorded event.
- #4 — deterministic and replayable: the JSONL round-trip must be lossless and
  serialization byte-identical for equal streams, so T08 can compare two runs.
- #2 — agents never read hidden state. T05 routes events; T04 only carries the
  `visibility`/`recipient` marker, and a private event with no addressee — a
  hidden-state-leak vector — must be rejected, never silently loaded.

If the log becomes mutable beyond appends, loses ordering, or the round-trip
drops a field, these fail loudly.
"""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from social_deduction_bench.engine import (
    Event,
    EventLog,
    EventStream,
    GameState,
    Phase,
    StreamHeader,
    Visibility,
    read_jsonl,
    write_jsonl,
)

# A canonical 4-player roster, in `GameState.initial`'s (name, role) shape.
PLAYERS = (("Alice", "villager"), ("Bob", "werewolf"), ("Cara", "seer"), ("Dan", "doctor"))


def _sample_stream() -> EventStream:
    """Build a small mixed public/private stream reused across round-trip tests."""
    log = EventLog()
    log.append(round=1, phase=Phase.NIGHT, type="phase_change", payload={"phase": "night"})
    log.append(
        round=1,
        phase=Phase.NIGHT,
        type="seer_inspect",
        payload={"target": "Bob", "is_werewolf": True},
        visibility=Visibility.PRIVATE,
        recipient="Cara",
    )
    log.append(round=1, phase=Phase.DAY, type="speak", payload={"text": "I suspect Bob"})
    header = StreamHeader(seed=20260518, game_id="game-test", players=PLAYERS)
    return EventStream(header=header, log=log)


def _tamper_value(line: str, old: object, new: object) -> str:
    """Return `line` (a JSON object) with the field currently equal to `old` set to `new`.

    Locates the field by value so the tests need not hardcode wire-format key
    names — only the documented values (`Phase.NIGHT.value`, a recipient name).
    """
    obj = json.loads(line)
    key = next(k for k, v in obj.items() if v == old)
    obj[key] = new
    return json.dumps(obj)


# --- Event schema & invariant enforcement ---------------------------------


def test_event_is_frozen() -> None:
    """An `Event` field cannot be reassigned in place.

    A recorded event is history; a mutable one could be edited after the log
    captured it, silently corrupting the replayable stream (invariant #5).
    """
    event = Event(seq=0, round=1, phase=Phase.NIGHT, type="speak", payload={})
    with pytest.raises(FrozenInstanceError):
        event.seq = 9  # type: ignore[misc]  # assigning to a frozen field is the point


def test_event_payload_cannot_be_mutated_after_construction() -> None:
    """The `payload` mapping is immutable and isolated from its source dict.

    A frozen dataclass around a mutable dict still leaks: the payload could be
    edited in place, or the caller's dict could mutate the event through a
    shared reference. Immutability must reach the payload.
    """
    source = {"votes": 3}
    event = Event(seq=0, round=1, phase=Phase.DAY, type="vote", payload=source)

    with pytest.raises(TypeError):
        event.payload["votes"] = 9  # type: ignore[index]  # payload is read-only

    source["votes"] = 99  # mutating the original dict must not reach the event
    assert event.payload["votes"] == 3


def test_private_event_without_recipient_raises() -> None:
    """A PRIVATE event with no `recipient` is rejected at construction.

    Invariant #2: an unaddressed private event cannot be routed by T05 — it
    would either leak to everyone or vanish. It must fail loud, not be stored.
    """
    with pytest.raises(ValueError, match="recipient"):
        Event(seq=0, round=1, phase=Phase.NIGHT, type="seer_inspect", payload={}, visibility=Visibility.PRIVATE)


def test_public_event_with_recipient_raises() -> None:
    """A PUBLIC event carrying a `recipient` is rejected.

    A public event addressed to one player is a contradiction that would make
    the routing marker ambiguous for T05; the marker must be unambiguous.
    """
    with pytest.raises(ValueError, match="recipient"):
        Event(
            seq=0,
            round=1,
            phase=Phase.DAY,
            type="speak",
            payload={},
            visibility=Visibility.PUBLIC,
            recipient="Alice",
        )


def test_public_event_defaults_to_no_recipient() -> None:
    """A PUBLIC event built without a recipient is valid and has `recipient is None`.

    Pins the common-case default so T05 can treat the absence of a recipient as
    "broadcast to everyone".
    """
    event = Event(seq=0, round=1, phase=Phase.DAY, type="speak", payload={})

    assert event.visibility is Visibility.PUBLIC
    assert event.recipient is None


def test_event_rejects_non_json_payload_value() -> None:
    """A payload value that is not a JSON primitive is rejected at construction.

    Invariant #4: the JSONL round-trip must be lossless. A tuple silently
    becomes a list on read-back, so `from_jsonl_lines(to_jsonl_lines())` would
    no longer equal the original — a non-JSON payload value must fail loud.
    """
    with pytest.raises(ValueError, match="payload"):
        Event(seq=0, round=1, phase=Phase.DAY, type="vote", payload={"picks": (1, 2)})


# --- EventLog append-only behavior ----------------------------------------


def test_append_assigns_contiguous_increasing_seq() -> None:
    """`append()` assigns `seq` 0, 1, 2 — callers never supply it.

    Invariant #5's ordering guarantee: a gap-free, log-assigned index is what
    T24/T28 reference and what read-back validates against.
    """
    log = EventLog()
    log.append(round=1, phase=Phase.NIGHT, type="a", payload={})
    log.append(round=1, phase=Phase.NIGHT, type="b", payload={})
    log.append(round=1, phase=Phase.DAY, type="c", payload={})

    assert [e.seq for e in log.events] == [0, 1, 2]


def test_append_returns_the_constructed_event() -> None:
    """`append()` returns the `Event` it built, carrying the fields passed in.

    The phase machine (T06) needs the event back — to log or assert on it —
    without re-reading the whole log.
    """
    log = EventLog()
    event = log.append(round=2, phase=Phase.DAY, type="exile", payload={"target": "Bob"})

    assert isinstance(event, Event)
    assert event.seq == 0
    assert event.round == 2
    assert event.phase is Phase.DAY
    assert event.type == "exile"
    assert event.payload["target"] == "Bob"


def test_events_view_is_immutable_tuple() -> None:
    """`EventLog.events` is a tuple snapshot; the log offers no remove/clear/insert.

    Append-only is the core safety property — the log must expose no way to
    delete, reorder, or otherwise rewrite recorded history.
    """
    log = EventLog()
    log.append(round=1, phase=Phase.NIGHT, type="a", payload={})
    snapshot = log.events
    assert isinstance(snapshot, tuple)

    for mutator in ("remove", "clear", "insert", "pop", "extend", "__setitem__", "__delitem__"):
        assert not hasattr(log, mutator), f"EventLog must not expose {mutator}"

    # A previously returned view must not change when the log grows further.
    log.append(round=1, phase=Phase.NIGHT, type="b", payload={})
    assert len(snapshot) == 1


def test_appended_event_is_visible_in_order() -> None:
    """Iterating the log yields events in insertion order.

    Replay (T28) and metric extraction (T24) depend on reading events in
    exactly the order they occurred.
    """
    log = EventLog()
    for name in ("a", "b", "c", "d"):
        log.append(round=1, phase=Phase.DAY, type=name, payload={})

    assert [e.type for e in log] == ["a", "b", "c", "d"]
    assert len(log) == 4


# --- StreamHeader ---------------------------------------------------------


def test_header_records_seed() -> None:
    """The header stores the engine seed unchanged.

    The locked T03 decision: the seed lives on `GameRNG` and is recorded once
    here. T08's determinism harness reads the seed from the header.
    """
    header = StreamHeader(seed=4242, game_id="g1", players=PLAYERS)

    assert header.seed == 4242


def test_header_roster_reconstructs_initial_game_state() -> None:
    """`GameState.initial(header.players)` rebuilds the start position.

    A transcript must be self-contained: replay (T28) reconstructs the game
    from the file alone, so the header roster must be exactly the
    `(name, role)` shape `GameState.initial` accepts.
    """
    header = StreamHeader(seed=1, game_id="g1", players=PLAYERS)
    state = GameState.initial(header.players)

    assert state.alive_names() == ("Alice", "Bob", "Cara", "Dan")
    assert state.player("Bob").role == "werewolf"


# --- JSONL lossless round-trip --------------------------------------------


def test_jsonl_round_trip_preserves_every_event_field() -> None:
    """A mixed public/private stream survives serialize -> deserialize intact.

    Invariant #4: replay requires a lossless round-trip — a dropped field
    silently corrupts every post-hoc metric computed from the stream.
    """
    original = _sample_stream()
    restored = EventStream.from_jsonl_lines(original.to_jsonl_lines())

    assert restored.log.events == original.log.events


def test_jsonl_round_trip_preserves_header() -> None:
    """The header — seed, game_id, roster — survives the round-trip unchanged.

    Losing the seed or roster makes the transcript unreplayable.
    """
    original = _sample_stream()
    restored = EventStream.from_jsonl_lines(original.to_jsonl_lines())

    assert restored.header == original.header


def test_jsonl_round_trip_preserves_enum_fields() -> None:
    """`phase` and `visibility` come back as enum members, not bare strings.

    A round-trip that returned strings would break every `is`-comparison and
    every routing check downstream.
    """
    restored = EventStream.from_jsonl_lines(_sample_stream().to_jsonl_lines())
    events = restored.log.events

    assert events[0].phase is Phase.NIGHT
    assert events[2].phase is Phase.DAY
    assert events[0].visibility is Visibility.PUBLIC
    assert events[1].visibility is Visibility.PRIVATE


def test_jsonl_first_line_is_the_header() -> None:
    """Line 1 of the JSONL is the header; loading it alone yields a header-only stream.

    Pins the documented file format so external tools and the replay reader
    (T28) can rely on the header sitting on the first line.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    header_only = EventStream.from_jsonl_lines([lines[0]])

    assert header_only.header == _sample_stream().header
    assert len(header_only.log) == 0


def test_jsonl_one_event_per_line() -> None:
    """A stream of N events serializes to exactly N+1 lines (header + one per event).

    The JSONL contract (design §11): one event per line keeps the stream
    append-friendly and line-grep-able for debugging.
    """
    stream = _sample_stream()
    lines = list(stream.to_jsonl_lines())

    assert len(lines) == len(stream.log) + 1


def test_jsonl_round_trip_handles_empty_log() -> None:
    """A stream with a header and zero events round-trips losslessly.

    A game that ends before any event is recorded is still a valid transcript;
    the round-trip must not depend on the log being non-empty.
    """
    stream = EventStream(header=StreamHeader(seed=7, game_id="g", players=PLAYERS), log=EventLog())
    restored = EventStream.from_jsonl_lines(stream.to_jsonl_lines())

    assert restored.header == stream.header
    assert restored.log.events == ()


# --- JSONL determinism ----------------------------------------------------


def test_jsonl_serialization_is_byte_identical_for_equal_streams() -> None:
    """Two equal streams serialize to identical text with canonically sorted keys.

    Invariant #4 — "same seed -> identical event stream". T08 compares runs at
    the text level, so serialization must be deterministic. The per-line sorted-
    key assertion makes this fail if `sort_keys` is ever dropped — without it,
    two streams built by the same helper would match on insertion order alone.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    assert lines == list(_sample_stream().to_jsonl_lines())

    # Re-dumping each parsed line with sorted keys must reproduce it byte-for-byte.
    for line in lines:
        assert json.dumps(json.loads(line), sort_keys=True) == line


# --- JSONL read-back validation (fail loud) -------------------------------


def test_from_jsonl_lines_rejects_non_contiguous_seq() -> None:
    """JSONL whose event `seq` values have a gap is rejected.

    A gap means an event was lost between write and read; a corrupt transcript
    must fail loud (invariant #5), never load silently.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    # Drop the middle event -> remaining events have seq 0 and 2.
    corrupted = [lines[0], lines[1], lines[3]]

    with pytest.raises(ValueError, match="seq"):
        EventStream.from_jsonl_lines(corrupted)


def test_from_jsonl_lines_rejects_out_of_order_seq() -> None:
    """JSONL whose event lines are reordered (descending `seq`) is rejected.

    Append-only ordering must be verified on read, not assumed — a reordered
    transcript is not the game that was played.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    shuffled = [lines[0], *reversed(lines[1:])]

    with pytest.raises(ValueError, match="seq"):
        EventStream.from_jsonl_lines(shuffled)


def test_from_jsonl_lines_rejects_private_event_missing_recipient() -> None:
    """JSONL with a `private` event whose recipient is null is rejected.

    Invariant #2: a tampered transcript that would leak a hidden-state event to
    everyone must be rejected at load time, not routed by T05.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    # The private event in _sample_stream() is addressed to "Cara".
    lines[2] = _tamper_value(lines[2], "Cara", None)

    with pytest.raises(ValueError, match="recipient"):
        EventStream.from_jsonl_lines(lines)


def test_from_jsonl_lines_rejects_unknown_phase_value() -> None:
    """JSONL with an unrecognized `phase` string is rejected.

    Fail loud (CLAUDE.md rule 11): a malformed transcript must surface, never
    coerce a bad phase to a default.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    lines[1] = _tamper_value(lines[1], Phase.NIGHT.value, "dusk")

    # The error must name the offending value, not coerce it to a default.
    with pytest.raises(ValueError, match="dusk"):
        EventStream.from_jsonl_lines(lines)


def test_from_jsonl_lines_rejects_empty_input() -> None:
    """Empty JSONL (no header line) is rejected.

    A stream with no header carries no seed and no roster — it is unreplayable;
    loading it must fail rather than yield a seedless stream.
    """
    with pytest.raises(ValueError, match="header"):
        EventStream.from_jsonl_lines([])


def test_from_jsonl_lines_rejects_unknown_visibility_value() -> None:
    """JSONL with an unrecognized `visibility` string is rejected.

    Symmetric to the unknown-`phase` case: invariant #2's routing marker must
    be a known value, or a tampered transcript could carry an unroutable event.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    lines[1] = _tamper_value(lines[1], Visibility.PUBLIC.value, "whispered")

    with pytest.raises(ValueError, match="whispered"):
        EventStream.from_jsonl_lines(lines)


def test_from_jsonl_lines_rejects_non_integer_seq() -> None:
    """JSONL whose event `seq` is not an integer is rejected naming the type fault.

    A tampered `seq` must fail loud as a type error, not slip through to a
    misleading "non-contiguous seq" message that hides the real corruption.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    lines[1] = _tamper_value(lines[1], 0, "0")  # seq 0 -> the string "0"

    with pytest.raises(ValueError, match="int"):
        EventStream.from_jsonl_lines(lines)


def test_from_jsonl_lines_rejects_malformed_json_line() -> None:
    """A line that is not valid JSON is rejected with a line-located error.

    A corrupt transcript must fail loud pointing at the offending line, not
    surface a raw decoder error from deep inside the parser.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    lines[1] = "{not json"

    with pytest.raises(ValueError, match="malformed JSONL"):
        EventStream.from_jsonl_lines(lines)


def test_from_jsonl_lines_rejects_malformed_player_pair() -> None:
    """A header player entry that is not a (name, role) pair is rejected.

    The roster must reconstruct `GameState.initial` exactly; a malformed entry
    must fail loud naming the roster fault, not unpack into a corrupt roster.
    """
    lines = list(_sample_stream().to_jsonl_lines())
    header_obj = json.loads(lines[0])
    header_obj["players"] = [["Alice", "villager", "extra"]]
    lines[0] = json.dumps(header_obj)

    with pytest.raises(ValueError, match="player"):
        EventStream.from_jsonl_lines(lines)


# --- File API -------------------------------------------------------------


def test_write_then_read_jsonl_round_trips_via_filesystem(tmp_path: Path) -> None:
    """`write_jsonl` then `read_jsonl` on a real file reproduces the stream.

    The on-disk `.jsonl` file is the artifact every game produces (design §11);
    the file layer must be lossless, not just the in-memory string layer.
    """
    original = _sample_stream()
    path = tmp_path / "game.jsonl"
    write_jsonl(original, path)
    restored = read_jsonl(path)

    assert restored.header == original.header
    assert restored.log.events == original.log.events


def test_written_jsonl_file_has_header_plus_one_line_per_event(tmp_path: Path) -> None:
    """The written file has N+1 lines and the header on line 1.

    Confirms the documented on-disk format so the replay tool (T28) and manual
    debugging can depend on it.
    """
    stream = _sample_stream()
    path = tmp_path / "game.jsonl"
    write_jsonl(stream, path)

    file_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(file_lines) == len(stream.log) + 1

    header_only = EventStream.from_jsonl_lines([file_lines[0]])
    assert header_only.header == stream.header
    assert len(header_only.log) == 0
