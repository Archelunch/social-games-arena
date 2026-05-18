"""Engine event stream.

Upholds benchmark invariant #5: every game is an append-only event stream
(JSONL) that is replayable and debuggable. An `Event` is frozen history — once
the log captures it, it cannot be reordered or edited — and `EventLog` exposes
only `append`, never a way to delete or rewrite a recorded event.

It also carries invariant #2's routing marker: each event is `PUBLIC` or
`PRIVATE`, and a `PRIVATE` event must name its `recipient`. An unaddressed
private event — a hidden-state-leak vector — is rejected at construction and at
read-back, never stored or routed silently.

Serialization is deterministic (fixed key order, default separators) so that,
per invariant #4, two runs of the same seeded game produce byte-identical
transcripts that T08's determinism harness can compare.
"""

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from social_deduction_bench.engine.state import Phase


def _check_json_payload_value(value: object) -> None:
    """Reject any payload value that would not survive the JSONL round-trip.

    Only JSON primitives, lists of allowed values, and string-keyed mappings of
    allowed values are permitted. A `tuple` or `set` would silently change type
    on read-back, breaking invariant #4's lossless round-trip.
    """
    if isinstance(value, bool | int | float | str) or value is None:
        return
    if isinstance(value, Mapping):
        for key, sub_value in value.items():
            if not isinstance(key, str):
                raise ValueError(f"payload mapping keys must be str, got {type(key).__name__}")
            _check_json_payload_value(sub_value)
        return
    if isinstance(value, list):
        for item in value:
            _check_json_payload_value(item)
        return
    raise ValueError(f"payload value {type(value).__name__} is not a JSON primitive")


class Visibility(StrEnum):
    """Who an event is addressed to; drives T05's observation routing.

    `PUBLIC` events broadcast to everyone; `PRIVATE` events reach only their
    named `recipient`.
    """

    PUBLIC = "public"
    PRIVATE = "private"


@dataclass(frozen=True, slots=True)
class Event:
    """One immutable, recorded event in the append-only stream.

    `seq` is the gap-free, log-assigned ordering index. `payload` is normalized
    to an isolated, read-only mapping so neither the caller's source dict nor
    the stored view can mutate recorded history.
    """

    seq: int
    round: int
    phase: Phase
    type: str
    payload: Mapping[str, object]
    visibility: Visibility = Visibility.PUBLIC
    recipient: str | None = None

    def __post_init__(self) -> None:
        """Enforce the routing marker and isolate the payload.

        A `PRIVATE` event must name a `recipient` and a `PUBLIC` one must not —
        either contradiction makes the event unroutable (invariant #2). The
        payload is copied into a `MappingProxyType` so it is read-only and
        decoupled from the caller's dict.
        """
        if self.visibility is Visibility.PRIVATE and self.recipient is None:
            raise ValueError("a PRIVATE event must name a recipient")
        if self.visibility is Visibility.PUBLIC and self.recipient is not None:
            raise ValueError("a PUBLIC event must not name a recipient")
        copied = dict(self.payload)
        for value in copied.values():
            _check_json_payload_value(value)
        object.__setattr__(self, "payload", MappingProxyType(copied))

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-ready dict; enum fields become their string values."""
        return {
            "seq": self.seq,
            "round": self.round,
            "phase": self.phase.value,
            "type": self.type,
            "payload": dict(self.payload),
            "visibility": self.visibility.value,
            "recipient": self.recipient,
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, object]) -> "Event":
        """Rebuild an `Event` from its JSON dict.

        `Phase(...)` and `Visibility(...)` raise `ValueError` on an unknown
        value, and `__post_init__` re-checks the routing marker — a corrupt or
        tampered record fails loud rather than loading silently. `seq` and
        `round` are type-checked so a tampered string does not slip through to
        a misleading "non-contiguous seq" message.
        """
        for int_field in ("seq", "round"):
            if not isinstance(raw[int_field], int):
                raise ValueError(f"event '{int_field}' must be an int, got {type(raw[int_field]).__name__}")
        return cls(
            seq=raw["seq"],  # type: ignore[arg-type]
            round=raw["round"],  # type: ignore[arg-type]
            phase=Phase(raw["phase"]),  # type: ignore[arg-type]
            type=raw["type"],  # type: ignore[arg-type]
            payload=raw["payload"],  # type: ignore[arg-type]
            visibility=Visibility(raw["visibility"]),  # type: ignore[arg-type]
            recipient=raw["recipient"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class StreamHeader:
    """The first line of a transcript: the seed and roster that replay needs.

    `players` is the ordered `(name, role)` roster in the exact shape
    `GameState.initial` accepts, so a transcript is self-contained.
    """

    seed: int
    game_id: str
    players: tuple[tuple[str, str], ...]

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-ready dict; players become a list of [name, role] lists."""
        return {
            "seed": self.seed,
            "game_id": self.game_id,
            "players": [[name, role] for name, role in self.players],
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, object]) -> "StreamHeader":
        """Rebuild a `StreamHeader`; the players list becomes a tuple of tuples.

        Each roster entry must be a `(name, role)` pair of strings; a malformed
        entry fails loud rather than unpacking into a corrupt roster.
        """
        players: list[tuple[str, str]] = []
        for entry in raw["players"]:  # type: ignore[union-attr]
            if not (isinstance(entry, list | tuple) and len(entry) == 2 and all(isinstance(x, str) for x in entry)):
                raise ValueError(f"header 'players' entry must be a (name, role) pair of strings, got {entry!r}")
            players.append((entry[0], entry[1]))
        return cls(
            seed=raw["seed"],  # type: ignore[arg-type]
            game_id=raw["game_id"],  # type: ignore[arg-type]
            players=tuple(players),
        )


class EventLog:
    """A mutable, append-only sequence of `Event`s.

    The log assigns each event its `seq` and exposes no remove/clear/insert —
    append-only is invariant #5's core safety property.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(
        self,
        *,
        round: int,
        phase: Phase,
        type: str,
        payload: Mapping[str, object] | None = None,
        visibility: Visibility = Visibility.PUBLIC,
        recipient: str | None = None,
    ) -> Event:
        """Build, store, and return an `Event` with the next contiguous `seq`.

        Callers never supply `seq`; it is `len(self)` so the index is gap-free
        and strictly increasing. A `None` payload becomes an empty mapping.
        """
        event = Event(
            seq=len(self._events),
            round=round,
            phase=phase,
            type=type,
            payload=payload if payload is not None else {},
            visibility=visibility,
            recipient=recipient,
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[Event, ...]:
        """Return a tuple snapshot; it does not change when the log grows later."""
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)


@dataclass(frozen=True, slots=True)
class EventStream:
    """A complete transcript: a `StreamHeader` plus its `EventLog`.

    Serializes to JSONL — header on line 1, one event per line thereafter —
    and reads back with full validation so a corrupt transcript fails loud.
    """

    header: StreamHeader
    log: EventLog = field(default_factory=EventLog)

    def to_jsonl_lines(self) -> Iterator[str]:
        """Yield the JSONL text: header line, then one line per event in `seq` order.

        `sort_keys=True` and default separators make equal streams serialize to
        byte-identical text (invariant #4). No trailing newline per line.
        """
        yield json.dumps(self.header.to_json_dict(), sort_keys=True)
        for event in self.log.events:
            yield json.dumps(event.to_json_dict(), sort_keys=True)

    @classmethod
    def from_jsonl_lines(cls, lines: Iterable[str]) -> "EventStream":
        """Parse JSONL back into an `EventStream`, validating fail-loud on corruption.

        Rejects empty input (no header), any line that is not valid JSON, and
        any event `seq` that is not the contiguous, strictly increasing
        sequence `0, 1, 2, ...`. Unknown enum values and unaddressed private
        events fail inside `Event`.
        """
        materialized = [line.rstrip("\n") for line in lines]
        if not materialized:
            raise ValueError("empty JSONL input: no header line")

        def _parse(raw_line: str, line_number: int) -> object:
            try:
                return json.loads(raw_line)
            except json.JSONDecodeError as e:
                raise ValueError(f"malformed JSONL: line {line_number} is not valid JSON") from e

        header = StreamHeader.from_json_dict(_parse(materialized[0], 1))  # type: ignore[arg-type]
        log = EventLog()
        for offset, raw_line in enumerate(materialized[1:]):
            event = Event.from_json_dict(_parse(raw_line, offset + 2))  # type: ignore[arg-type]
            if event.seq != offset:
                raise ValueError(f"non-contiguous event seq: expected {offset}, found {event.seq}")
            log.append(
                round=event.round,
                phase=event.phase,
                type=event.type,
                payload=event.payload,
                visibility=event.visibility,
                recipient=event.recipient,
            )
        return cls(header=header, log=log)


def write_jsonl(stream: EventStream, path: Path) -> None:
    """Write `stream` to `path` as UTF-8 JSONL, one line per `to_jsonl_lines` entry."""
    with path.open("w", encoding="utf-8") as handle:
        for line in stream.to_jsonl_lines():
            handle.write(line)
            handle.write("\n")


def read_jsonl(path: Path) -> EventStream:
    """Read a UTF-8 JSONL file and rebuild the `EventStream`, validating on the way."""
    text = path.read_text(encoding="utf-8")
    return EventStream.from_jsonl_lines(text.splitlines())
