"""Engine event stream.

Upholds benchmark invariant #5: every game is an append-only event stream
(JSONL) that is replayable and debuggable. An `Event` is frozen history — once
the log captures it, it cannot be reordered or edited — and `EventLog` exposes
only `append`, never a way to delete or rewrite a recorded event.

It also carries invariant #2's routing marker: each event names its
`recipients`. An empty tuple means a public broadcast (visible to everyone); a
non-empty tuple means the event is private to exactly those players — the
werewolf pack chat is the multi-recipient case, one private channel shared by
the whole pack. A blank or duplicate recipient — a routing-corruption vector —
is rejected at construction and at read-back, never stored or routed silently.

Because "public" is defined as the empty tuple, this game-agnostic core — which
treats `type` as an opaque string — cannot distinguish an intentional broadcast
from a private event emitted without recipients. The game layer that emits
private event types (seer results, werewolf chat) owns that check.

Serialization is deterministic (fixed key order, default separators) so that,
per invariant #4, two runs of the same seeded game produce byte-identical
transcripts that T08's determinism harness can compare. Recipients are stored
in canonical sorted order so two semantically-equal events serialize identically.
"""

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class Event:
    """One immutable, recorded event in the append-only stream.

    `seq` is the gap-free, log-assigned ordering index. `recipients` is the
    routing marker (invariant #2): an empty tuple is a public broadcast, a
    non-empty tuple is private to exactly those players — the werewolf pack
    chat is the multi-recipient case. `payload` is normalized to an isolated,
    read-only mapping so neither the caller's source dict nor the stored view
    can mutate recorded history.
    """

    seq: int
    round: int
    phase: Phase
    type: str
    payload: Mapping[str, object]
    recipients: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate `recipients`, store them sorted, and isolate the payload.

        Every recipient must be a non-empty `str` and no recipient may repeat —
        a blank entry matches no player and a duplicate would deliver one
        private event twice, both routing-corruption vectors (invariant #2).
        Validated recipients are stored in canonical sorted order so two
        semantically-equal events serialize identically (invariant #4). The
        payload is copied into a `MappingProxyType` so it is read-only and
        decoupled from the caller's dict.
        """
        seen: set[str] = set()
        for recipient in self.recipients:
            if not isinstance(recipient, str) or not recipient:
                raise ValueError(f"every recipient must be a non-empty str, got {recipient!r}")
            if recipient in seen:
                raise ValueError(f"duplicate recipient {recipient!r}")
            seen.add(recipient)
        object.__setattr__(self, "recipients", tuple(sorted(self.recipients)))
        copied = dict(self.payload)
        for value in copied.values():
            _check_json_payload_value(value)
        object.__setattr__(self, "payload", MappingProxyType(copied))

    @property
    def is_public(self) -> bool:
        """Return whether the event is a public broadcast (has no recipients)."""
        return not self.recipients

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain, JSON-ready dict; enum fields become their string values."""
        return {
            "seq": self.seq,
            "round": self.round,
            "phase": self.phase.value,
            "type": self.type,
            "payload": dict(self.payload),
            "recipients": list(self.recipients),
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, object]) -> "Event":
        """Rebuild an `Event` from its JSON dict.

        `Phase(...)` raises `ValueError` on an unknown value, and
        `__post_init__` re-validates `recipients` — a corrupt or tampered
        record fails loud rather than loading silently. `seq` and `round` are
        type-checked so a tampered string does not slip through to a misleading
        "non-contiguous seq" message. `recipients` must be a list: a bare JSON
        string would otherwise splay into one recipient per character.
        """
        for int_field in ("seq", "round"):
            if not isinstance(raw[int_field], int):
                raise ValueError(f"event '{int_field}' must be an int, got {type(raw[int_field]).__name__}")
        if not isinstance(raw["recipients"], list):
            raise ValueError(f"event 'recipients' must be a list, got {type(raw['recipients']).__name__}")
        return cls(
            seq=raw["seq"],  # type: ignore[arg-type]
            round=raw["round"],  # type: ignore[arg-type]
            phase=Phase(raw["phase"]),  # type: ignore[arg-type]
            type=raw["type"],  # type: ignore[arg-type]
            payload=raw["payload"],  # type: ignore[arg-type]
            recipients=tuple(raw["recipients"]),
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
        recipients: tuple[str, ...] = (),
    ) -> Event:
        """Build, store, and return an `Event` with the next contiguous `seq`.

        Callers never supply `seq`; it is `len(self)` so the index is gap-free
        and strictly increasing. A `None` payload becomes an empty mapping.
        An empty `recipients` tuple makes the event a public broadcast.
        """
        event = Event(
            seq=len(self._events),
            round=round,
            phase=phase,
            type=type,
            payload=payload if payload is not None else {},
            recipients=recipients,
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
        sequence `0, 1, 2, ...`. Unknown enum values and corrupt `recipients`
        fail inside `Event`.
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
                recipients=event.recipients,
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
