"""Per-run provenance record (T24).

The `RunManifest` is game-agnostic run metadata written next to the event and
trajectory JSONL: which model sat in which seat, the sampling config, the git
revision, and the outcome. It is the seat->model record cross-model rating
(T26+) needs, since model identity otherwise lives only inside
`trajectories.jsonl > lm_calls[].model` — and a seat that never calls an LM has
none.

Serialization mirrors `engine/events.py`: a frozen value type with deterministic
`to_json_dict` / `from_json_dict`, written with `sort_keys=True` so two equal
manifests serialize byte-identically (invariant #4 spirit). Read-back is
fail-loud — a missing key raises `KeyError`, a malformed seat/model pair raises
`ValueError` — never a silent corrupt manifest.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _coerce_pairs(raw: object) -> tuple[tuple[str, str], ...]:
    """Coerce an iterable of (a, b) entries to a tuple of (str, str) pairs, fail-loud.

    Each entry must be a 2-element list/tuple of strings; a malformed entry raises
    `ValueError` rather than unpacking into a corrupt pair (mirrors
    `StreamHeader.from_json_dict`'s validation).
    """
    pairs: list[tuple[str, str]] = []
    for entry in raw:  # type: ignore[union-attr]
        if not (isinstance(entry, list | tuple) and len(entry) == 2 and all(isinstance(x, str) for x in entry)):
            raise ValueError(f"manifest pair entry must be a 2-element pair of strings, got {entry!r}")
        pairs.append((entry[0], entry[1]))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """One game's run-level provenance: seating, models, sampling config, outcome.

    `players` is the `(name, role)` roster in seating order (mirrors the stream
    header) — kept as given, never sorted. `models` is the `(seat, model)` map
    stored sorted at construction so a byte-stable manifest serializes
    identically regardless of input order.
    """

    game_id: str
    seed: int
    players: tuple[tuple[str, str], ...]
    models: tuple[tuple[str, str], ...]
    model_arg: str
    temperature: float
    max_tokens: int
    max_iters: int
    reasoning: bool
    git_sha: str | None
    created_at: str
    winner: str
    rounds: int

    def __post_init__(self) -> None:
        """Coerce `players`/`models` to tuples-of-tuples; store `models` sorted."""
        object.__setattr__(self, "players", _coerce_pairs(self.players))
        object.__setattr__(self, "models", tuple(sorted(_coerce_pairs(self.models))))

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-ready dict; players/models become lists of [a, b] lists."""
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "players": [[name, role] for name, role in self.players],
            "models": [[seat, model] for seat, model in self.models],
            "model_arg": self.model_arg,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_iters": self.max_iters,
            "reasoning": self.reasoning,
            "git_sha": self.git_sha,
            "created_at": self.created_at,
            "winner": self.winner,
            "rounds": self.rounds,
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, object]) -> "RunManifest":
        """Rebuild a `RunManifest` from its JSON dict, fail-loud on corruption.

        A missing key raises `KeyError` (direct indexing, like
        `Event.from_json_dict`); a malformed players/models pair raises
        `ValueError`. `git_sha=None` round-trips.
        """
        return cls(
            game_id=raw["game_id"],  # type: ignore[arg-type]
            seed=raw["seed"],  # type: ignore[arg-type]
            players=_coerce_pairs(raw["players"]),
            models=_coerce_pairs(raw["models"]),
            model_arg=raw["model_arg"],  # type: ignore[arg-type]
            temperature=raw["temperature"],  # type: ignore[arg-type]
            max_tokens=raw["max_tokens"],  # type: ignore[arg-type]
            max_iters=raw["max_iters"],  # type: ignore[arg-type]
            reasoning=raw["reasoning"],  # type: ignore[arg-type]
            git_sha=raw["git_sha"],  # type: ignore[arg-type]
            created_at=raw["created_at"],  # type: ignore[arg-type]
            winner=raw["winner"],  # type: ignore[arg-type]
            rounds=raw["rounds"],  # type: ignore[arg-type]
        )


def write_json(manifest: RunManifest, path: Path) -> None:
    """Write `manifest` to `path` as UTF-8 JSON, deterministic (sorted keys, indented)."""
    path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> RunManifest:
    """Read a UTF-8 JSON file and rebuild the `RunManifest`, fail-loud on corruption.

    A malformed file raises `ValueError` (wrapped with a stable message, mirroring
    the engine/trajectory stream readers); a structurally-invalid manifest fails
    inside `from_json_dict`.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed manifest JSON in {path}") from e
    return RunManifest.from_json_dict(raw)
