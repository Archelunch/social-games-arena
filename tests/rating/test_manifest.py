"""Tests for `RunManifest` — the per-run provenance record (T24).

The manifest is game-agnostic run metadata written next to the event/trajectory
JSONL: which model sat in which seat, the sampling config, the git revision, and
the outcome. It is the seat->model record cross-model rating (T26+) needs, since
model identity otherwise lives only inside `trajectories.jsonl > lm_calls[].model`
and a seat that never calls an LM has none. These tests pin the value type's
immutability, deterministic serialization, and fail-loud read-back.
"""

from __future__ import annotations

import json

import pytest

from social_deduction_bench.rating.manifest import RunManifest, read_json, write_json

_PLAYERS: tuple[tuple[str, str], ...] = (
    ("Alice", "seer"),
    ("Bob", "doctor"),
    ("Carol", "villager"),
    ("Dave", "werewolf"),
)


def _manifest(**overrides: object) -> RunManifest:
    base: dict[str, object] = {
        "game_id": "20260522-141757-werewolf",
        "seed": 69,
        "players": _PLAYERS,
        "models": (
            ("Alice", "openrouter/qwen/qwen3.5-9b"),
            ("Bob", "openrouter/qwen/qwen3.5-9b"),
            ("Carol", "openrouter/qwen/qwen3.5-9b"),
            ("Dave", "openrouter/qwen/qwen3.5-9b"),
        ),
        "model_arg": "openrouter/qwen/qwen3.5-9b",
        "temperature": 0.7,
        "max_tokens": 8000,
        "max_iters": 6,
        "reasoning": False,
        "git_sha": "abc1234",
        "created_at": "2026-05-22T14:17:57+00:00",
        "winner": "werewolves",
        "rounds": 2,
    }
    base.update(overrides)
    return RunManifest(**base)  # type: ignore[arg-type]


def test_run_manifest_is_frozen() -> None:
    manifest = _manifest()
    with pytest.raises((AttributeError, TypeError)):
        manifest.seed = 1  # type: ignore[misc]


def test_two_manifests_with_same_fields_are_equal() -> None:
    assert _manifest() == _manifest()


def test_models_are_stored_sorted_by_seat() -> None:
    # Reason: the CLI may build the seat->model pairs in roster order, but a
    # byte-stable manifest (invariant #4 spirit) must serialize identically
    # regardless of input order. Sorting at construction is the anchor.
    out_of_order = _manifest(
        models=(
            ("Dave", "model-d"),
            ("Alice", "model-a"),
            ("Carol", "model-c"),
            ("Bob", "model-b"),
        )
    )
    assert out_of_order.models == (
        ("Alice", "model-a"),
        ("Bob", "model-b"),
        ("Carol", "model-c"),
        ("Dave", "model-d"),
    )


def test_players_keep_roster_order() -> None:
    # Players are the seating order (mirrors the stream header), not sorted.
    assert _manifest().players == _PLAYERS


def test_json_round_trip_preserves_all_fields() -> None:
    manifest = _manifest()
    assert RunManifest.from_json_dict(manifest.to_json_dict()) == manifest


def test_to_json_dict_is_byte_stable_for_equal_manifests() -> None:
    a = json.dumps(_manifest().to_json_dict(), sort_keys=True)
    b = json.dumps(_manifest().to_json_dict(), sort_keys=True)
    assert a == b


def test_write_then_read_round_trips_through_disk(tmp_path) -> None:
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    write_json(manifest, path)
    assert read_json(path) == manifest


def test_git_sha_none_is_tolerated() -> None:
    # A run outside a git checkout (or with git missing) records git_sha=None;
    # the manifest must still round-trip.
    manifest = _manifest(git_sha=None)
    assert manifest.git_sha is None
    assert RunManifest.from_json_dict(manifest.to_json_dict()) == manifest


def test_from_json_dict_missing_key_fails_loud() -> None:
    raw = _manifest().to_json_dict()
    del raw["winner"]
    with pytest.raises(KeyError):
        RunManifest.from_json_dict(raw)


def test_from_json_dict_malformed_players_entry_fails_loud() -> None:
    raw = _manifest().to_json_dict()
    raw["players"] = [["Alice"]]  # not a (name, role) pair
    with pytest.raises(ValueError, match="pair"):
        RunManifest.from_json_dict(raw)


def test_from_json_dict_malformed_models_entry_fails_loud() -> None:
    raw = _manifest().to_json_dict()
    raw["models"] = [["Alice", "m", "extra"]]  # not a (seat, model) pair
    with pytest.raises(ValueError, match="pair"):
        RunManifest.from_json_dict(raw)


def test_read_malformed_json_fails_loud(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed manifest"):
        read_json(path)
