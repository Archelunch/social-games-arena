"""Tests for `social_deduction_bench.settings`.

The settings module is the *single* source of `os.getenv` reads in the
project (CLAUDE.md: "All config / env vars in one settings module — no
`os.getenv()` elsewhere"). `load()` is a fresh-snapshot factory so tests
can monkeypatch the environment and observe the change immediately.
"""

from __future__ import annotations

import dataclasses

import pytest

from social_deduction_bench import settings


def test_load_returns_none_for_each_field_when_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SDB_SMOKE_MODEL", raising=False)

    cfg = settings.load()

    assert cfg.openrouter_api_key is None
    assert cfg.smoke_model is None


def test_load_reads_openrouter_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins that `OPENROUTER_API_KEY` is the env var the smoke game checks against.

    A regression renaming this key would silently turn the smoke test into a
    permanent skip — caught here, not in production.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-value-1234")

    cfg = settings.load()

    assert cfg.openrouter_api_key == "sk-or-test-value-1234"


def test_load_reads_smoke_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the override env var name; the smoke test falls back to a hardcoded default when unset."""
    monkeypatch.setenv("SDB_SMOKE_MODEL", "anthropic/claude-3.5-haiku")

    cfg = settings.load()

    assert cfg.smoke_model == "anthropic/claude-3.5-haiku"


def test_settings_is_immutable_after_construction() -> None:
    """A `Settings` snapshot must not mutate post-construction.

    Frozen dataclass + slots: a typo'd assignment (`cfg.api_key = ...`) at a
    call site should raise rather than silently drift the value used by
    downstream code.
    """
    cfg = settings.Settings(openrouter_api_key="sk", smoke_model="m")

    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.openrouter_api_key = "tampered"  # type: ignore[misc]


def test_settings_exposes_only_the_declared_fields() -> None:
    """Pin the public field list — adding a field without updating `load()` should fail this.

    A new field added to `Settings` without a matching env-var read in `load()`
    would default to `None` forever; this test catches the omission.
    """
    fields = {f.name for f in dataclasses.fields(settings.Settings)}
    assert fields == {"openrouter_api_key", "smoke_model"}
