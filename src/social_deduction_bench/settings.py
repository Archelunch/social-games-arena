"""Project settings — single source for environment-variable reads.

Per CLAUDE.md, every `os.getenv` call in the project lives here. Call
sites read from a `Settings` snapshot returned by `load()` rather than
touching the environment directly, so the surface that depends on
deployment state is small enough to grep in one place.

`load()` re-reads the environment on every call (not at import time) so
tests can `monkeypatch.setenv(...)` and observe the change without
reloading the module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    openrouter_api_key: str | None
    smoke_model: str | None


def load() -> Settings:
    """Read the relevant environment variables and return an immutable snapshot."""
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        smoke_model=os.getenv("SDB_SMOKE_MODEL"),
    )
