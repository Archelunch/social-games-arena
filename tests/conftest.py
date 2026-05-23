"""Shared pytest fixtures.

`write_game_dir` writes a minimal but valid Werewolf game directory (events +
trajectories + manifest) into a run dir, the on-disk shape the metric extractor and
the static-site builder read. `two_game_run` composes two cross-play games into a
ready-made run directory. The manifest deliberately carries a `git_sha` and
`created_at` so tests can prove the public site payload never leaks them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from social_deduction_bench.agents.trajectory import TrajectoryStream
from social_deduction_bench.agents.trajectory import write_jsonl as write_trajectories
from social_deduction_bench.engine import EventLog, Phase
from social_deduction_bench.engine import write_jsonl as write_events
from social_deduction_bench.engine.events import EventStream, StreamHeader
from social_deduction_bench.games.werewolf.events import EXILE_RESOLVED, GAME_OVER, KILL_RESOLVED
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.rating.manifest import RunManifest
from social_deduction_bench.rating.manifest import write_json as write_manifest

GameDirWriter = Callable[..., Path]

_PLAYERS: tuple[tuple[str, str], ...] = (
    ("Alice", Role.WEREWOLF.value),
    ("Bob", Role.WEREWOLF.value),
    ("Carol", Role.SEER.value),
    ("Dave", Role.DOCTOR.value),
    ("Eve", Role.VILLAGER.value),
)


@pytest.fixture
def write_game_dir() -> GameDirWriter:
    """Return a factory that writes one valid (or torn) game dir under a run dir."""

    def _write(
        run: Path,
        game_id: str,
        *,
        wolf_model: str,
        village_model: str,
        winner: str,
        with_game_over: bool = True,
    ) -> Path:
        models = {
            "Alice": wolf_model,
            "Bob": wolf_model,
            "Carol": village_model,
            "Dave": village_model,
            "Eve": village_model,
        }
        d = run / game_id
        d.mkdir(parents=True)

        log = EventLog()
        log.append(round=1, phase=Phase.NIGHT, type=KILL_RESOLVED, payload={"victim": "Carol"}, recipients=())
        log.append(
            round=1,
            phase=Phase.DAY,
            type=EXILE_RESOLVED,
            payload={"ballots": {}, "exiled": "Alice"},  # a wolf -> a correct exile
            recipients=(),
        )
        if with_game_over:  # a torn / in-progress dir has no GAME_OVER yet
            log.append(round=1, phase=Phase.DAY, type=GAME_OVER, payload={"winner": winner}, recipients=())
        header = StreamHeader(seed=7, game_id=game_id, players=_PLAYERS)
        write_events(EventStream(header=header, log=log), d / "events.jsonl")
        write_trajectories(TrajectoryStream(header=header, trajectories=()), d / "trajectories.jsonl")
        write_manifest(
            RunManifest(
                game_id=game_id,
                seed=7,
                players=_PLAYERS,
                models=tuple(sorted(models.items())),
                model_arg="test",
                temperature=1.0,
                max_tokens=8000,
                max_iters=9,
                reasoning=False,
                git_sha="deadbeefcafe",
                created_at="2026-05-23T00:00:00+00:00",
                winner=winner,
                rounds=1,
            ),
            d / "manifest.json",
        )
        return d

    return _write


@pytest.fixture
def two_game_run(tmp_path: Path, write_game_dir: GameDirWriter) -> Path:
    """A run dir with two side-swapped cross-play games (models A and B)."""
    run = tmp_path / "run2"
    write_game_dir(run, "g0000-A-vs-B", wolf_model="A", village_model="B", winner="werewolves")
    write_game_dir(run, "g0001-B-vs-A", wolf_model="B", village_model="A", winner="villagers")
    return run
