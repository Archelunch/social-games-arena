"""Engine game-state model.

Upholds benchmark invariant #1: the engine is the single source of truth, and
every game position is an immutable snapshot. The state types here are frozen
dataclasses, so a snapshot recorded into the append-only event stream (T04)
can never be retroactively edited. State evolves only through pure derivation
helpers that return a new `GameState`, leaving the original untouched.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum


class Phase(StrEnum):
    """The two phases of a round; values match the WEREWOLF_DESIGN.md loop."""

    NIGHT = "night"
    DAY = "day"


@dataclass(frozen=True, slots=True)
class PlayerState:
    """An immutable snapshot of one player.

    `name` is the stable identity and lookup key; `role` is an opaque string so
    the engine stays game-agnostic (Werewolf roles arrive in a later task).
    """

    name: str
    role: str
    alive: bool = True

    def killed(self) -> "PlayerState":
        """Return a copy with `alive=False`; idempotent on an already-dead player."""
        return replace(self, alive=False)


@dataclass(frozen=True, slots=True)
class GameState:
    """An immutable snapshot of the whole game position."""

    players: tuple[PlayerState, ...]
    round: int
    phase: Phase

    @classmethod
    def initial(cls, players: Sequence[tuple[str, str]]) -> "GameState":
        """Build the canonical start position: all players alive, round 1, NIGHT.

        `players` is an ordered sequence of `(name, role)` pairs. Duplicate
        names are rejected because `name` is the lookup and observation-routing
        key — a duplicate would misroute private events.
        """
        names = [name for name, _ in players]
        if len(names) != len(set(names)):
            raise ValueError("player names must be unique; duplicate name(s) given")
        built = tuple(PlayerState(name=name, role=role) for name, role in players)
        return cls(players=built, round=1, phase=Phase.NIGHT)

    def player(self, name: str) -> PlayerState:
        """Return the player with `name`; raise `KeyError` on miss (fail loud)."""
        for p in self.players:
            if p.name == name:
                return p
        raise KeyError(name)

    def alive_players(self) -> tuple[PlayerState, ...]:
        """Return only the living players, in order."""
        return tuple(p for p in self.players if p.alive)

    def alive_names(self) -> tuple[str, ...]:
        """Return the names of the living players, in order."""
        return tuple(p.name for p in self.players if p.alive)

    def is_alive(self, name: str) -> bool:
        """Return whether the named player is alive."""
        return self.player(name).alive

    def with_player_killed(self, name: str) -> "GameState":
        """Return a new state with only the named player killed.

        Raise `KeyError` if `name` is unknown; every other player is untouched.
        """
        self.player(name)  # fail loud on an unknown target before deriving
        players = tuple(p.killed() if p.name == name else p for p in self.players)
        return replace(self, players=players)

    def with_phase(self, phase: Phase) -> "GameState":
        """Return a new state with `phase` replaced."""
        return replace(self, phase=phase)

    def advanced_round(self) -> "GameState":
        """Return a new state with `round` incremented by one."""
        return replace(self, round=self.round + 1)
