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
    the engine stays game-agnostic. `bid_budget` is an opaque per-player integer
    resource the engine does not interpret — the Werewolf layer uses it as the
    depleting speaking-bid pool, seeded at game start and spent by the discussion
    resolver. It defaults to `0` so the engine carries no game-specific number;
    games that use it pass a budget to `GameState.initial`.
    """

    name: str
    role: str
    alive: bool = True
    bid_budget: int = 0

    def killed(self) -> "PlayerState":
        """Return a copy with `alive=False`; idempotent on an already-dead player."""
        return replace(self, alive=False)

    def spend(self, amount: int) -> "PlayerState":
        """Return a copy with `amount` deducted from `bid_budget`, clamped at 0.

        Clamping means a deduction can never produce a negative budget even if a
        caller bypasses the tool-layer validation that normally guards
        `amount <= bid_budget`.
        """
        return replace(self, bid_budget=max(0, self.bid_budget - amount))


@dataclass(frozen=True, slots=True)
class GameState:
    """An immutable snapshot of the whole game position."""

    players: tuple[PlayerState, ...]
    round: int
    phase: Phase

    @classmethod
    def initial(cls, players: Sequence[tuple[str, str]], *, bid_budget: int = 0) -> "GameState":
        """Build the canonical start position: all players alive, round 1, NIGHT.

        `players` is an ordered sequence of `(name, role)` pairs. Duplicate
        names are rejected because `name` is the lookup and observation-routing
        key — a duplicate would misroute private events. `bid_budget` seeds every
        player's opaque resource pool (the Werewolf layer passes its configured
        speaking budget); it defaults to `0` so the engine holds no game number.
        """
        names = [name for name, _ in players]
        if len(names) != len(set(names)):
            raise ValueError("player names must be unique; duplicate name(s) given")
        built = tuple(PlayerState(name=name, role=role, bid_budget=bid_budget) for name, role in players)
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

    def with_bid_spent(self, name: str, amount: int) -> "GameState":
        """Return a new state with only the named player's `bid_budget` reduced.

        Raise `KeyError` if `name` is unknown; every other player is untouched.
        The deduction is clamped at 0 by `PlayerState.spend`.
        """
        self.player(name)  # fail loud on an unknown name before deriving
        players = tuple(p.spend(amount) if p.name == name else p for p in self.players)
        return replace(self, players=players)

    def with_phase(self, phase: Phase) -> "GameState":
        """Return a new state with `phase` replaced."""
        return replace(self, phase=phase)

    def advanced_round(self) -> "GameState":
        """Return a new state with `round` incremented by one."""
        return replace(self, round=self.round + 1)
