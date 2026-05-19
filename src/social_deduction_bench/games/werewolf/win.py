"""Werewolf win-condition checks (WEREWOLF_DESIGN.md §2).

Upholds benchmark invariant #1: the engine owns no win condition. `winner` is a
pure read over an immutable `GameState` snapshot — it inspects the position and
never derives a new one. `is_game_over` is the game's `TerminalCheck`; the
engine's `is_terminal` seam (T06) only relays this verdict, keeping the referee
game-agnostic.

Check order is a correctness invariant. The villager win is tested before the
werewolf parity rule because when the last werewolf dies into a board with zero
wolves, `0 >= 0` parity is ALSO true. Testing parity first would wrongly credit
the werewolves a win at the exact moment they are wiped out, so a board with no
werewolves must always resolve to a villager win.
"""

from enum import StrEnum

from social_deduction_bench.engine import GameState
from social_deduction_bench.games.werewolf.roles import Faction, faction_of


class Winner(StrEnum):
    """The side that won a finished Werewolf game.

    The string values are written into the public `game_over` event payload and
    every transcript, so they are pinned literals — a rename breaks replays.
    """

    WEREWOLVES = "werewolves"
    VILLAGERS = "villagers"


def winner(state: GameState) -> Winner | None:
    """Return the winning side, or `None` while the game is still in progress.

    A pure read over `state` (invariant #1): it never mutates the snapshot.
    Villagers are checked before werewolf parity — see the module docstring for
    why the check order is a correctness invariant.
    """
    alive = state.alive_players()
    wolves = [p for p in alive if faction_of(p.role) is Faction.WEREWOLVES]

    if len(wolves) == 0:
        return Winner.VILLAGERS
    if len(wolves) >= len(alive) - len(wolves):
        return Winner.WEREWOLVES
    return None


def is_game_over(state: GameState) -> bool:
    """Return whether the game has reached a terminal position.

    This is the game's `TerminalCheck`: the engine's `is_terminal` seam (T06)
    consumes it positionally, so a plain single-arg function matches the
    Protocol and lets the engine relay the verdict without owning the rule.
    """
    return winner(state) is not None
