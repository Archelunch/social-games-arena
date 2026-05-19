"""Werewolf role catalogue and the role->faction map.

The engine keeps a player's `role` as an opaque string so it stays game-agnostic;
this module is the Werewolf game layer that pins that string set. The role->faction
map is the basis of every win-condition check, so it is the correctness anchor a
misassigned faction would silently corrupt. `faction_of` fails loud on an
unrecognized role rather than defaulting to a faction.
"""

from enum import StrEnum
from types import MappingProxyType


class Faction(StrEnum):
    """The two sides a Werewolf game can be won by."""

    WEREWOLVES = "werewolves"
    VILLAGERS = "villagers"


class Role(StrEnum):
    """The four roles dealt in the benchmark's Werewolf config."""

    WEREWOLF = "werewolf"
    SEER = "seer"
    DOCTOR = "doctor"
    VILLAGER = "villager"


# The closed role->faction map every win check reads. Read-only so it cannot be
# edited mid-game and silently flip a game's winner.
_ROLE_FACTIONS: MappingProxyType[Role, Faction] = MappingProxyType(
    {
        Role.WEREWOLF: Faction.WEREWOLVES,
        Role.SEER: Faction.VILLAGERS,
        Role.DOCTOR: Faction.VILLAGERS,
        Role.VILLAGER: Faction.VILLAGERS,
    }
)


def faction_of(role: str) -> Faction:
    """Return the `Faction` a role belongs to.

    Accepts the engine's opaque `str` role (not just a `Role`) so call sites need
    no conversion. An unknown role raises `ValueError` — silently filing it under a
    faction would corrupt the win check.
    """
    try:
        return _ROLE_FACTIONS[Role(role)]
    except ValueError as e:
        raise ValueError(f"unknown role {role!r}") from e
