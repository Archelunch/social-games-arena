"""Tests for the Werewolf role catalogue and faction map (T09).

The engine keeps a player's `role` as an opaque string (T03) so it stays
game-agnostic; this module is the Werewolf game layer that pins that string
set. The role->faction map is the basis of every win-condition check (T13), so
these tests guard it as a correctness invariant: a misassigned faction would
silently flip a game's winner. Role string values are pinned literals because
they cross into `GameState.role`, JSONL transcripts (T04), and
`ToolRequirement.role` (T07) — a rename would break recorded replays.
"""

import pytest

from social_deduction_bench.games.werewolf.roles import Faction, Role, faction_of


def test_each_role_maps_to_its_expected_faction() -> None:
    """The role->faction map is pinned exactly.

    Werewolf is the sole WEREWOLVES role; Seer, Doctor, and Villager are all
    VILLAGERS. Every win check (T13) reads this map — a single misassignment
    (e.g. Doctor filed under WEREWOLVES) would silently corrupt the result.
    """
    assert faction_of(Role.WEREWOLF) is Faction.WEREWOLVES
    assert faction_of(Role.SEER) is Faction.VILLAGERS
    assert faction_of(Role.DOCTOR) is Faction.VILLAGERS
    assert faction_of(Role.VILLAGER) is Faction.VILLAGERS


def test_role_string_values_are_stable() -> None:
    """Role string values are pinned literals.

    These strings are written into `GameState.role`, every JSONL transcript,
    and `ToolRequirement.role`. A rename would make every recorded game
    un-replayable, so the literals are frozen here.
    """
    assert Role.WEREWOLF.value == "werewolf"
    assert Role.SEER.value == "seer"
    assert Role.DOCTOR.value == "doctor"
    assert Role.VILLAGER.value == "villager"


def test_faction_string_values_are_stable() -> None:
    """Faction string values are pinned literals.

    Faction values reach JSONL via the private `seer_inspect` payload (T11);
    a rename would break replay of recorded seer results.
    """
    assert Faction.WEREWOLVES.value == "werewolves"
    assert Faction.VILLAGERS.value == "villagers"


def test_faction_of_accepts_a_plain_role_string() -> None:
    """`faction_of` accepts the engine's opaque `str` role, not just `Role`.

    The engine hands roles around as plain strings; forcing callers to convert
    to `Role` first would be friction at every call site (T11, T13).
    """
    assert faction_of("werewolf") is Faction.WEREWOLVES
    assert faction_of("villager") is Faction.VILLAGERS


def test_faction_of_rejects_an_unknown_role() -> None:
    """An unknown role fails loud rather than defaulting to a faction.

    Silently filing an unrecognized role under a faction would corrupt the win
    check (T13). A typo or a role from another game must raise immediately.
    """
    with pytest.raises(ValueError, match="unknown role"):
        faction_of("vampire")


def test_exactly_four_roles_and_two_factions_are_defined() -> None:
    """The role and faction sets are closed.

    The 7-player benchmark config (and every win check) assumes exactly these
    four roles and two factions; an extra member added without updating the
    config would desync role counts from the game balance.
    """
    assert set(Role) == {Role.WEREWOLF, Role.SEER, Role.DOCTOR, Role.VILLAGER}
    assert set(Faction) == {Faction.WEREWOLVES, Faction.VILLAGERS}
