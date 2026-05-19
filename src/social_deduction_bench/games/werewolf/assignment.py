"""Seeded Werewolf role assignment.

Upholds benchmark invariant #4: the name->role mapping derives entirely from the
seeded `GameRNG`, so the same seed yields the identical assignment and a recorded
game replays from turn one. The roles (not the names) are shuffled, leaving the
transcript roster in caller order with only the role column varying by seed.
"""

from collections.abc import Sequence

from social_deduction_bench.engine import GameRNG
from social_deduction_bench.games.werewolf.config import default_role_multiset
from social_deduction_bench.games.werewolf.roles import Role


def assign_roles(names: Sequence[str], role_multiset: Sequence[str], rng: GameRNG) -> tuple[tuple[str, str], ...]:
    """Deal `role_multiset` onto `names` via the seeded `rng`.

    Shuffles the roles (a new list — `GameRNG.shuffle` is non-mutating) and zips
    them against the fixed-order names. A name/role count mismatch raises
    `ValueError` rather than letting `zip` silently truncate, which would undeal a
    role (e.g. a werewolf) and make the game unwinnable. Each role string is
    checked against `Role` so a typo fails loud here, at deal time, rather than
    surfacing far later as a corrupt win-condition count (T13).
    """
    if len(names) != len(role_multiset):
        raise ValueError(f"{len(names)} players but {len(role_multiset)} roles to deal")

    for role in role_multiset:
        if role not in Role.__members__.values():
            raise ValueError(f"unknown role {role!r}: not a Werewolf role")

    shuffled = rng.shuffle(list(role_multiset))
    return tuple(zip(names, shuffled, strict=True))


def assign_default_roles(names: Sequence[str], rng: GameRNG) -> tuple[tuple[str, str], ...]:
    """Deal the default 7-player role multiset onto `names` via the seeded `rng`."""
    return assign_roles(names, default_role_multiset(), rng)
