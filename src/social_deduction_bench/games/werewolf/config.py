"""Werewolf benchmark configuration: the 7-player default and its private channels.

This pins the benchmark's default game size and role counts (WEREWOLF_DESIGN.md
§2): wrong counts change game balance and would invalidate cross-game leaderboard
results. It also declares `PRIVATE_EVENT_TYPES`, the set the engine's private-event
guard consumes — an omission silently re-opens the hidden-state leak invariant #2
exists to prevent. `default_role_multiset` expands the counts deterministically so
seeded role assignment has a stable input to shuffle (invariant #4).
"""

from types import MappingProxyType
from typing import Final

from social_deduction_bench.games.werewolf.roles import Role

DEFAULT_PLAYER_COUNT = 7

# Pinned 2 / 1 / 1 / 3 balance for the 7-player benchmark default. Read-only so the
# counts cannot drift away from the rated game balance.
DEFAULT_ROLE_COUNTS: MappingProxyType[Role, int] = MappingProxyType(
    {
        Role.WEREWOLF: 2,
        Role.SEER: 1,
        Role.DOCTOR: 1,
        Role.VILLAGER: 3,
    }
)

# The six Werewolf event types that must be routed to specific recipients only.
# This frozenset is the input to the engine's private-event guard; a `frozenset`
# so it cannot be mutated mid-game and drop a type from the guard's coverage.
PRIVATE_EVENT_TYPES: frozenset[str] = frozenset(
    {"seer_inspect", "werewolf_chat", "doctor_protect", "bid", "tool_rejected", "kill_ballots"}
)

# Number of speaking slots awarded each day's discussion (WEREWOLF_DESIGN.md §4).
# The Werewolf Arena baseline for 7-player games; `resolve_discussion` clamps to
# the number of bidders when fewer than K are present. `Final` matches the
# immutability discipline of the adjacent `MappingProxyType` / `frozenset`
# guards so Pyrefly flags any reassignment.
K_DISCUSSION_SLOTS: Final[int] = 3

# Upper inclusive bound on a `submit_bid` amount (WEREWOLF_DESIGN.md §6.1's `0..N`).
# Bounded so an agent cannot grief a rated game with an unbounded bid (RNG cost,
# prompt-token bloat, integer-overflow surface); the lower bound 0 stays from T15.
MAX_BID: Final[int] = 100


def default_role_multiset() -> tuple[str, ...]:
    """Expand `DEFAULT_ROLE_COUNTS` into one role string per seat.

    Returns the role *string values* in a fixed, deterministic order (the
    `DEFAULT_ROLE_COUNTS` insertion order) — seeded role assignment shuffles this
    tuple, so the tuple itself must be stable or replay would diverge before the
    shuffle runs (invariant #4).
    """
    return tuple(role.value for role, count in DEFAULT_ROLE_COUNTS.items() for _ in range(count))
