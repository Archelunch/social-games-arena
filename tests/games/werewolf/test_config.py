"""Tests for the Werewolf game configuration (T09).

These pin the benchmark's 7-player default (WEREWOLF_DESIGN.md §2): wrong role
counts change game balance and would invalidate every leaderboard result. They
also pin `PRIVATE_EVENT_TYPES` — the declared set the engine's private-event
guard (T11) consumes. An omission from that set silently re-opens the
hidden-state leak invariant #2 exists to prevent.
"""

from collections import Counter

from social_deduction_bench.engine import GameState
from social_deduction_bench.games.werewolf.config import (
    BID_BUDGET,
    DEFAULT_PLAYER_COUNT,
    DEFAULT_ROLE_COUNTS,
    K_DISCUSSION_SLOTS,
    MAX_BID,
    PRIVATE_EVENT_TYPES,
    REACTION_MAX_TOKENS,
    default_role_multiset,
)
from social_deduction_bench.games.werewolf.roles import Role


def test_default_config_is_seven_players() -> None:
    """The benchmark default is 7 players (WEREWOLF_DESIGN.md §2).

    7 is the pinned balance point — long enough that memory/planning matter,
    short enough to stay cheap. The leaderboard is only comparable across games
    of the same size.
    """
    assert DEFAULT_PLAYER_COUNT == 7


def test_default_role_counts_are_two_wolves_one_seer_one_doctor_three_villagers() -> None:
    """Role counts are pinned to 2 / 1 / 1 / 3 (WEREWOLF_DESIGN.md §2).

    These exact counts set the game's balance; changing one (e.g. 3 werewolves)
    is a different game and would silently invalidate cross-game ratings.
    """
    assert DEFAULT_ROLE_COUNTS == {
        Role.WEREWOLF: 2,
        Role.SEER: 1,
        Role.DOCTOR: 1,
        Role.VILLAGER: 3,
    }


def test_default_role_multiset_totals_the_declared_player_count() -> None:
    """The role multiset has exactly one entry per seat.

    A drift between `DEFAULT_PLAYER_COUNT` and the sum of role counts would
    leave a seat unassigned or a role undealt in T10 — a count mismatch must be
    impossible by construction.
    """
    assert len(default_role_multiset()) == DEFAULT_PLAYER_COUNT


def test_default_role_multiset_contents_match_the_role_counts() -> None:
    """The multiset contains each role exactly `DEFAULT_ROLE_COUNTS` times.

    Pins that the expansion places every declared role the right number of
    times — a dealt multiset missing a werewolf would make the game unwinnable
    for villagers.
    """
    assert Counter(default_role_multiset()) == Counter(
        {role.value: count for role, count in DEFAULT_ROLE_COUNTS.items()}
    )


def test_default_role_multiset_is_deterministic() -> None:
    """Two calls return the identical, order-stable multiset.

    Role assignment (T10) derives all of its randomness from the seeded RNG;
    the multiset it shuffles must itself be deterministic, or replay would
    diverge before the shuffle even runs (invariant #4).
    """
    assert default_role_multiset() == default_role_multiset()


def test_private_event_types_are_exactly_the_declared_set() -> None:
    """`PRIVATE_EVENT_TYPES` is exactly the six Werewolf private channels.

    This frozenset *is* the input to the engine's private-event guard (T11). An
    omitted type would let that event be emitted with empty recipients —
    broadcasting hidden state and breaking invariant #2. T29 added `bid`
    (per-bidder private), `tool_rejected` (private to the caller), and
    `kill_ballots` (private to the living werewolf pack — the public
    `kill_resolved` carries only the resolved victim).
    """
    assert PRIVATE_EVENT_TYPES == frozenset(
        {"seer_inspect", "werewolf_chat", "doctor_protect", "bid", "tool_rejected", "kill_ballots"}
    )


def test_private_event_types_is_an_immutable_frozenset() -> None:
    """The declared private set is a `frozenset` — it cannot be mutated.

    A mutable set could be edited mid-game, silently dropping a private type
    from the guard's coverage.
    """
    assert isinstance(PRIVATE_EVENT_TYPES, frozenset)


def test_k_discussion_slots_is_three() -> None:
    """`K_DISCUSSION_SLOTS` is pinned at 3 (Werewolf Arena baseline for 7-player games).

    The slot count gates how much signal the day discussion produces; a silent
    bump up or down would shift every rated game's information density. Pinned
    in config too so a drift between code and `WEREWOLF_DESIGN.md` §4 is caught.
    """
    assert K_DISCUSSION_SLOTS == 3


def test_max_bid_is_one_hundred() -> None:
    """`MAX_BID` is pinned at 100, the upper bound on `submit_bid` (§6.1 `0..N`).

    Bounded so an agent cannot grief the rated game with an unbounded bid (RNG
    cost, prompt-token bloat, integer-overflow surface). The value is shared
    config — `submit_bid` reads it — so changing it is one edit.
    """
    assert MAX_BID == 100


def test_bid_budget_is_one_hundred() -> None:
    """`BID_BUDGET` is pinned at 100 — the per-player speaking-bid pool for the
    whole game (§6.1). It is the balance knob for the bid economy; changing it
    alters how often players can outbid across rounds, so it is a single shared
    constant `run_game` seeds into every player's `bid_budget`.
    """
    assert BID_BUDGET == 100


def test_reaction_max_tokens_is_two_fifty_six() -> None:
    """`REACTION_MAX_TOKENS` is pinned at 256 — the output cap for the day
    reaction round (§4).

    Every living player reacts once per day; capping the reaction loop's output
    keeps that extra per-player call cheap and terse. It is a shared knob the
    adapter reads to clone a shorter LM for the reaction loop, so a drift between
    code and `WEREWOLF_DESIGN.md` §4 is caught here.
    """
    assert REACTION_MAX_TOKENS == 256


def test_default_multiset_builds_a_valid_game_state() -> None:
    """The role multiset zips onto names into a `GameState` without raising.

    Pins the contract with the engine (T03): the config's output is exactly the
    `(name, role)` shape `GameState.initial` accepts, with no duplicate names.
    """
    multiset = default_role_multiset()
    roster = [(f"P{i}", role) for i, role in enumerate(multiset)]
    state = GameState.initial(roster)

    assert len(state.players) == DEFAULT_PLAYER_COUNT
    assert tuple(p.role for p in state.players) == multiset
