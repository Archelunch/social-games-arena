"""Tests for Werewolf bidding-based speech ordering (T18).

`resolve_discussion` picks the top-`K_DISCUSSION_SLOTS` bidders in descending
bid order; bid ties are broken by the engine seed (WEREWOLF_DESIGN.md sections 4-5).
These tests encode the rules and the benchmark invariants:

- #1 — `resolve_discussion` is a pure derivation; it never mutates input state.
- #4 — the only stochastic point, the bid tie-break, derives entirely from the
  engine seed: same seed + same bids -> same speaker order.

Resolution takes *decided* bids as input — agent deliberation and tool
validation are M3/M4 concerns, out of scope here. Like the other resolvers,
*voter*-legality (the bidder is alive) is enforced; *target* legality has no
analogue (a bid has no player target).
"""

from itertools import permutations

import pytest

from social_deduction_bench.engine import GameRNG, GameState, Phase
from social_deduction_bench.games.werewolf.discussion import (
    BiddingActions,
    DiscussionResult,
    resolve_discussion,
)

ROSTER = (
    ("Wolf1", "werewolf"),
    ("Wolf2", "werewolf"),
    ("Seer", "seer"),
    ("Doc", "doctor"),
    ("Vil1", "villager"),
    ("Vil2", "villager"),
    ("Vil3", "villager"),
)


def _day_state() -> GameState:
    """A fresh 7-player game advanced into the DAY phase."""
    return GameState.initial(ROSTER).with_phase(Phase.DAY)


def test_returns_top_k_winners_in_descending_bid_order() -> None:
    """With more than K distinct bids, only the top K win — in descending bid order.

    The core §4 rule: bidding decides *who* speaks and *in what order*. Here
    `Vil3` bids zero and would still be legal (zero is the floor, not invalid)
    but is excluded only because K is full.
    """
    state = _day_state()
    actions = BiddingActions(bids={"Wolf1": 5, "Seer": 3, "Doc": 1, "Vil3": 0})

    result = resolve_discussion(state, actions, GameRNG(0))

    assert result.speakers == ("Wolf1", "Seer", "Doc")


def test_returns_all_bidders_when_fewer_than_k() -> None:
    """Fewer bidders than slots returns every bidder, still in bid order.

    A short discussion is a legal one; the resolver must clamp to the number
    of bidders rather than fail loud or pad with empty slots.
    """
    state = _day_state()
    actions = BiddingActions(bids={"Wolf1": 5, "Seer": 3})

    result = resolve_discussion(state, actions, GameRNG(0))

    assert result.speakers == ("Wolf1", "Seer")


def test_zero_bid_can_win_a_slot() -> None:
    """A bid of 0 is a *low* bid, not an *invalid* one — and can still win.

    §5 says bids range `0..N` and §4 says every alive agent submits one; a
    table of three all-zero bidders with K=3 must seat all three. Pins that
    the resolver tallies zeros instead of silently dropping them.
    """
    state = _day_state()
    actions = BiddingActions(bids={"Wolf1": 0, "Seer": 0, "Doc": 0})

    result = resolve_discussion(state, actions, GameRNG(0))

    assert set(result.speakers) == {"Wolf1", "Seer", "Doc"}
    assert len(result.speakers) == 3


def test_tie_is_broken_by_the_seed() -> None:
    """An all-tied bid is resolved by the engine seed, not arbitrarily.

    With three bidders tied and K=2, the seed picks which two speak. The seed
    pair (0, 1) was hand-verified against `GameRNG` to diverge — there are
    only six legal ordered pairs for K=2 from {Wolf1, Seer, Doc}, so colliding
    seeds would silently weaken this test. The pair is pinned here; a future
    RNG change that makes them collide should be caught by this assertion.
    The tie-break derives from the seed (invariant #4).
    """
    state = _day_state()
    actions = BiddingActions(bids={"Wolf1": 5, "Seer": 5, "Doc": 5})

    seed0 = resolve_discussion(state, actions, GameRNG(0)).speakers[:2]
    seed1 = resolve_discussion(state, actions, GameRNG(1)).speakers[:2]

    legal_pairs = set(permutations(("Wolf1", "Seer", "Doc"), 2))
    assert seed0 in legal_pairs
    assert seed1 in legal_pairs
    assert seed0 != seed1  # the seed actually changes which pair speaks


def test_tie_break_is_seed_deterministic() -> None:
    """Same seed + same tied bids -> identical speakers, every time.

    Invariant #4: the tie-break must be replayable. Two independent
    `GameRNG(7)` runs over identical bids must pick the same speaker tuple,
    or recorded games would diverge on replay.
    """
    state = _day_state()
    actions = BiddingActions(bids={"Wolf1": 5, "Seer": 5, "Doc": 5})

    first = resolve_discussion(state, actions, GameRNG(7)).speakers
    second = resolve_discussion(state, actions, GameRNG(7)).speakers

    assert first == second


def test_mixed_ties_randomize_only_the_tie_group() -> None:
    """A unique top bid always wins slot 0; the RNG only governs scopes of ties.

    Bids {Wolf1: 9, Seer: 5, Doc: 5, Vil1: 1} with K=3: Wolf1's 9 is the unique
    top, so slot 0 is `Wolf1` regardless of seed. Slots 1 and 2 are the tied
    {Seer, Doc} in some seed-dependent order. Vil1's 1 is excluded — K fills
    before the bottom group is reached. Pins that the RNG never scrambles the
    *between-tie* order.
    """
    state = _day_state()
    actions = BiddingActions(bids={"Wolf1": 9, "Seer": 5, "Doc": 5, "Vil1": 1})

    result = resolve_discussion(state, actions, GameRNG(0))

    assert result.speakers[0] == "Wolf1"
    assert set(result.speakers[1:]) == {"Seer", "Doc"}
    assert "Vil1" not in result.speakers


def test_empty_bids_returns_empty_speakers() -> None:
    """A day where no one bids returns no speakers — a legal silent day.

    Not an error: a `BiddingActions` with zero bids resolves to an empty
    speaker tuple. The loop turns this into a zero-speech discussion.
    """
    state = _day_state()
    actions = BiddingActions(bids={})

    result = resolve_discussion(state, actions, GameRNG(0))

    assert result.speakers == ()
    assert isinstance(result, DiscussionResult)


def test_resolve_discussion_does_not_mutate_inputs() -> None:
    """`resolve_discussion` is a pure derivation — state and bids are untouched.

    Invariant #1: an earlier recorded position and the decided-bid mapping
    must stay intact after resolution, or replay is corrupted. Asserts whole
    structural equality (mirrors `test_day.py`).
    """
    state = _day_state()
    state_before = _day_state()  # independent, structurally-equal snapshot
    bids = {"Wolf1": 5, "Seer": 3, "Doc": 1}
    bids_before = dict(bids)
    actions = BiddingActions(bids=bids)

    resolve_discussion(state, actions, GameRNG(0))

    assert state == state_before
    assert bids == bids_before


def test_resolve_discussion_rejects_a_non_day_phase() -> None:
    """Resolving a discussion during the NIGHT phase fails loud.

    Discussion outside the day phase is a caller bug — silently resolving
    it would corrupt the round's event ordering.
    """
    night_state = GameState.initial(ROSTER)
    actions = BiddingActions(bids={"Wolf1": 5, "Seer": 3})

    with pytest.raises(ValueError, match="day phase"):
        resolve_discussion(night_state, actions, GameRNG(0))


def test_resolve_discussion_rejects_a_bid_from_a_dead_player() -> None:
    """A bid cast by a dead player fails loud — the referee rejects it.

    The day discussion is "every *alive* player submits a bid"; a dead bidder
    left in the pool could win a slot they cannot use. The engine-as-referee
    must reject the ineligible bid, not trust the caller to have filtered the
    dead. Mirrors the `resolve_day` / `resolve_night` voter-legality guard.
    """
    state = GameState.initial(ROSTER).with_player_killed("Vil3").with_phase(Phase.DAY)
    actions = BiddingActions(bids={"Wolf1": 5, "Vil3": 3})

    with pytest.raises(ValueError, match="not alive"):
        resolve_discussion(state, actions, GameRNG(0))


def test_resolve_discussion_rejects_a_bid_from_an_unknown_player() -> None:
    """A bid from a name that is not on the roster fails loud.

    The voter-alive guard relies on `state.alive_names()`; an unknown name is
    not in that set and so is caught by the same `not alive` rejection. Pins
    the behavior so a future refactor (e.g. checking `state.player(...).alive`
    directly) does not silently let unknown bidders through and seat a phantom
    speaker.
    """
    state = _day_state()
    actions = BiddingActions(bids={"Wolf1": 5, "Phantom": 3})

    with pytest.raises(ValueError, match="not alive"):
        resolve_discussion(state, actions, GameRNG(0))
