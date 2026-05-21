"""Werewolf bidding-based speech ordering (WEREWOLF_DESIGN.md sections 4-5).

`resolve_discussion` derives the day discussion's speaker order from a decided
bid mapping: the top `K_DISCUSSION_SLOTS` bidders speak in descending bid
order. It is a pure derivation that mirrors `resolve_night` / `resolve_day` —
it never mutates the input `GameState` (invariant #1) and returns the chosen
speakers as a frozen `DiscussionResult`.

Bid ties are broken by one seeded `rng.shuffle` per tie group — the single
stochastic point of discussion ordering, derived entirely from the engine seed
(invariant #4) so the same seed plus the same bids yield the same speaker
tuple on replay. The leader list is sorted before the shuffle so the RNG sees
a byte-identical input on every run regardless of bid-dict iteration order.

This resolver does *not* emit events; the loop integration (DSPy agent layer,
T21) is where bids/speeches enter the transcript.
"""

from collections import defaultdict
from dataclasses import dataclass

from social_deduction_bench.engine import GameRNG, GameState, Phase
from social_deduction_bench.games.werewolf.config import K_DISCUSSION_SLOTS


@dataclass(frozen=True, slots=True)
class BiddingActions:
    """The decided bids resolution takes as input.

    Agent deliberation and tool validation are out of scope here — this is an
    already-decided `bidder name -> non-negative amount` mapping. *Bid* legality
    (the amount falls in `[0, MAX_BID]`) belongs to tool-call validation at the
    game-loop boundary, not this resolver. *Voter* legality is different:
    `resolve_discussion` rejects a bid from a non-living player itself, because
    the referee must never seat an ineligible bidder.

    WEREWOLF_DESIGN.md §4 says "every alive agent submits a bid" — the resolver
    accepts an arbitrary subset (or none) because the "every-alive bids"
    invariant is the loop's responsibility, not the resolver's. The agent
    adapter raises `RuntimeError` when a seat fails to commit a bid — a
    silent default to 0 would absorb an agent-side bug, contrary to
    CLAUDE.md rule 11 (fail loud).
    """

    bids: dict[str, int]


@dataclass(frozen=True, slots=True)
class DiscussionResult:
    """The outcome of one resolved discussion.

    `speakers` is the chosen speaking order: the top `K_DISCUSSION_SLOTS`
    bidders, in descending bid order, with seeded tie-breaks within each
    tied-amount group. Length is `min(K_DISCUSSION_SLOTS, len(bids))`.
    """

    speakers: tuple[str, ...]


def resolve_discussion(state: GameState, actions: BiddingActions, rng: GameRNG) -> DiscussionResult:
    """Resolve the day discussion into the seeded speaker order.

    Pure: `state` and `actions.bids` are never mutated (invariant #1). The bid
    tie-break is the only stochastic step and is seeded via `rng` over each
    sorted tied-amount group, so it is replayable (invariant #4). Raises
    `ValueError` outside the day phase, or when any bidder is not a living
    player — the engine-as-referee must reject an ineligible bidder rather
    than seat them. Both are caller bugs.
    """
    if state.phase is not Phase.DAY:
        raise ValueError(f"resolve_discussion requires the day phase, got {state.phase.value}")

    alive = set(state.alive_names())
    illegal_bidders = sorted(bidder for bidder in actions.bids if bidder not in alive)
    if illegal_bidders:
        raise ValueError(f"resolve_discussion received bids from players who are not alive: {illegal_bidders}")

    by_amount: dict[int, list[str]] = defaultdict(list)
    for bidder, amount in actions.bids.items():
        by_amount[amount].append(bidder)

    # Sort the amount keys descending *and* sort each tie group by name before
    # shuffling so the RNG sees byte-identical input regardless of dict
    # insertion order — the per-group draw is the only stochastic step, and it
    # must be replayable (invariant #4).
    ordered: list[str] = []
    for amount in sorted(by_amount, reverse=True):
        group = sorted(by_amount[amount])
        ordered.extend(rng.shuffle(group))

    return DiscussionResult(speakers=tuple(ordered[:K_DISCUSSION_SLOTS]))
