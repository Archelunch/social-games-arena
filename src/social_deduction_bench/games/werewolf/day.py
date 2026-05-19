"""Werewolf day resolution (WEREWOLF_DESIGN.md §4).

`resolve_day` derives the outcome of the public day phase: it tallies the exile
vote and removes the plurality target from the game. It is a pure derivation
that mirrors `resolve_night` — it never mutates the input `GameState`
(invariant #1), and it returns a new state plus the event drafts the game loop
will log. The exile announcement is a public broadcast (invariant #2). Day
resolution takes no `GameRNG`: the §12 tie-break is "no exile", a deterministic
rule with no revote and no seed draw, so the day phase is replayable by
construction (invariant #4).
"""

from collections import Counter
from dataclasses import dataclass

from social_deduction_bench.engine import GameState, Phase
from social_deduction_bench.games.werewolf.events import ABSTAIN, EXILE_RESOLVED, EventDraft


@dataclass(frozen=True, slots=True)
class DayActions:
    """The decided exile votes resolution takes as input.

    Agent deliberation and tool validation are out of scope here — this is an
    already-decided `voter name -> target name` mapping, where a target may be
    `ABSTAIN` to mean "no choice". *Target* legality (the target is a real,
    living player) belongs to tool-call validation at the game-loop boundary,
    not this resolver. *Voter* legality is different: `resolve_day` rejects a
    vote cast by a non-living player itself, because the referee must never
    count an ineligible ballot into the tally.
    """

    exile_votes: dict[str, str]


@dataclass(frozen=True, slots=True)
class DayResult:
    """The outcome of one resolved day.

    `state` is the post-day snapshot, `drafts` the ordered events for the loop
    to log, `exiled` the player removed from the game (`None` on a tie or an
    all-abstain day — both resolve to no exile).
    """

    state: GameState
    drafts: tuple[EventDraft, ...]
    exiled: str | None


def resolve_day(state: GameState, actions: DayActions) -> DayResult:
    """Resolve the day phase into a new state and an exile event draft.

    Pure: `state` is never mutated (invariant #1). Abstentions are excluded from
    the tally; a clear plurality is exiled, while a tie resolves to no exile
    (WEREWOLF_DESIGN.md §12 — no revote, no seed tie-break, hence no RNG and
    deterministic resolution, invariant #4). The exile announcement is a public
    broadcast (invariant #2). Raises `ValueError` outside the day phase, or when
    any voter is not a living player — the engine-as-referee must reject an
    ineligible ballot rather than count it into the tally. Both are caller bugs.
    """
    if state.phase is not Phase.DAY:
        raise ValueError(f"resolve_day requires the day phase, got {state.phase.value}")

    alive = set(state.alive_names())
    illegal_voters = sorted(voter for voter in actions.exile_votes if voter not in alive)
    if illegal_voters:
        raise ValueError(f"resolve_day received exile votes from players who are not alive: {illegal_voters}")

    counts = Counter(target for target in actions.exile_votes.values() if target != ABSTAIN)
    if not counts:
        exiled: str | None = None
    else:
        max_count = max(counts.values())
        leaders = [name for name, c in counts.items() if c == max_count]
        exiled = leaders[0] if len(leaders) == 1 else None

    new_state = state.with_player_killed(exiled) if exiled is not None else state

    exile_draft = EventDraft(type=EXILE_RESOLVED, payload={"exiled": exiled})

    return DayResult(state=new_state, drafts=(exile_draft,), exiled=exiled)
