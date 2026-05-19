"""Werewolf night resolution (WEREWOLF_DESIGN.md §4).

`resolve_night` derives the outcome of the hidden-information night phase: the
werewolves' joint kill, the seer's private inspect, and the doctor's protect. It
is a pure derivation that mirrors `advance_phase` — it never mutates the input
`GameState` (invariant #1), and it returns a new state plus the ordered event
drafts the game loop will log. The kill-vote tie-break is the single stochastic
point and derives entirely from the engine seed (invariant #4); the seer's
result is private to the seer (invariant #2).
"""

from collections import Counter
from dataclasses import dataclass

from social_deduction_bench.engine import GameRNG, GameState, Phase
from social_deduction_bench.games.werewolf.events import (
    DOCTOR_PROTECT,
    KILL_RESOLVED,
    SEER_INSPECT,
    EventDraft,
)
from social_deduction_bench.games.werewolf.roles import Role, faction_of


@dataclass(frozen=True, slots=True)
class NightActions:
    """The decided night actions resolution takes as input.

    Agent deliberation and tool validation are out of scope here — these are
    already-decided actions: a `werewolf name -> kill target name` mapping, and
    the optional seer/doctor targets. *Target* legality (the target is alive, a
    real player, the right phase/role) is trusted as given — that belongs to
    tool-call validation at the game-loop boundary (T07/T15), not this resolver.
    *Voter* legality is different: `resolve_night` rejects a kill vote cast by a
    non-living player itself, because the referee must never tally an
    ineligible ballot.
    """

    kill_votes: dict[str, str]
    seer_inspect: str | None = None
    doctor_protect: str | None = None


@dataclass(frozen=True, slots=True)
class NightResult:
    """The outcome of one resolved night.

    `state` is the post-night snapshot, `drafts` the ordered events for the loop
    to log, `killed` the player who died (`None` when protected or no kill).
    """

    state: GameState
    drafts: tuple[EventDraft, ...]
    killed: str | None


def _living_player_with_role(state: GameState, role: Role) -> str:
    """Return the single living player whose role is `role`; fail loud otherwise.

    The benchmark config deals exactly one seer and one doctor; anything but a
    single living holder is a caller bug, not a state this resolver guesses past.
    """
    names = [p.name for p in state.alive_players() if p.role == role.value]
    if len(names) != 1:
        raise ValueError(f"expected exactly one living {role.value}, found {len(names)}")
    return names[0]


def resolve_night(state: GameState, actions: NightActions, rng: GameRNG) -> NightResult:
    """Resolve the night phase into a new state and ordered event drafts.

    Pure: `state` is never mutated (invariant #1). The kill-vote tie-break is the
    only stochastic step and is seeded via `rng` over a sorted leader list, so it
    is replayable (invariant #4). The seer's result draft is private to the seer
    (invariant #2). Raises `ValueError` outside the night phase, on empty
    `kill_votes`, or when a kill vote is cast by a non-living player — the
    engine-as-referee must reject an ineligible ballot rather than tally it.
    All are caller bugs.
    """
    if state.phase is not Phase.NIGHT:
        raise ValueError(f"resolve_night requires the night phase, got {state.phase.value}")
    if not actions.kill_votes:
        raise ValueError("resolve_night requires at least one werewolf kill vote")

    alive = set(state.alive_names())
    illegal_voters = sorted(voter for voter in actions.kill_votes if voter not in alive)
    if illegal_voters:
        raise ValueError(f"resolve_night received kill votes from players who are not alive: {illegal_voters}")

    counts = Counter(actions.kill_votes.values())
    max_count = max(counts.values())
    leaders = sorted(name for name, c in counts.items() if c == max_count)
    kill_target = leaders[0] if len(leaders) == 1 else rng.choice(leaders)

    drafts: list[EventDraft] = []

    if actions.seer_inspect is not None:
        seer_name = _living_player_with_role(state, Role.SEER)
        faction = faction_of(state.player(actions.seer_inspect).role)
        drafts.append(
            EventDraft(
                type=SEER_INSPECT,
                payload={"target": actions.seer_inspect, "faction": faction.value},
                recipients=(seer_name,),
            )
        )

    if actions.doctor_protect is not None:
        doctor_name = _living_player_with_role(state, Role.DOCTOR)
        drafts.append(
            EventDraft(
                type=DOCTOR_PROTECT,
                payload={"target": actions.doctor_protect},
                recipients=(doctor_name,),
            )
        )

    killed = None if actions.doctor_protect == kill_target else kill_target
    new_state = state.with_player_killed(killed) if killed is not None else state

    drafts.append(EventDraft(type=KILL_RESOLVED, payload={"victim": killed}))

    return NightResult(state=new_state, drafts=tuple(drafts), killed=killed)
