"""Tests for Werewolf night resolution (T11).

Night resolution is the core hidden-information phase: the werewolves' joint
kill, the seer's private inspect, the doctor's protect. These tests encode the
rules from WEREWOLF_DESIGN.md §4 and the benchmark invariants:

- #1 — `resolve_night` is a pure derivation; it never mutates the input state.
- #2 — the seer's result is private (carries the seer as its sole recipient);
  a public death announcement carries none.
- #4 — the only stochastic point, the kill-vote tie-break, derives entirely
  from the engine seed: same seed + same votes -> same victim.

Resolution takes *decided* actions as inputs — agent deliberation and tool
validation are M3/M4 concerns, out of scope here.
"""

import pytest

from social_deduction_bench.engine import GameRNG, GameState, Phase, assert_recipients_present
from social_deduction_bench.games.werewolf.config import PRIVATE_EVENT_TYPES
from social_deduction_bench.games.werewolf.events import DOCTOR_PROTECT, KILL_RESOLVED, SEER_INSPECT, EventDraft
from social_deduction_bench.games.werewolf.night import NightActions, resolve_night

ROSTER = (
    ("Wolf1", "werewolf"),
    ("Wolf2", "werewolf"),
    ("Seer", "seer"),
    ("Doc", "doctor"),
    ("Vil1", "villager"),
    ("Vil2", "villager"),
    ("Vil3", "villager"),
)


def _draft(drafts: tuple[EventDraft, ...], type_: str) -> EventDraft:
    """Return the single draft of `type_`; fail loud if absent or duplicated."""
    matches = [d for d in drafts if d.type == type_]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {type_!r} draft, found {len(matches)}")
    return matches[0]


def test_plurality_kill_target_dies() -> None:
    """The werewolves' plurality kill target dies — the core night-kill rule.

    Both werewolves voting the same villager leaves no tie; that villager must
    be dead in the post-night state and the result must name them.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})

    result = resolve_night(state, actions, GameRNG(0))

    assert result.killed == "Vil1"
    assert result.state.player("Vil1").alive is False
    assert _draft(result.drafts, KILL_RESOLVED).payload["victim"] == "Vil1"


def test_protection_on_the_kill_target_suppresses_the_kill() -> None:
    """A doctor protecting the kill target suppresses the death — the central T11 rule.

    When `doctor_protect` equals the kill target, nobody dies: `killed` is None
    and the roster is unchanged.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(
        kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"},
        doctor_protect="Vil1",
    )

    result = resolve_night(state, actions, GameRNG(0))

    assert result.killed is None
    assert result.state.player("Vil1").alive is True
    assert _draft(result.drafts, KILL_RESOLVED).payload["victim"] is None


def test_protection_on_a_non_target_does_not_save_the_target() -> None:
    """Protecting a player who is not the kill target does not suppress the kill.

    Guards against a protect that suppresses unconditionally — only protection
    *of the kill target* may save a life.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(
        kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"},
        doctor_protect="Vil2",
    )

    result = resolve_night(state, actions, GameRNG(0))

    assert result.killed == "Vil1"
    assert result.state.player("Vil1").alive is False


def test_kill_vote_tie_is_broken_by_the_seed() -> None:
    """A split werewolf vote is resolved by the engine seed, not arbitrarily.

    With Wolf1->Vil1 and Wolf2->Vil2 the vote ties; seed 1 and seed 0 are known
    to break the (sorted) ties to different victims. The tie-break derives from
    the seed (invariant #4), and both outcomes are legitimate leaders.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil2"})

    seed1_victim = resolve_night(state, actions, GameRNG(1)).killed
    seed0_victim = resolve_night(state, actions, GameRNG(0)).killed

    assert {seed1_victim, seed0_victim} <= {"Vil1", "Vil2"}  # only the two tied leaders are legal
    assert seed1_victim != seed0_victim  # the seed actually changes which leader dies
    assert seed1_victim == "Vil1"  # golden pin: seed 1 -> Vil1
    assert seed0_victim == "Vil2"  # golden pin: seed 0 -> Vil2


def test_kill_vote_tie_break_is_seed_deterministic() -> None:
    """Same seed + same tied votes -> identical victim, every time.

    Invariant #4: the tie-break must be replayable. Two runs with seed 1 must
    pick the same villager, or recorded games would diverge on replay.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil2"})

    first = resolve_night(state, actions, GameRNG(1)).killed
    second = resolve_night(state, actions, GameRNG(1)).killed

    assert first == second


def test_seer_result_is_private_to_the_seer() -> None:
    """The seer's inspect result is delivered only to the seer (invariant #2).

    The `seer_inspect` draft must carry the seer as its sole recipient — a
    public seer result would hand the villagers' best information to the
    werewolves.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"}, seer_inspect="Wolf1")

    result = resolve_night(state, actions, GameRNG(0))
    seer_draft = _draft(result.drafts, SEER_INSPECT)

    assert seer_draft.recipients == ("Seer",)
    assert seer_draft.payload["target"] == "Wolf1"
    assert seer_draft.payload["faction"] == "werewolves"


def test_seer_inspecting_a_villager_returns_the_villagers_faction() -> None:
    """Inspecting a villager yields the VILLAGERS faction — the result is correct.

    The seer learns a faction, not a role; a villager and the doctor must both
    read back as `villagers`.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(kill_votes={"Wolf1": "Vil2", "Wolf2": "Vil2"}, seer_inspect="Vil1")

    result = resolve_night(state, actions, GameRNG(0))

    assert _draft(result.drafts, SEER_INSPECT).payload["faction"] == "villagers"


def test_kill_resolved_event_is_public() -> None:
    """The death announcement is a public broadcast — empty recipients.

    Who died at night is common knowledge; the `kill_resolved` event must reach
    every player (invariant #2: empty recipients = public).
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})

    result = resolve_night(state, actions, GameRNG(0))

    assert _draft(result.drafts, KILL_RESOLVED).recipients == ()


def test_drafts_are_in_fixed_seer_doctor_kill_order() -> None:
    """Drafts are emitted in a fixed order: seer, doctor, kill-resolved.

    A fixed order makes the transcript byte-identical on replay (invariant #4);
    an order that depended on dict iteration or input shape would break T08.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(
        kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"},
        seer_inspect="Wolf1",
        doctor_protect="Vil2",
    )

    result = resolve_night(state, actions, GameRNG(0))

    assert [d.type for d in result.drafts] == [SEER_INSPECT, DOCTOR_PROTECT, KILL_RESOLVED]


def test_no_seer_or_doctor_action_emits_only_the_kill_event() -> None:
    """With no seer/doctor action, only the public kill event is drafted.

    A night where the seer and doctor are absent (or chose not to act) produces
    exactly one draft — no empty private events that the guard would reject.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})

    result = resolve_night(state, actions, GameRNG(0))

    assert [d.type for d in result.drafts] == [KILL_RESOLVED]


def test_every_night_draft_passes_the_private_event_guard() -> None:
    """Every draft `resolve_night` emits satisfies the engine private-event guard.

    The seer and doctor drafts are declared-private types; a draft of one with
    empty recipients would broadcast hidden state (invariant #2). This ties the
    resolver's output directly to `assert_recipients_present` now, rather than
    relying on the T14 game loop to be the first thing that catches a leak.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(
        kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"},
        seer_inspect="Wolf1",
        doctor_protect="Vil2",
    )

    result = resolve_night(state, actions, GameRNG(0))

    for draft in result.drafts:
        assert_recipients_present(draft.type, draft.recipients, PRIVATE_EVENT_TYPES)


def test_resolve_night_does_not_mutate_the_input_state() -> None:
    """`resolve_night` is a pure derivation — the input snapshot is untouched.

    Invariant #1: an earlier recorded position must stay intact after the game
    moves on, or replay is corrupted.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})

    resolve_night(state, actions, GameRNG(0))

    assert state.player("Vil1").alive is True
    assert state.phase is Phase.NIGHT


def test_resolve_night_rejects_a_non_night_phase() -> None:
    """Resolving a night during the DAY phase fails loud.

    Night resolution outside the night phase is a caller bug — silently
    resolving it would corrupt the round's event ordering.
    """
    state = GameState.initial(ROSTER).with_phase(Phase.DAY)
    actions = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})

    with pytest.raises(ValueError, match="night"):
        resolve_night(state, actions, GameRNG(0))


def test_resolve_night_rejects_a_kill_vote_from_a_dead_werewolf() -> None:
    """A kill vote cast by a dead werewolf fails loud — the referee rejects it.

    A dead werewolf must not influence the joint kill; tallying its vote could
    tip the target. This mirrors day resolution's living-voter guard so both
    resolvers are consistent referees.
    """
    state = GameState.initial(ROSTER).with_player_killed("Wolf2")
    actions = NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})

    with pytest.raises(ValueError, match="not alive"):
        resolve_night(state, actions, GameRNG(0))


def test_empty_kill_votes_is_rejected() -> None:
    """A night with no werewolf kill votes fails loud.

    Living werewolves always cast a joint kill (the game is terminal before a
    night with zero werewolves); an empty `kill_votes` is a caller bug, not a
    silent no-kill night.
    """
    state = GameState.initial(ROSTER)
    actions = NightActions(kill_votes={})

    with pytest.raises(ValueError, match="kill"):
        resolve_night(state, actions, GameRNG(0))
