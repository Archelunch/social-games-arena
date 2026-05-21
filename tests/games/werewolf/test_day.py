"""Tests for Werewolf day resolution (T12).

Day resolution tallies the exile vote (WEREWOLF_DESIGN.md §4). These tests
encode the rules and the benchmark invariants:

- the most-voted player is exiled (plurality);
- a tie produces *no exile* — the WEREWOLF_DESIGN.md §12 decision pinned in
  BACKLOG T12 (no revote, no seed tie-break), so day resolution uses no RNG;
- abstentions are excluded from the tally and never tip a plurality;
- #1 — `resolve_day` is a pure derivation; it never mutates the input state;
- #2 — the exile announcement is a public broadcast (empty recipients);
- #4 — with no RNG, day resolution is deterministic by construction.

Resolution takes *decided* votes as input — agent deliberation and tool
validation are M3/M4 concerns, out of scope here.
"""

import pytest

from social_deduction_bench.engine import GameState, Phase
from social_deduction_bench.games.werewolf.day import DayActions, resolve_day
from social_deduction_bench.games.werewolf.events import ABSTAIN, EXILE_RESOLVED

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


def test_plurality_target_is_exiled() -> None:
    """The most-voted player is exiled and dead in the post-day state.

    The core day rule: a clear plurality removes that player from the game.
    """
    state = _day_state()
    actions = DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1", "Seer": "Vil1", "Doc": "Vil2"})

    result = resolve_day(state, actions)

    assert result.exiled == "Vil1"
    assert result.state.player("Vil1").alive is False


def test_a_tie_results_in_no_exile() -> None:
    """A tied vote exiles nobody and leaves the roster unchanged.

    WEREWOLF_DESIGN.md §12 / BACKLOG T12: a tie is resolved to no-exile — not a
    revote, not a seed tie-break. Nobody dies on a tie.
    """
    state = _day_state()
    actions = DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1", "Seer": "Vil2", "Doc": "Vil2"})

    result = resolve_day(state, actions)

    assert result.exiled is None
    assert result.state.alive_names() == state.alive_names()


def test_all_abstain_results_in_no_exile() -> None:
    """A day where every voter abstains exiles nobody.

    Abstention is a legal vote; a day of all-abstentions is a valid no-exile
    outcome, not an error.
    """
    state = _day_state()
    actions = DayActions(exile_votes={"Wolf1": ABSTAIN, "Wolf2": ABSTAIN, "Seer": ABSTAIN})

    result = resolve_day(state, actions)

    assert result.exiled is None


def test_abstentions_are_excluded_from_the_tally() -> None:
    """Abstain votes do not count toward any candidate's total.

    Here Vil1 and Vil2 each have one real vote — a tie — and three abstentions.
    If abstentions were miscounted as a candidate they could manufacture or
    break a plurality; the outcome must be no-exile.
    """
    state = _day_state()
    actions = DayActions(
        exile_votes={
            "Wolf1": "Vil1",
            "Wolf2": "Vil2",
            "Seer": ABSTAIN,
            "Doc": ABSTAIN,
            "Vil3": ABSTAIN,
        }
    )

    result = resolve_day(state, actions)

    assert result.exiled is None


def test_exile_resolved_event_is_public() -> None:
    """The exile announcement is a public broadcast — empty recipients.

    An exile is common knowledge; the `exile_resolved` event must reach every
    player (invariant #2: empty recipients = public).
    """
    state = _day_state()
    actions = DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})

    result = resolve_day(state, actions)
    exile_drafts = [d for d in result.drafts if d.type == EXILE_RESOLVED]

    assert len(exile_drafts) == 1
    assert exile_drafts[0].recipients == ()
    assert exile_drafts[0].payload["exiled"] == "Vil1"


def test_exile_resolved_payload_names_no_one_on_a_tie() -> None:
    """On a no-exile day the `exile_resolved` payload carries `None`.

    The public record must still announce the day's outcome — that nobody was
    exiled — so post-hoc metrics (T24) can count no-exile days.
    """
    state = _day_state()
    actions = DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil2"})

    result = resolve_day(state, actions)
    exile_draft = next(d for d in result.drafts if d.type == EXILE_RESOLVED)

    assert exile_draft.payload["exiled"] is None


def test_exile_resolved_payload_carries_the_full_ballot_map() -> None:
    """`EXILE_RESOLVED.payload["ballots"]` carries the voter -> target map.

    Replay and T24's exile-accuracy attribution need the per-voter ballot, not
    only the resolved exile. The map must equal the input `exile_votes`
    exactly. A tied vote still records both ballots — the tie-resolution
    (no exile) is recorded in `exiled`, separately from the ballots.
    """
    state = _day_state()
    exile_votes = {"Wolf1": "Vil1", "Wolf2": "Vil2", "Seer": "Wolf1", "Doc": "Wolf1"}
    actions = DayActions(exile_votes=exile_votes)

    result = resolve_day(state, actions)
    exile_draft = next(d for d in result.drafts if d.type == EXILE_RESOLVED)

    assert exile_draft.payload["ballots"] == exile_votes
    assert exile_draft.payload["exiled"] == "Wolf1"


def test_exile_resolved_ballots_preserve_abstain_targets() -> None:
    """Abstain votes appear in `ballots` as the literal `"abstain"`.

    Abstain is excluded from the *tally*, but the ballot itself is part of the
    transcript — replay and metrics must be able to count abstentions per
    voter. An all-abstain day still records every voter mapped to `"abstain"`.
    """
    state = _day_state()
    actions = DayActions(exile_votes={"Wolf1": ABSTAIN, "Wolf2": ABSTAIN, "Seer": ABSTAIN})

    result = resolve_day(state, actions)
    exile_draft = next(d for d in result.drafts if d.type == EXILE_RESOLVED)

    assert exile_draft.payload["ballots"] == {"Wolf1": ABSTAIN, "Wolf2": ABSTAIN, "Seer": ABSTAIN}
    assert exile_draft.payload["exiled"] is None


def test_exile_resolved_ballots_are_a_fresh_dict_not_the_input() -> None:
    """Mutating the input `exile_votes` after resolution does not change the logged ballots."""
    state = _day_state()
    exile_votes = {"Wolf1": "Vil1", "Wolf2": "Vil1"}
    actions = DayActions(exile_votes=exile_votes)

    result = resolve_day(state, actions)
    exile_votes["Wolf1"] = "Vil3"

    exile_draft = next(d for d in result.drafts if d.type == EXILE_RESOLVED)
    assert exile_draft.payload["ballots"] == {"Wolf1": "Vil1", "Wolf2": "Vil1"}


def test_plurality_below_a_majority_still_exiles() -> None:
    """A plurality short of an outright majority still exiles the top candidate.

    Votes split 3/2/2 across three candidates: Vil1's 3 is the plurality but
    not a majority of the 7 voters. WEREWOLF_DESIGN.md §4's "majority" is
    implemented as plurality (a strict-majority rule would stall most days);
    this locks that so a future switch to strict majority fails loudly here.
    """
    state = _day_state()
    actions = DayActions(
        exile_votes={
            "Wolf1": "Vil1",
            "Wolf2": "Vil1",
            "Seer": "Vil1",
            "Doc": "Vil2",
            "Vil3": "Vil2",
            "Vil1": "Wolf1",
            "Vil2": "Wolf1",
        }
    )

    result = resolve_day(state, actions)

    assert result.exiled == "Vil1"


def test_resolve_day_rejects_a_vote_from_a_dead_player() -> None:
    """An exile vote cast by a dead player fails loud — the referee rejects it.

    The day vote is "every *alive* player casts one"; a dead voter left in the
    tally could silently tip a plurality. The engine-as-referee must reject the
    ineligible ballot, not trust the caller to have filtered the dead.
    """
    state = GameState.initial(ROSTER).with_player_killed("Vil3").with_phase(Phase.DAY)
    actions = DayActions(exile_votes={"Wolf1": "Vil1", "Vil3": "Vil1"})

    with pytest.raises(ValueError, match="not alive"):
        resolve_day(state, actions)


def test_resolve_day_does_not_mutate_the_input_state() -> None:
    """`resolve_day` is a pure derivation — the input snapshot is untouched.

    Invariant #1: an earlier recorded position must stay intact after the game
    moves on, or replay is corrupted. Asserts the whole input state, not just
    one player, is structurally unchanged.
    """
    state = _day_state()
    before = _day_state()  # an independent, structurally-equal snapshot
    actions = DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})

    resolve_day(state, actions)

    assert state == before


def test_resolve_day_rejects_a_non_day_phase() -> None:
    """Resolving a day during the NIGHT phase fails loud.

    Day resolution outside the day phase is a caller bug — silently resolving
    it would corrupt the round's event ordering.
    """
    night_state = GameState.initial(ROSTER)
    actions = DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"})

    with pytest.raises(ValueError, match="day"):
        resolve_day(night_state, actions)


def test_resolve_day_is_deterministic() -> None:
    """Identical votes resolve to an identical result, every time.

    Day resolution uses no RNG (the §12 no-exile tie-break is deterministic);
    this pins that it introduces no nondeterminism — two runs must be equal.
    """
    state = _day_state()
    actions = DayActions(exile_votes={"Wolf1": "Vil1", "Wolf2": "Vil1", "Seer": "Vil2"})

    assert resolve_day(state, actions) == resolve_day(state, actions)
