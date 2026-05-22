"""Tests for `ReActDecisionSource`.

The adapter wires a per-player ReAct loop (from `react_decide`) into the
`run_game` driver. Per round, it walks alive players in roster order and runs
one decision-point loop per acting role; after each phase, the driver calls
`observe` with newly-appended events, which the adapter routes through
`observations_for` into each living player's `GameMemory`.

Tests use `DummyLM` to script each loop's LM responses in the order they will
be consumed. A loop ends the moment it commits a terminal action — there is no
separate `finish` step — so a single terminal commit is one ReAct iteration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from dspy.utils.dummies import DummyLM

from social_deduction_bench.agents import GameMemory
from social_deduction_bench.agents.decisions import ReActDecisionSource
from social_deduction_bench.engine import (
    EventLog,
    GameState,
    Phase,
    advance_phase,
    assert_deterministic,
    assert_streams_identical,
)
from social_deduction_bench.engine.events import EventStream
from social_deduction_bench.games.werewolf.config import BID_BUDGET
from social_deduction_bench.games.werewolf.events import (
    ABSTAIN,
    ACCUSATION,
    BID,
    DEFENSE,
    EXILE_RESOLVED,
    GAME_OVER,
    KILL_RESOLVED,
    SEER_INSPECT,
    SPEECH,
    TOOL_REJECTED,
    WEREWOLF_CHAT,
)
from social_deduction_bench.games.werewolf.loop import run_game
from social_deduction_bench.games.werewolf.roles import Role

ROSTER: tuple[tuple[str, str], ...] = (
    ("Wolf1", Role.WEREWOLF.value),
    ("Wolf2", Role.WEREWOLF.value),
    ("Seer1", Role.SEER.value),
    ("Doc1", Role.DOCTOR.value),
    ("Vil1", Role.VILLAGER.value),
    ("Vil2", Role.VILLAGER.value),
    ("Vil3", Role.VILLAGER.value),
)


def _step(tool: str, args: dict[str, Any], thought: str = "step") -> dict[str, Any]:
    return {"next_thought": thought, "next_tool_name": tool, "next_tool_args": args}


def _commit_pair(tool: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    # A terminal commit ends the turn in one iteration (no finish step), so a
    # "commit" is a single scripted answer. Kept as a list (and named for its
    # historical commit+finish shape) so call sites that splice it stay unchanged.
    return [_step(tool, args)]


def _uniform_lms(lm: DummyLM, roster: Sequence[tuple[str, str]] = ROSTER) -> dict[str, DummyLM]:
    """Seat the same `DummyLM` instance at every roster name.

    Pre-T22 tests scripted one global answer queue in roster-iteration order.
    Sharing a single `DummyLM` across seats keeps that single-cursor semantics
    intact — the answer queue is consumed exactly as before.
    """
    return {name: lm for name, _ in roster}


def _empty_lms(roster: Sequence[tuple[str, str]] = ROSTER) -> dict[str, DummyLM]:
    """Seat an empty `DummyLM` at every roster name (no game actions expected)."""
    return {name: DummyLM([]) for name, _ in roster}


def test_observe_routes_public_event_to_every_alive_player() -> None:
    source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
    state = GameState.initial(ROSTER)

    log = EventLog()
    log.append(round=1, phase=Phase.NIGHT, type=KILL_RESOLVED, payload={"victim": "Vil1"})
    source.observe(state, log.events)

    for name in (n for n, _ in ROSTER):
        assert len(source.memories[name].events) == 1
        assert source.memories[name].events[0].type == KILL_RESOLVED


def test_observe_routes_private_event_to_named_recipients_only() -> None:
    source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
    state = GameState.initial(ROSTER)

    log = EventLog()
    log.append(
        round=1,
        phase=Phase.NIGHT,
        type=SEER_INSPECT,
        payload={"target": "Wolf1", "faction": "werewolves"},
        recipients=("Seer1",),
    )
    source.observe(state, log.events)

    assert len(source.memories["Seer1"].events) == 1
    for name in (n for n, _ in ROSTER if n != "Seer1"):
        assert source.memories[name].events == ()


def test_observe_does_not_duplicate_events_across_two_calls() -> None:
    source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
    state = GameState.initial(ROSTER)

    log = EventLog()
    log.append(round=1, phase=Phase.NIGHT, type=KILL_RESOLVED, payload={"victim": "Vil1"})
    source.observe(state, log.events[:1])
    log.append(round=2, phase=Phase.DAY, type=EXILE_RESOLVED, payload={"exiled": "Wolf2"})
    source.observe(state, log.events[1:])

    for name in (n for n, _ in ROSTER):
        types = tuple(e.type for e in source.memories[name].events)
        assert types == (KILL_RESOLVED, EXILE_RESOLVED)


def test_night_actions_aggregates_per_role_terminals() -> None:
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))
    state = GameState.initial(ROSTER)

    actions = source.night_actions(state)

    assert actions.kill_votes == {"Wolf1": "Vil1", "Wolf2": "Vil1"}
    assert actions.seer_inspect == "Wolf1"
    assert actions.doctor_protect == "Vil1"


def test_night_actions_skips_dead_seer_returns_none() -> None:
    state = GameState.initial(ROSTER).with_player_killed("Seer1")

    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))

    actions = source.night_actions(state)

    assert actions.seer_inspect is None
    assert actions.kill_votes == {"Wolf1": "Vil1", "Wolf2": "Vil1"}
    assert actions.doctor_protect == "Vil1"


def test_night_actions_skips_dead_doctor_returns_none() -> None:
    state = GameState.initial(ROSTER).with_player_killed("Doc1")

    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))

    actions = source.night_actions(state)

    assert actions.doctor_protect is None
    assert actions.seer_inspect == "Wolf1"


def test_day_actions_collects_a_vote_from_every_alive_player() -> None:
    state = advance_phase(GameState.initial(ROSTER))

    targets = ["Wolf2", "Vil1", "Wolf1", "Wolf1", "Wolf1", "Wolf2", "Vil2"]
    answers: list[dict[str, Any]] = []
    for target in targets:
        answers.extend(_commit_pair("submit_exile_vote", {"target": target}))
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))

    actions = source.day_actions(state)

    assert set(actions.exile_votes.keys()) == {n for n, _ in ROSTER}
    assert actions.exile_votes["Wolf1"] == "Wolf2"
    assert actions.exile_votes["Vil3"] == "Vil2"


def test_day_actions_passes_abstain_through() -> None:
    state = advance_phase(GameState.initial(ROSTER))
    answers = (
        _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_exile_vote", {"target": "Wolf2"})
        + _commit_pair("submit_exile_vote", {"target": "Wolf1"})
        + _commit_pair("submit_exile_vote", {"target": "Wolf1"})
        + _commit_pair("submit_exile_vote", {"target": "Wolf1"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))

    actions = source.day_actions(state)

    assert actions.exile_votes["Wolf1"] == ABSTAIN
    assert actions.exile_votes["Vil2"] == ABSTAIN
    assert actions.exile_votes["Vil3"] == ABSTAIN


def test_roster_order_determines_lm_call_order() -> None:
    swapped: tuple[tuple[str, str], ...] = (
        ("Wolf2", Role.WEREWOLF.value),
        ("Wolf1", Role.WEREWOLF.value),
        ("Seer1", Role.SEER.value),
        ("Doc1", Role.DOCTOR.value),
        ("Vil1", Role.VILLAGER.value),
        ("Vil2", Role.VILLAGER.value),
        ("Vil3", Role.VILLAGER.value),
    )
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    source = ReActDecisionSource(roster=swapped, lms=_uniform_lms(DummyLM(answers), swapped))
    state = GameState.initial(swapped)

    actions = source.night_actions(state)

    assert actions.kill_votes == {"Wolf2": "Vil1", "Wolf1": "Vil2"}


_K_SPEAKERS = 3  # `K_DISCUSSION_SLOTS` — see `games.werewolf.config`.


def _day_dialogue_pairs(n_alive: int) -> list[dict[str, Any]]:
    """Bid + speech + reaction answers for one day, in `run_game` call order.

    The loop walks the roster for `bids()` (one `submit_bid` commit per alive
    player), then the chosen `speakers` tuple (length `min(K, n_alive)`) for
    `speeches()`, then every alive player once for the reaction round (here every
    seat passes). The uniform-LM tests share one answer queue across seats, so
    the queue must hold every commit in the order they will be consumed — and
    since all reactions are `pass_turn`, the seeded reaction order does not change
    which answer is drawn.
    """
    pairs: list[dict[str, Any]] = []
    for _ in range(n_alive):
        pairs += _commit_pair("submit_bid", {"amount": 0})
    for _ in range(min(_K_SPEAKERS, n_alive)):
        pairs += _commit_pair("speak", {"message": "no comment"})
    for _ in range(n_alive):
        pairs += _commit_pair("pass_turn", {})
    return pairs


def _werewolf_sweep_script() -> list[dict[str, Any]]:
    """A scripted full-game LM transcript ending in a werewolf victory.

    Three villagers die over three nights; days all abstain (no exile). The
    village goes from 5 villagers + 2 wolves to 2 villagers + 2 wolves at
    parity — a werewolf win on the post-night terminal check. Each day
    interleaves `bids()` + `speeches()` between the night kill and the exile
    vote (T29 wiring); answers cover all three sub-phases.
    """
    answers: list[dict[str, Any]] = []

    # Each night now opens with a wolf chat sub-phase (Wolf1 then Wolf2 in
    # roster order) before the kill/inspect/protect actions.
    # Round 1: night chat (2 wolves) -> night actions (4) -> day.
    answers += _commit_pair("werewolf_chat", {"message": "take Vil1"})
    answers += _commit_pair("werewolf_chat", {"message": "agreed"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil1"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil1"})
    answers += _commit_pair("seer_inspect", {"target": "Wolf1"})
    answers += _commit_pair("doctor_protect", {"target": "Seer1"})
    answers += _day_dialogue_pairs(6)
    for _ in range(6):
        answers += _commit_pair("submit_exile_vote", {"target": ABSTAIN})

    # Round 2: night chat (2 wolves) -> night actions (4) -> day.
    answers += _commit_pair("werewolf_chat", {"message": "now Vil2"})
    answers += _commit_pair("werewolf_chat", {"message": "ok"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil2"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil2"})
    answers += _commit_pair("seer_inspect", {"target": "Wolf2"})
    answers += _commit_pair("doctor_protect", {"target": "Vil3"})
    answers += _day_dialogue_pairs(5)
    for _ in range(5):
        answers += _commit_pair("submit_exile_vote", {"target": ABSTAIN})

    # Round 3 night ends the game on werewolf parity.
    answers += _commit_pair("werewolf_chat", {"message": "finish them"})
    answers += _commit_pair("werewolf_chat", {"message": "Vil3"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil3"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil3"})
    answers += _commit_pair("seer_inspect", {"target": "Doc1"})
    answers += _commit_pair("doctor_protect", {"target": "Seer1"})

    return answers


def test_run_game_with_react_source_reaches_terminal_state() -> None:
    answers = _werewolf_sweep_script()
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))

    stream = run_game(ROSTER, seed=42, decisions=source)
    events = stream.log.events

    assert events[-1].type == GAME_OVER
    assert events[-1].payload["winner"] == "werewolves"


def test_no_recorded_event_violates_hidden_state_for_any_player() -> None:
    """Every event in each player's memory is either public or names them as a recipient."""
    answers = _werewolf_sweep_script()
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))
    run_game(ROSTER, seed=42, decisions=source)

    for name in (n for n, _ in ROSTER):
        for event in source.memories[name].events:
            assert event.recipients == () or name in event.recipients, (
                f"player {name!r} has private event {event.type!r} not addressed to them"
            )


def test_memories_property_is_a_read_only_mapping() -> None:
    source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
    assert isinstance(source.memories, Mapping)
    with pytest.raises(TypeError):
        source.memories["Wolf1"] = GameMemory()  # type: ignore[index]


def test_init_rejects_lms_missing_a_roster_name() -> None:
    """Construction fails loud when a roster seat has no LM assignment.

    A silent default would let a typo'd seat name play under the wrong model
    and corrupt cross-play ratings; the message must name the missing seat
    so the operator can correct the call site.
    """
    incomplete = {name: DummyLM([]) for name, _ in ROSTER if name != "Vil3"}
    with pytest.raises(ValueError, match="Vil3"):
        ReActDecisionSource(roster=ROSTER, lms=incomplete)


def test_init_rejects_lms_with_extra_name_not_in_roster() -> None:
    """Construction fails loud when `lms` contains a name not in the roster.

    A phantom entry usually means a typo on the intended seat name: silently
    accepting it would seat the *real* roster name under a fallback model
    and lose the operator-intended LM for that seat.
    """
    bloated = {name: DummyLM([]) for name, _ in ROSTER}
    bloated["GhostX"] = DummyLM([])
    with pytest.raises(ValueError, match="GhostX"):
        ReActDecisionSource(roster=ROSTER, lms=bloated)


def test_init_accepts_lms_keyed_by_exact_roster_set() -> None:
    lms = {name: DummyLM([]) for name, _ in ROSTER}
    source = ReActDecisionSource(roster=ROSTER, lms=lms)
    assert set(source.lms.keys()) == {name for name, _ in ROSTER}


def test_lms_property_is_a_read_only_mapping() -> None:
    source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
    assert isinstance(source.lms, Mapping)
    with pytest.raises(TypeError):
        source.lms["Wolf1"] = DummyLM([])  # type: ignore[index]


def test_each_seat_consumes_only_its_own_lm_in_night_actions() -> None:
    """Per-seat LM seating routes every acting LM call to that seat's LM only.

    Each acting seat is scripted with a *distinct* target so a cross-seat
    swap (e.g. Wolf1 <-> Wolf2) produces a wrong `kill_votes` mapping
    rather than passing silently — `history`-length alone would not catch
    a swap because both swapped LMs still get 2 calls. A villager's
    DummyLM remaining empty additionally guards against a shared-LM
    fallback that round-robined or pooled across seats.
    """
    seat_lms: dict[str, DummyLM] = {
        "Wolf1": DummyLM(_commit_pair("submit_kill_vote", {"target": "Vil1"})),
        "Wolf2": DummyLM(_commit_pair("submit_kill_vote", {"target": "Vil2"})),
        "Seer1": DummyLM(_commit_pair("seer_inspect", {"target": "Wolf1"})),
        "Doc1": DummyLM(_commit_pair("doctor_protect", {"target": "Vil3"})),
        "Vil1": DummyLM([]),
        "Vil2": DummyLM([]),
        "Vil3": DummyLM([]),
    }
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)
    state = GameState.initial(ROSTER)

    actions = source.night_actions(state)

    assert actions.kill_votes == {"Wolf1": "Vil1", "Wolf2": "Vil2"}
    assert actions.seer_inspect == "Wolf1"
    assert actions.doctor_protect == "Vil3"
    for acting in ("Wolf1", "Wolf2", "Seer1", "Doc1"):
        assert len(seat_lms[acting].history) == 1, (
            f"acting seat {acting!r} should consume exactly 1 LM call (the commit ends the turn), "
            f"got {len(seat_lms[acting].history)}"
        )
    for villager in ("Vil1", "Vil2", "Vil3"):
        assert seat_lms[villager].history == [], (
            f"villager {villager!r} acts at night for no role; its LM must not be called"
        )


def test_dead_seat_does_not_consume_its_lm_in_day_actions() -> None:
    """Dead seats are skipped at the LM layer, not just at the action-dict layer.

    `day_actions` must not enter the ReAct loop for a dead player; a dead
    seat's DummyLM must be untouched (`history == []`) after the day.
    """
    state = advance_phase(GameState.initial(ROSTER)).with_player_killed("Vil1")
    seat_lms: dict[str, DummyLM] = {
        name: DummyLM(_commit_pair("submit_exile_vote", {"target": ABSTAIN})) for name, _ in ROSTER
    }

    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)
    actions = source.day_actions(state)

    assert "Vil1" not in actions.exile_votes
    assert seat_lms["Vil1"].history == []
    for alive in ("Wolf1", "Wolf2", "Seer1", "Doc1", "Vil2", "Vil3"):
        assert len(seat_lms[alive].history) == 1, (
            f"alive seat {alive!r} should consume 1 LM call (the commit ends the turn), "
            f"got {len(seat_lms[alive].history)}"
        )


def _bid_then_maybe_speech(amount: int, *, speaks: bool) -> list[dict[str, Any]]:
    """One day's per-seat answers: a `submit_bid` commit, an optional `speak`
    commit when chosen, then a `pass_turn` for the reaction round.

    Every living seat reacts once per day after the statements; here each seat
    passes. The reaction commit lands between the seat's speech (if any) and its
    exile vote, matching the loop's bid → speak → react → vote call order.
    """
    out = _commit_pair("submit_bid", {"amount": amount})
    if speaks:
        out += _commit_pair("speak", {"message": "no comment"})
    out += _commit_pair("pass_turn", {})
    return out


def _per_seat_werewolf_sweep_lms() -> dict[str, DummyLM]:
    """Per-seat scripted answers for the same werewolf-win sweep as the legacy fixture.

    Three nights of wolves killing Vil1/Vil2/Vil3; days in between are unanimous
    abstains. The seer inspects Wolf1/Wolf2/Doc1; the doctor protects Seer1/Vil3/Seer1.

    Bids are distinct (Wolf1=10, Wolf2=9, Seer1=8, Doc1=7, Vil2=5, Vil3=4) so
    `resolve_discussion` always picks Wolf1, Wolf2, Seer1 as the top-3
    speakers — no seeded tie-break is in play, the speaker set is fixed every
    day and the script aligns the `speak` answers to exactly those seats.
    """
    # Each night opens with a wolf chat commit (the two-phase night), then the
    # kill vote. Three nights → three chat+kill pairs per wolf.
    chat = _commit_pair("werewolf_chat", {"message": "coordinating"})
    wolf1 = (
        chat
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _bid_then_maybe_speech(10, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + chat
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _bid_then_maybe_speech(10, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + chat
        + _commit_pair("submit_kill_vote", {"target": "Vil3"})
    )
    wolf2 = (
        chat
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _bid_then_maybe_speech(9, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + chat
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _bid_then_maybe_speech(9, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + chat
        + _commit_pair("submit_kill_vote", {"target": "Vil3"})
    )
    seer1 = (
        _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _bid_then_maybe_speech(8, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("seer_inspect", {"target": "Wolf2"})
        + _bid_then_maybe_speech(8, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("seer_inspect", {"target": "Doc1"})
    )
    doc1 = (
        _commit_pair("doctor_protect", {"target": "Seer1"})
        + _bid_then_maybe_speech(7, speaks=False)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("doctor_protect", {"target": "Vil3"})
        + _bid_then_maybe_speech(7, speaks=False)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("doctor_protect", {"target": "Seer1"})
    )
    vil2 = _bid_then_maybe_speech(5, speaks=False) + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
    vil3 = (
        _bid_then_maybe_speech(4, speaks=False)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _bid_then_maybe_speech(4, speaks=False)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
    )
    return {
        "Wolf1": DummyLM(wolf1),
        "Wolf2": DummyLM(wolf2),
        "Seer1": DummyLM(seer1),
        "Doc1": DummyLM(doc1),
        "Vil1": DummyLM([]),
        "Vil2": DummyLM(vil2),
        "Vil3": DummyLM(vil3),
    }


def test_run_game_with_per_seat_lms_reaches_terminal_state() -> None:
    """Full werewolf-win sweep with each seat scripted on its own `DummyLM` queue."""
    source = ReActDecisionSource(roster=ROSTER, lms=_per_seat_werewolf_sweep_lms())

    stream = run_game(ROSTER, seed=42, decisions=source)
    events = stream.log.events

    assert events[-1].type == GAME_OVER
    assert events[-1].payload["winner"] == "werewolves"


def _day_state() -> GameState:
    """A fresh day-phase game seeded with the full speaking budget."""
    return advance_phase(GameState.initial(ROSTER, bid_budget=BID_BUDGET))


def test_next_reaction_accuse_stages_a_public_accusation_draft() -> None:
    """A committed `accuse` reaction stages one public `ACCUSATION` event.

    The reaction round is public deliberation: the accusation must be broadcast
    (empty recipients) and carry `{accuser, target, reason}` so every player —
    and the suspicion-accuracy metric — can read who accused whom and why.
    """
    seat_lms = _empty_lms()
    seat_lms["Vil1"] = DummyLM([_step("accuse", {"target": "Wolf2", "reason": "dodged the vote"})])
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    source.next_reaction(_day_state(), "Vil1")
    drafts = source.drain_drafts()

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.type == ACCUSATION
    assert draft.recipients == ()
    assert draft.payload == {"accuser": "Vil1", "target": "Wolf2", "reason": "dodged the vote"}


def test_next_reaction_defend_stages_a_public_defense_draft() -> None:
    """A committed `defend` reaction stages one public `DEFENSE` event.

    Third-party defense is the signal we most want legible; the event carries
    `{defender, defended, reason}` and is broadcast so a wolf-defends-wolf tell
    is visible to the whole table.
    """
    seat_lms = _empty_lms()
    seat_lms["Vil1"] = DummyLM([_step("defend", {"target": "Seer1", "reason": "their read checks out"})])
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    source.next_reaction(_day_state(), "Vil1")
    drafts = source.drain_drafts()

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.type == DEFENSE
    assert draft.recipients == ()
    assert draft.payload == {"defender": "Vil1", "defended": "Seer1", "reason": "their read checks out"}


def test_next_reaction_pass_turn_stages_no_draft() -> None:
    """A `pass_turn` reaction stages nothing — silence is silent.

    The reaction loop still records a trajectory (the seat acted), but no public
    event is emitted, so a passing player adds no noise to the transcript.
    """
    seat_lms = _empty_lms()
    seat_lms["Vil1"] = DummyLM([_step("pass_turn", {})])
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    source.next_reaction(_day_state(), "Vil1")

    assert source.drain_drafts() == ()
    assert len(source.trajectories) == 1  # the seat still committed a decision


def test_next_reaction_rejects_a_dead_reactor() -> None:
    """`next_reaction` fails loud for a dead reactor — the driver must not seat one."""
    state = _day_state().with_player_killed("Vil1")
    source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
    with pytest.raises(ValueError, match="dead reactor"):
        source.next_reaction(state, "Vil1")


def test_failed_reaction_loop_degrades_to_a_silent_pass() -> None:
    """A reaction whose loop never commits stages nothing and does NOT raise.

    The reaction round is optional signal — a seat whose LM produces a
    truncated / unparseable response (or never commits within max_iters) must
    silently pass, not abort the game. This is the fix for the live crash where a
    too-small reaction token cap truncated the model mid-thought and the
    `AdapterParseError` propagated out of `run_game`. Here an empty `DummyLM`
    exhausts the loop with no commit; `next_reaction` must swallow it, stage no
    draft, and record no trajectory.
    """
    lms = _empty_lms()  # every seat's DummyLM is empty -> the loop cannot commit
    source = ReActDecisionSource(roster=ROSTER, lms=lms, max_iters=2)

    source.next_reaction(_day_state(), "Vil1")  # must not raise

    assert source.drain_drafts() == ()  # no public accusation/defense staged
    assert source.trajectories == ()  # a failed loop records no trajectory


def test_reaction_accusation_is_observed_by_other_living_players() -> None:
    """A public accusation is routed into every living player's memory.

    The reaction round drains and `observe`s each reaction before the next, so a
    non-accuser records the accusation and can act on it. Pins that the reaction
    event flows through the same observation routing as speeches — without it the
    whole point of public deliberation (others reading the accusation) is lost.
    """
    lms = _per_seat_werewolf_sweep_lms()
    # Day 1: Vil2 accuses Wolf1 instead of passing (same call order: bid -> react
    # -> vote, so the rest of the sweep is undisturbed). Vil2 is alive day 1 only.
    lms["Vil2"] = DummyLM(
        _commit_pair("submit_bid", {"amount": 5})
        + _commit_pair("accuse", {"target": "Wolf1", "reason": "the seer claim is fake"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=lms)

    run_game(ROSTER, seed=42, decisions=source)

    wolf1_events = source.memories["Wolf1"].events
    assert any(e.type == ACCUSATION and e.payload.get("accuser") == "Vil2" for e in wolf1_events)


def test_two_sources_with_identical_per_seat_inputs_produce_identical_streams() -> None:
    """Invariant #4 holds across the per-seat path: identical seat -> LM inputs replay byte-identically.

    A regression that bound LM choice to non-seed state (insertion order,
    a wall-clock factory, hash randomization across `set(lms.keys())`)
    would diverge on the second run and be caught here.
    """

    def produce(seed: int) -> EventStream:
        source = ReActDecisionSource(roster=ROSTER, lms=_per_seat_werewolf_sweep_lms())
        return run_game(ROSTER, seed=seed, decisions=source)

    assert_deterministic(produce, seed=42)


def test_werewolf_chat_intermediate_emits_drafts_before_kill_vote() -> None:
    """A werewolf that chats before committing produces one `WEREWOLF_CHAT` draft per call.

    The draft is private to the *living* werewolf pack and carries
    `{"speaker", "message"}`. The kill-vote terminal still ends the loop;
    the chat call(s) are intermediate, not terminal.
    """
    seat_lms: dict[str, DummyLM] = {
        "Wolf1": DummyLM(
            [
                _step("werewolf_chat", {"message": "hunt the seer"}, "talk"),
                _step("submit_kill_vote", {"target": "Vil1"}, "vote"),
            ]
        ),
        "Wolf2": DummyLM(_commit_pair("submit_kill_vote", {"target": "Vil1"})),
        "Seer1": DummyLM(_commit_pair("seer_inspect", {"target": "Wolf1"})),
        "Doc1": DummyLM(_commit_pair("doctor_protect", {"target": "Vil1"})),
        "Vil1": DummyLM([]),
        "Vil2": DummyLM([]),
        "Vil3": DummyLM([]),
    }
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)
    state = GameState.initial(ROSTER)

    actions = source.night_actions(state)
    drafts = source.drain_drafts()

    chat_drafts = [d for d in drafts if d.type == WEREWOLF_CHAT]
    assert len(chat_drafts) == 1
    assert chat_drafts[0].payload == {"speaker": "Wolf1", "message": "hunt the seer"}
    assert chat_drafts[0].recipients == ("Wolf1", "Wolf2")
    assert actions.kill_votes["Wolf1"] == "Vil1"


def test_werewolf_chat_recipients_exclude_dead_pack_members() -> None:
    """A werewolf killed earlier in the game is not a recipient of new pack chat.

    Pre-T29 the pack was a static set; T29 makes it dynamic — a dead wolf
    cannot read messages addressed to the living pack. Without this guard a
    spectator wolf would re-enter the chat stream via memory.
    """
    state = GameState.initial(ROSTER).with_player_killed("Wolf2")
    seat_lms: dict[str, DummyLM] = {
        "Wolf1": DummyLM(
            [
                _step("werewolf_chat", {"message": "I'm alone now"}, "talk"),
                _step("submit_kill_vote", {"target": "Vil1"}, "vote"),
            ]
        ),
        "Wolf2": DummyLM([]),
        "Seer1": DummyLM(_commit_pair("seer_inspect", {"target": "Wolf1"})),
        "Doc1": DummyLM(_commit_pair("doctor_protect", {"target": "Vil1"})),
        "Vil1": DummyLM([]),
        "Vil2": DummyLM([]),
        "Vil3": DummyLM([]),
    }
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    source.night_actions(state)
    drafts = source.drain_drafts()
    chat_drafts = [d for d in drafts if d.type == WEREWOLF_CHAT]

    assert len(chat_drafts) == 1
    assert chat_drafts[0].recipients == ("Wolf1",)


def test_bids_emits_one_private_bid_per_alive_player() -> None:
    """`bids()` runs one ReAct loop per alive player; each emits a private `BID` draft.

    Each seat is scripted with a *distinct* amount equal to its roster index;
    a regression that swapped seats <-> amounts (round-robin amount
    assignment, dict-order leak, off-by-one) would produce a wrong
    `{name: amount}` map and fail the explicit per-seat pin below.
    """
    state = advance_phase(GameState.initial(ROSTER, bid_budget=BID_BUDGET))
    expected = {name: amount for amount, (name, _role) in enumerate(ROSTER)}
    seat_lms: dict[str, DummyLM] = {
        name: DummyLM(_commit_pair("submit_bid", {"amount": amount})) for name, amount in expected.items()
    }
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    bids = source.bids(state)
    drafts = source.drain_drafts()

    assert bids == expected
    bid_drafts = [d for d in drafts if d.type == BID]
    assert {d.payload["bidder"] for d in bid_drafts} == set(expected)
    for draft in bid_drafts:
        bidder = draft.payload["bidder"]
        amount = draft.payload["amount"]
        assert isinstance(bidder, str)
        assert isinstance(amount, int)
        assert draft.recipients == (bidder,)
        assert amount == expected[bidder]


def test_bids_skips_dead_players() -> None:
    """A dead seat does not bid; its LM is untouched."""
    state = advance_phase(GameState.initial(ROSTER, bid_budget=BID_BUDGET)).with_player_killed("Vil1")
    seat_lms: dict[str, DummyLM] = {name: DummyLM(_commit_pair("submit_bid", {"amount": 1})) for name, _ in ROSTER}
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    bids = source.bids(state)

    assert "Vil1" not in bids
    assert seat_lms["Vil1"].history == []


def test_next_speech_emits_public_drafts_one_speaker_at_a_time() -> None:
    """`next_speech(state, speaker)` runs one ReAct loop for that speaker and
    emits one public `SPEECH` draft.

    The driver calls `next_speech` per speaker (rather than a batch) so it can
    drain + observe each speech before the next speaker runs — the basis for
    intra-phase speech visibility.
    """
    state = advance_phase(GameState.initial(ROSTER))
    seat_lms: dict[str, DummyLM] = {
        "Wolf1": DummyLM(_commit_pair("speak", {"message": "I am the seer"})),
        "Wolf2": DummyLM([]),
        "Seer1": DummyLM(_commit_pair("speak", {"message": "No, I am"})),
        "Doc1": DummyLM([]),
        "Vil1": DummyLM([]),
        "Vil2": DummyLM(_commit_pair("speak", {"message": "I trust Seer1"})),
        "Vil3": DummyLM([]),
    }
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    messages = [source.next_speech(state, s) for s in ("Wolf1", "Seer1", "Vil2")]
    drafts = source.drain_drafts()
    speech_drafts = [d for d in drafts if d.type == SPEECH]

    assert messages == ["I am the seer", "No, I am", "I trust Seer1"]
    assert len(speech_drafts) == 3
    assert [d.payload["speaker"] for d in speech_drafts] == ["Wolf1", "Seer1", "Vil2"]


def test_later_speaker_sees_earlier_speech_in_brief() -> None:
    """A speaker who goes second sees the first speaker's speech in its brief.

    The day-phase fix: previously every speaker spoke from the same pre-speech
    snapshot, so speeches were monologues. With per-speaker `next_speech` and
    the engine draining+observing between speeches, speaker 2's memory holds
    speaker 1's `SPEECH`, and the grounded brief renders it — so the agent
    can actually react.
    """
    state = advance_phase(GameState.initial(ROSTER))
    seat_lms: dict[str, DummyLM] = {name: DummyLM([]) for name, _ in ROSTER}
    seat_lms["Wolf1"] = DummyLM(_commit_pair("speak", {"message": "Seer1 is lying"}))
    seat_lms["Seer1"] = DummyLM(_commit_pair("speak", {"message": "rebuttal"}))
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    # Speaker 1 speaks; engine drains + observes the speech (public → all alive).
    source.next_speech(state, "Wolf1")
    log = EventLog()
    for draft in source.drain_drafts():
        log.append(
            round=state.round,
            phase=state.phase,
            type=draft.type,
            payload=draft.payload,
            recipients=draft.recipients,
        )
    source.observe(state, log.events)

    # Speaker 2's brief now carries Wolf1's speech (via the memory context block).
    from social_deduction_bench.agents.decisions import _SPEECH_BRIEF, _format_brief

    brief = _format_brief(
        _SPEECH_BRIEF,
        caller="Seer1",
        role=Role.SEER.value,
        state=state,
        memory=source.memories["Seer1"],
    )
    assert "Seer1 is lying" in brief


def test_tool_rejection_emits_private_tool_rejected_draft() -> None:
    """A rejected terminal call emits exactly one `TOOL_REJECTED` draft to the caller.

    Wolf1 first tries to vote for itself (rejected: self-target), then a valid
    vote. The buffer holds exactly one `TOOL_REJECTED` whose payload names the
    tool, the rejected args, and the engine's reason, with `recipients=("Wolf1",)`.
    """
    seat_lms: dict[str, DummyLM] = {
        "Wolf1": DummyLM(
            [
                _step("submit_kill_vote", {"target": "Wolf1"}, "bad"),
                _step("submit_kill_vote", {"target": "Vil1"}, "retry"),
            ]
        ),
        "Wolf2": DummyLM(_commit_pair("submit_kill_vote", {"target": "Vil1"})),
        "Seer1": DummyLM(_commit_pair("seer_inspect", {"target": "Wolf1"})),
        "Doc1": DummyLM(_commit_pair("doctor_protect", {"target": "Vil1"})),
        "Vil1": DummyLM([]),
        "Vil2": DummyLM([]),
        "Vil3": DummyLM([]),
    }
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    source.night_actions(GameState.initial(ROSTER))
    drafts = source.drain_drafts()
    rejected = [d for d in drafts if d.type == TOOL_REJECTED]

    assert len(rejected) == 1
    assert rejected[0].recipients == ("Wolf1",)
    assert rejected[0].payload["tool"] == "submit_kill_vote"
    assert rejected[0].payload["args"] == {"target": "Wolf1"}
    reason = rejected[0].payload["reason"]
    assert isinstance(reason, str)
    # Match the rejection *cause* (self-target), not just the player name in passing —
    # a regression that rejected for a different reason but mentioned Wolf1 would slip
    # through a name-only check.
    assert "cannot target the caller" in reason


def test_drain_drafts_empties_buffer_and_returns_tuple() -> None:
    """`drain_drafts()` returns a tuple snapshot and clears the buffer (idempotent).

    Script: Wolf1 chats once then commits. Wolf2 commits straight. Seer/Doc
    each commit straight. No rejections. So the buffer should contain
    exactly one draft (the WEREWOLF_CHAT). Pinning the exact count catches
    a regression that double-emits or stages stray drafts.
    """
    seat_lms: dict[str, DummyLM] = {
        "Wolf1": DummyLM(
            [
                _step("werewolf_chat", {"message": "go"}, "talk"),
                _step("submit_kill_vote", {"target": "Vil1"}, "vote"),
            ]
        ),
        "Wolf2": DummyLM(_commit_pair("submit_kill_vote", {"target": "Vil1"})),
        "Seer1": DummyLM(_commit_pair("seer_inspect", {"target": "Wolf1"})),
        "Doc1": DummyLM(_commit_pair("doctor_protect", {"target": "Vil1"})),
        "Vil1": DummyLM([]),
        "Vil2": DummyLM([]),
        "Vil3": DummyLM([]),
    }
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    source.night_actions(GameState.initial(ROSTER))
    first = source.drain_drafts()
    second = source.drain_drafts()

    assert isinstance(first, tuple)
    assert len(first) == 1
    assert first[0].type == WEREWOLF_CHAT
    assert second == ()


def test_drain_drafts_is_empty_after_construction() -> None:
    """A freshly-built source has nothing pending — `drain_drafts()` returns `()`.

    Guards against a regression that pre-populates `_pending_drafts` (e.g.,
    a future "init-time announcement" feature that forgets to gate behind
    the loop's drain points).
    """
    source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
    assert source.drain_drafts() == ()


def test_tool_rejected_lands_in_the_event_stream_through_run_game() -> None:
    """A rejected ReAct tool call surfaces as a private `TOOL_REJECTED` event in the transcript.

    End-to-end pin closing the T17/T19/T21 carry-over: `validate_tool_call`
    rejects a self-target kill, the adapter stages a `TOOL_REJECTED` draft,
    and `run_game` routes that draft through `_log_drafts` so it lands in the
    `EventStream` private to the caller. T24's illegal-move-rate metric will
    consume this signal — make it integration-tested now.
    """
    lms = _per_seat_werewolf_sweep_lms()
    # Override Wolf1's night-1 commit to add a self-target rejection then a valid retry.
    # The valid retry produces the same `Vil1` kill commit the original script expected,
    # so the rest of the game-long queue lines up unchanged.
    wolf1_answers = lms["Wolf1"].history if hasattr(lms["Wolf1"], "history") else []
    # Build fresh — DummyLM is easier to replace than to patch in place.
    chat = _commit_pair("werewolf_chat", {"message": "coordinating"})
    night1_with_rejection = [
        *chat,
        _step("submit_kill_vote", {"target": "Wolf1"}, "bad"),  # self-target — rejected
        _step("submit_kill_vote", {"target": "Vil1"}, "retry"),
    ]
    # The rest of Wolf1's day-1 / night-2 / day-2 / night-3 queue from the sweep
    # (each night now opens with a chat commit before the kill vote).
    rest_of_wolf1 = (
        _bid_then_maybe_speech(10, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + chat
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _bid_then_maybe_speech(10, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + chat
        + _commit_pair("submit_kill_vote", {"target": "Vil3"})
    )
    del wolf1_answers
    lms["Wolf1"] = DummyLM(night1_with_rejection + rest_of_wolf1)
    source = ReActDecisionSource(roster=ROSTER, lms=lms)

    stream = run_game(ROSTER, seed=42, decisions=source)
    rejections = [e for e in stream.log.events if e.type == TOOL_REJECTED]

    assert len(rejections) == 1
    assert rejections[0].recipients == ("Wolf1",)
    assert rejections[0].payload["tool"] == "submit_kill_vote"
    assert rejections[0].payload["args"] == {"target": "Wolf1"}
    reason = rejections[0].payload["reason"]
    assert isinstance(reason, str)
    assert "cannot target the caller" in reason

    # The villager view of the same stream sees no TOOL_REJECTED event (invariant #2 closure).
    from social_deduction_bench.engine import observations_for

    villager_view = observations_for(stream.log.events, "Vil3")
    assert all(e.type != TOOL_REJECTED for e in villager_view)


def test_lms_dict_insertion_order_does_not_affect_event_stream() -> None:
    """Iteration order is roster-order, not `lms.keys()` order.

    A regression that walked `self._lms` (insertion order) instead of
    `self._roster` would diverge when the LM mapping is built in a
    different key order. Build the same per-seat sweep twice — once
    in roster order, once in reversed insertion order — and assert
    the two streams are byte-identical.
    """
    forward = _per_seat_werewolf_sweep_lms()
    reversed_keys: dict[str, DummyLM] = {name: _per_seat_werewolf_sweep_lms()[name] for name in reversed(forward)}
    assert list(reversed_keys.keys()) != list(forward.keys())

    forward_stream = run_game(ROSTER, seed=42, decisions=ReActDecisionSource(roster=ROSTER, lms=forward))
    reversed_stream = run_game(ROSTER, seed=42, decisions=ReActDecisionSource(roster=ROSTER, lms=reversed_keys))
    assert_streams_identical(forward_stream, reversed_stream)


# --- T30: per-decision trajectory accumulation ---------------------------


def test_trajectories_is_empty_on_construction() -> None:
    """A freshly-built source has no trajectories.

    The accumulator must start empty so `decision_seq=0` is the first
    decision of the game, not a leftover from prior wiring.
    """
    source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
    assert source.trajectories == ()


def test_night_actions_accumulates_one_trajectory_per_acting_player() -> None:
    """One night = one trajectory per acting living player (wolves + seer + doctor).

    Villagers are not asked at night — they must not contribute a
    trajectory. Verifies that the sidecar's "one line per decision-point
    loop" rule maps cleanly onto the engine's role-gating.
    """
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))
    state = GameState.initial(ROSTER)

    source.night_actions(state)

    trajectories = source.trajectories
    assert len(trajectories) == 4

    callers = tuple(t.caller for t in trajectories)
    assert callers == ("Wolf1", "Wolf2", "Seer1", "Doc1")

    # decision_seq is gap-free starting at 0
    assert tuple(t.decision_seq for t in trajectories) == (0, 1, 2, 3)

    # role + terminal_tool agree with each seat's role gate
    terminals_by_caller = {t.caller: t.terminal_tool for t in trajectories}
    assert terminals_by_caller == {
        "Wolf1": "submit_kill_vote",
        "Wolf2": "submit_kill_vote",
        "Seer1": "seer_inspect",
        "Doc1": "doctor_protect",
    }
    roles_by_caller = {t.caller: t.role for t in trajectories}
    assert roles_by_caller == {
        "Wolf1": Role.WEREWOLF.value,
        "Wolf2": Role.WEREWOLF.value,
        "Seer1": Role.SEER.value,
        "Doc1": Role.DOCTOR.value,
    }
    # phase is the engine's NIGHT value, captured at decision time
    assert all(t.phase == Phase.NIGHT.value for t in trajectories)
    assert all(t.round == 1 for t in trajectories)


def test_trajectory_committed_value_matches_commit() -> None:
    """`Trajectory.committed_value` equals the terminal's parsed value.

    For target-shaped terminals this is the player name; for bids it is
    the amount. A mismatch would mean the sidecar's "what did the agent
    do" column is silently wrong.
    """
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _commit_pair("submit_kill_vote", {"target": "Vil3"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Seer1"})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))
    source.night_actions(GameState.initial(ROSTER))

    by_caller = {t.caller: t for t in source.trajectories}
    assert by_caller["Wolf1"].committed_value == "Vil2"
    assert by_caller["Wolf2"].committed_value == "Vil3"
    assert by_caller["Seer1"].committed_value == "Wolf1"
    assert by_caller["Doc1"].committed_value == "Seer1"


def test_trajectory_lm_calls_non_empty_per_loop() -> None:
    """Every trajectory has at least one `LMCallRecord` (each loop calls the LM).

    A loop with no LM calls would mean `react_decide` returned without
    talking to the model — that's only possible on a bug. Pin it so a
    regression that broke the snapshot doesn't ship.
    """
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))
    source.night_actions(GameState.initial(ROSTER))

    for trajectory in source.trajectories:
        assert len(trajectory.lm_calls) > 0
        for call in trajectory.lm_calls:
            assert call.model == "dummy"
            assert call.cost_usd is None
            assert call.latency_ms >= 0.0


def test_werewolf_chat_intermediate_lands_in_react_trajectory_before_terminal() -> None:
    """A chat-then-kill turn produces ONE trajectory whose steps go chat -> kill.

    The intermediate (`werewolf_chat`) must appear as a `ReActStep`
    before the terminal commit, in the same trajectory. Without this,
    the sidecar would hide the wolves' coordination from replay.
    """
    seat_lms: dict[str, DummyLM] = {
        "Wolf1": DummyLM(
            [
                _step("werewolf_chat", {"message": "lets kill Vil1"}),
                _step("submit_kill_vote", {"target": "Vil1"}),
            ]
        ),
        "Wolf2": DummyLM(_commit_pair("submit_kill_vote", {"target": "Vil1"})),
        "Seer1": DummyLM(_commit_pair("seer_inspect", {"target": "Wolf1"})),
        "Doc1": DummyLM(_commit_pair("doctor_protect", {"target": "Vil1"})),
        "Vil1": DummyLM([]),
        "Vil2": DummyLM([]),
        "Vil3": DummyLM([]),
    }
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)
    source.night_actions(GameState.initial(ROSTER))

    wolf1_traj = next(t for t in source.trajectories if t.caller == "Wolf1")
    tools_seen = tuple(step.tool for step in wolf1_traj.react_trajectory)
    assert "werewolf_chat" in tools_seen
    assert "submit_kill_vote" in tools_seen
    # The chat happens BEFORE the kill commit in the recorded trajectory.
    assert tools_seen.index("werewolf_chat") < tools_seen.index("submit_kill_vote")
    # And it's still one trajectory — the intermediate did not split the loop.
    assert sum(1 for t in source.trajectories if t.caller == "Wolf1") == 1


def test_rejected_terminal_shows_up_as_error_step_in_same_trajectory() -> None:
    """A rejected terminal attempt is a step (error: ...) in the trajectory that finally commits.

    The audit trail must show the agent's bad call so debugging can see
    what the model tried before retrying. Without it, the sidecar would
    look like the agent always got things right on the first try.
    """
    seat_lms: dict[str, DummyLM] = {
        "Wolf1": DummyLM(
            [
                _step("submit_kill_vote", {"target": "Wolf1"}),  # self-target -> rejected
                _step("submit_kill_vote", {"target": "Vil1"}),  # valid
            ]
        ),
        "Wolf2": DummyLM(_commit_pair("submit_kill_vote", {"target": "Vil1"})),
        "Seer1": DummyLM(_commit_pair("seer_inspect", {"target": "Wolf1"})),
        "Doc1": DummyLM(_commit_pair("doctor_protect", {"target": "Vil1"})),
        "Vil1": DummyLM([]),
        "Vil2": DummyLM([]),
        "Vil3": DummyLM([]),
    }
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)
    source.night_actions(GameState.initial(ROSTER))

    wolf1_traj = next(t for t in source.trajectories if t.caller == "Wolf1")
    has_error_step = any(step.observation.startswith("error:") for step in wolf1_traj.react_trajectory)
    has_ok_step = any(step.observation.startswith("ok:") for step in wolf1_traj.react_trajectory)
    assert has_error_step
    assert has_ok_step
    # One trajectory, with both the rejection and the success.
    assert sum(1 for t in source.trajectories if t.caller == "Wolf1") == 1


def test_full_run_game_accumulates_trajectories_with_gap_free_decision_seq() -> None:
    """A full scripted game produces a gap-free decision_seq from 0.

    `decision_seq` is the visualizer's ordering key. A gap or duplicate
    would split or merge two decision points in the replay, silently
    misrepresenting the game.
    """
    answers = _werewolf_sweep_script()
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))
    run_game(ROSTER, seed=42, decisions=source)

    trajectories = source.trajectories
    assert len(trajectories) > 0
    assert tuple(t.decision_seq for t in trajectories) == tuple(range(len(trajectories)))

    # The first decision is now the night chat sub-phase (roster order: Wolf1
    # chats before any kill vote is committed).
    first = trajectories[0]
    assert first.caller == "Wolf1"
    assert first.phase == Phase.NIGHT.value
    assert first.terminal_tool == "werewolf_chat"
    assert first.round == 1
    # The first kill vote follows the chat round.
    first_kill = next(t for t in trajectories if t.terminal_tool == "submit_kill_vote")
    assert first_kill.caller == "Wolf1"
    assert first_kill.round == 1


def test_trajectories_property_returns_snapshot_not_live_view() -> None:
    """`source.trajectories` returns an immutable snapshot of the source's accumulator.

    The visualizer (T31) may grab `source.trajectories` and serialize it
    while a later phase is still running on the same source. The captured
    snapshot must not grow when new trajectories land — a live view would
    race.
    """
    n_alive = len(ROSTER)
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    for _ in range(n_alive):
        answers += _commit_pair("submit_bid", {"amount": 0})
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))
    state = GameState.initial(ROSTER)
    source.night_actions(state)

    snapshot = source.trajectories
    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 4

    # Run more decisions on the SAME source: the snapshot must not grow.
    day_state = advance_phase(state)
    source.bids(day_state)
    assert len(source.trajectories) == 4 + n_alive
    assert len(snapshot) == 4


def test_bid_and_speech_loops_each_emit_a_trajectory() -> None:
    """Each `bids()` and `speeches()` ReAct loop contributes one trajectory.

    Bids and speeches are decision points — the sidecar audits them just
    like night and exile votes. A regression that wired the trace_sink
    only into one phase would fail here.
    """
    state = advance_phase(GameState.initial(ROSTER))
    n_alive = len(ROSTER)
    bid_answers: list[dict[str, Any]] = []
    for _ in range(n_alive):
        bid_answers += _commit_pair("submit_bid", {"amount": 0})
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(bid_answers)))

    source.bids(state)
    bid_trajectories = source.trajectories
    assert len(bid_trajectories) == n_alive
    assert all(t.terminal_tool == "submit_bid" for t in bid_trajectories)
    assert all(t.phase == Phase.DAY.value for t in bid_trajectories)
    assert all(isinstance(t.committed_value, int) for t in bid_trajectories)

    # Speeches: one trajectory per chosen speaker.
    speech_answers: list[dict[str, Any]] = []
    for _ in range(3):
        speech_answers += _commit_pair("speak", {"message": "hi"})
    speech_source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(speech_answers)))
    for speaker in ("Wolf1", "Seer1", "Vil2"):
        speech_source.next_speech(state, speaker)

    speech_trajectories = speech_source.trajectories
    assert len(speech_trajectories) == 3
    assert tuple(t.caller for t in speech_trajectories) == ("Wolf1", "Seer1", "Vil2")
    assert all(t.terminal_tool == "speak" for t in speech_trajectories)


def test_on_trajectory_callback_fires_per_decision_in_commit_order() -> None:
    """`on_trajectory` fires once per committed decision, matching accumulator order.

    Used by the CLI printer (`sdb-werewolf`) to render each decision as it
    commits, since trajectories happen DURING a phase (LLM-call waits) but
    `observe` only fires AFTER the phase resolves. A live observer needs
    per-decision granularity.
    """
    from social_deduction_bench.agents.trajectory import Trajectory

    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    captured: list[Trajectory] = []
    source = ReActDecisionSource(
        roster=ROSTER,
        lms=_uniform_lms(DummyLM(answers)),
        on_trajectory=captured.append,
    )

    source.night_actions(GameState.initial(ROSTER))

    # The callback fires for every decision commit (4 night terminals).
    assert len(captured) == 4
    # And the order matches the accumulator — callback fires after appending.
    assert tuple(t.decision_seq for t in captured) == (0, 1, 2, 3)
    assert tuple(t.caller for t in captured) == ("Wolf1", "Wolf2", "Seer1", "Doc1")
    # The captured trajectories are identical to what the source accumulated.
    assert captured == list(source.trajectories)


def test_on_trajectory_default_none_is_a_no_op() -> None:
    """Omitting `on_trajectory` keeps the legacy contract: silent accumulation.

    Back-compat for every existing `ReActDecisionSource(...)` construction
    site (tests, smoke runs). The new keyword must not become required.
    """
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"}) * 2
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))

    source.night_actions(GameState.initial(ROSTER))

    # Accumulator still works; no callback was wired so nothing extra fires.
    assert len(source.trajectories) == 4


def test_on_decision_start_fires_once_per_player_before_their_iterations() -> None:
    """`on_decision_start` fires exactly once per committed decision, before any step.

    Used by the CLI to print a "thinking…" banner immediately when a
    player's turn begins — without it the operator sees nothing for
    seconds while the LM call is in flight.
    """
    from social_deduction_bench.agents.trajectory import ReActStep

    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    timeline: list[tuple[str, str, str]] = []

    def on_start(caller: str, role: str) -> None:
        timeline.append(("start", caller, role))

    def on_step(caller: str, role: str, step: ReActStep, _lm_calls: object) -> None:
        timeline.append(("step", caller, step.tool))

    source = ReActDecisionSource(
        roster=ROSTER,
        lms=_uniform_lms(DummyLM(answers)),
        on_decision_start=on_start,
        on_step=on_step,
    )

    source.night_actions(GameState.initial(ROSTER))

    # Four decisions → four "start" events, each preceding that player's steps.
    starts = [t for t in timeline if t[0] == "start"]
    assert [t[1] for t in starts] == ["Wolf1", "Wolf2", "Seer1", "Doc1"]

    # For each "start", the immediately following events belong to that caller.
    for idx, (_, caller, _) in enumerate(starts):
        first_after_start = timeline.index(starts[idx]) + 1
        next_start = timeline.index(starts[idx + 1]) if idx + 1 < len(starts) else len(timeline)
        between = timeline[first_after_start:next_start]
        assert all(event[1] == caller for event in between), (
            f"events between starts {idx} and {idx + 1} must all be from {caller}, got {between}"
        )


def test_on_step_fires_per_iteration_with_role_and_lm_calls() -> None:
    """`on_step` fires once per ReAct iteration with caller+role+step+lm_calls.

    Streams agent progress in real time. The role is included so the CLI
    can colorize per role without re-lookup. `lm_calls` carries the
    per-iteration telemetry the streaming line renders.
    """
    from social_deduction_bench.agents.trajectory import LMCallRecord, ReActStep

    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    captured: list[tuple[str, str, ReActStep, tuple[LMCallRecord, ...]]] = []

    def on_step(caller: str, role: str, step: ReActStep, lm_calls: tuple[LMCallRecord, ...]) -> None:
        captured.append((caller, role, step, lm_calls))

    source = ReActDecisionSource(
        roster=ROSTER,
        lms=_uniform_lms(DummyLM(answers)),
        on_step=on_step,
    )

    source.night_actions(GameState.initial(ROSTER))

    # Every iteration of every decision fires the hook.
    assert captured, "on_step should have fired at least once"
    # Roles line up with the roster mapping.
    roles_seen = {(caller, role) for caller, role, _, _ in captured}
    expected = {("Wolf1", "werewolf"), ("Wolf2", "werewolf"), ("Seer1", "seer"), ("Doc1", "doctor")}
    assert expected <= roles_seen


def test_on_decision_start_and_on_step_default_none_are_no_ops() -> None:
    """Omitting the streaming hooks keeps legacy callers happy.

    Existing tests and the smoke harness construct `ReActDecisionSource`
    without these kwargs; that path must continue to work silently.
    """
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"}) * 2
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    source = ReActDecisionSource(roster=ROSTER, lms=_uniform_lms(DummyLM(answers)))
    source.night_actions(GameState.initial(ROSTER))
    assert len(source.trajectories) == 4


# --- decision brief grounding ------------------------------------------------


def test_decision_brief_names_the_caller_and_lists_living_players() -> None:
    """Every brief must include `you=NAME` and the current alive roster.

    Without this grounding, small models hallucinate target names (e.g.
    qwen3.5-9b calling `submit_kill_vote(target=Emma)` when Emma is not
    a player) instead of calling `get_public_state` to learn the truth.
    The brief is the first thing the model sees — putting the roster
    there saves an iteration and prevents the hallucination class.
    """
    from social_deduction_bench.agents.react import Commit

    captured_briefs: list[str] = []

    async def fake_react_decide_async(*, decision_brief: str, caller: str, **_kw: object) -> Commit:
        captured_briefs.append(decision_brief)
        # Return a synthetic commit so the loop can proceed.
        from social_deduction_bench.agents.trajectory import ReActStep

        trace_sink = _kw.get("trace_sink")
        if callable(trace_sink):
            step = ReActStep(iter=0, thought="t", tool="submit_kill_vote", args={"target": "Vil1"}, observation="ok")
            trace_sink((step,), ())
        # Caller's role dictates the synthetic commit shape.
        terminal_tools = _kw.get("terminal_tools") or {}
        if isinstance(terminal_tools, Mapping):
            terminal_name = next(iter(terminal_tools.keys()), "submit_kill_vote")
        else:
            terminal_name = "submit_kill_vote"
        if terminal_name == "submit_bid":
            return Commit(tool=terminal_name, value=0)
        if terminal_name == "speak":
            return Commit(tool=terminal_name, value="hello")
        return Commit(tool=terminal_name, value="Vil1")

    import social_deduction_bench.agents.decisions as decisions_mod

    original = decisions_mod.react_decide_async
    decisions_mod.react_decide_async = fake_react_decide_async  # type: ignore[assignment]
    try:
        source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
        source.night_actions(GameState.initial(ROSTER))
    finally:
        decisions_mod.react_decide_async = original  # type: ignore[assignment]

    # 4 night-acting roles → 4 briefs.
    assert len(captured_briefs) == 4
    alive_csv = ", ".join(name for name, _ in ROSTER)
    for brief, (name, _role) in zip(
        captured_briefs,
        [(n, r) for n, r in ROSTER if r in (Role.WEREWOLF.value, Role.SEER.value, Role.DOCTOR.value)],
        strict=True,
    ):
        assert f"You are {name}" in brief, f"brief missing 'You are {name}': {brief}"
        assert alive_csv in brief, f"brief missing alive list: {brief}"


def test_decision_brief_alive_list_reflects_current_state() -> None:
    """A dead player must NOT appear in the alive list seen by later briefs.

    The brief is built per decision with `state.alive_names()`. If the
    list were captured at construction time it would silently include
    corpses, and the model would target a dead player on every turn.
    """
    from social_deduction_bench.agents.decisions import _format_brief

    state = GameState.initial(ROSTER)
    dead_state = state.with_player_killed("Vil1")

    brief = _format_brief(
        "You are {you}. Living: {alive_csv}.",
        caller="Wolf1",
        role=Role.WEREWOLF.value,
        state=dead_state,
        memory=GameMemory(),
    )
    assert "Vil1" not in brief
    assert "Wolf1" in brief


# --- brief grounding: pack identity + memory context (D1) --------------------


def test_wolf_brief_names_its_ally_excluding_itself() -> None:
    """A werewolf's brief names its living ally — and NOT itself — so it never
    confuses its own name with its packmate's.

    Real-run bug: with the ally line reading "Your pack: Alice, Bob", a wolf
    seated at Bob latched onto "Alice" as itself ("me (Alice)") and defended the
    wrong player. Excluding the caller's own name from the ally line removes that
    ambiguity. Rendered only into the wolf's OWN brief, so no hidden state leaks.
    """
    from social_deduction_bench.agents.decisions import _NIGHT_BRIEFS, _format_brief

    state = GameState.initial(ROSTER)
    brief = _format_brief(
        _NIGHT_BRIEFS[Role.WEREWOLF.value],
        caller="Wolf1",
        role=Role.WEREWOLF.value,
        state=state,
        memory=GameMemory(),
    )
    ally_line = next(ln for ln in brief.splitlines() if "besides you" in ln)
    assert "Wolf2" in ally_line  # the living ally is named
    assert "Wolf1" not in ally_line  # the caller's own name is excluded


def test_last_living_wolf_brief_says_no_allies_remain() -> None:
    """A lone surviving wolf's brief says no allies remain, not an empty ally list.

    With self excluded, a sole wolf has no names to list; the brief must read
    cleanly ("only living werewolf") rather than an empty/dangling line.
    """
    from social_deduction_bench.agents.decisions import _NIGHT_BRIEFS, _format_brief

    state = GameState.initial(ROSTER).with_player_killed("Wolf2")
    brief = _format_brief(
        _NIGHT_BRIEFS[Role.WEREWOLF.value],
        caller="Wolf1",
        role=Role.WEREWOLF.value,
        state=state,
        memory=GameMemory(),
    )
    assert "only living werewolf" in brief.lower()


def test_brief_states_caller_identity_and_role_every_phase() -> None:
    """Every brief restates "you are <name>, <role>" — including the day briefs.

    Real-run bug: the day briefs (bid/speak/react/vote) dropped the role, so a
    werewolf mid-day asserted "I am a villager" while naming its own pack. The
    identity line keeps name + role present each turn. It is the caller's own
    role, rendered only in its brief, so it leaks nothing.
    """
    from social_deduction_bench.agents.decisions import _BID_BRIEF, _DAY_BRIEF, _format_brief

    wolf_day = _format_brief(
        _BID_BRIEF, caller="Wolf1", role=Role.WEREWOLF.value, state=GameState.initial(ROSTER), memory=GameMemory()
    )
    villager_day = _format_brief(
        _DAY_BRIEF, caller="Vil1", role=Role.VILLAGER.value, state=GameState.initial(ROSTER), memory=GameMemory()
    )
    seer_day = _format_brief(
        _DAY_BRIEF, caller="Seer1", role=Role.SEER.value, state=GameState.initial(ROSTER), memory=GameMemory()
    )
    assert "You are Wolf1, a werewolf" in wolf_day
    assert "You are Vil1, a villager" in villager_day
    assert "You are Seer1, the seer" in seer_day


def test_villager_brief_omits_pack_line() -> None:
    """A non-wolf brief must not carry a pack line — it has no pack and the
    line would be a hidden-state leak besides.

    The pack render is gated on the caller's role being werewolf; a villager
    or seer brief never enumerates the wolves.
    """
    from social_deduction_bench.agents.decisions import _DAY_BRIEF, _format_brief

    state = GameState.initial(ROSTER)
    brief = _format_brief(
        _DAY_BRIEF,
        caller="Vil1",
        role=Role.VILLAGER.value,
        state=state,
        memory=GameMemory(),
    )
    assert "pack" not in brief.lower()


def test_brief_renders_memory_context_block() -> None:
    """The brief pre-renders the agent's plan, suspicions, and recent events so
    it stops spending iterations on read-only `recall` / `get_beliefs` calls.

    Real-run waste: villagers called `get_private_info` / `get_beliefs` /
    `recall` purely to read state that could have been handed to them. Pre-
    rendering the memory in the brief eliminates the read-only iterations;
    the agent only calls a cognitive tool now to WRITE.
    """
    from social_deduction_bench.agents.decisions import _DAY_BRIEF, _format_brief
    from social_deduction_bench.engine import EventLog, Phase

    memory = GameMemory()
    memory.set_plan("lie low until I have a read on the wolves")
    memory.set_belief("Wolf1", "werewolf", "high", "deflected hard on day 1")
    log = EventLog()
    log.append(round=1, phase=Phase.NIGHT, type="kill_resolved", payload={"victim": "Carol"})
    memory.record_event(log.events[0])

    state = GameState.initial(ROSTER)
    brief = _format_brief(
        _DAY_BRIEF,
        caller="Vil1",
        role=Role.VILLAGER.value,
        state=state,
        memory=memory,
    )
    assert "lie low until I have a read on the wolves" in brief
    assert "Wolf1" in brief
    assert "werewolf" in brief
    assert "high" in brief
    # Recent events are legible prose, not the raw `kill_resolved` event type —
    # a villager reads "Carol was killed by the werewolves", distinguishing a
    # night-kill from a day-exile.
    assert "Carol was killed by the werewolves" in brief
    assert "kill_resolved" not in brief


def test_brief_states_channel_visibility_and_win_conditions() -> None:
    """Briefs convey the rules a model needs to play: how each faction wins and
    which channels are public — stated as facts, never as strategy.

    In the real run a wolf said "I stand with my pack" in a *public* speech,
    outing itself, because nothing told it `speak` is broadcast; villagers
    acted without knowing exiling a wolf wins. Surfacing the public rules fixes
    an information deficiency. Crucially the brief must NOT inject strategy
    (no "bluff", "hide your role") — that is exactly what the benchmark
    measures, so this test guards both the presence of rules and the absence
    of strategy coaching.
    """
    from social_deduction_bench.agents.decisions import _DAY_BRIEF, _NIGHT_BRIEFS, _SPEECH_BRIEF, _format_brief

    state = GameState.initial(ROSTER)
    wolf_brief = _format_brief(
        _NIGHT_BRIEFS[Role.WEREWOLF.value],
        caller="Wolf1",
        role=Role.WEREWOLF.value,
        state=state,
        memory=GameMemory(),
    )
    villager_brief = _format_brief(
        _DAY_BRIEF,
        caller="Vil1",
        role=Role.VILLAGER.value,
        state=state,
        memory=GameMemory(),
    )
    # The speech brief is the most strategy-tempting surface (a wolf speaking
    # publicly) — it must carry the rules and still inject no strategy.
    wolf_speech_brief = _format_brief(
        _SPEECH_BRIEF,
        caller="Wolf1",
        role=Role.WEREWOLF.value,
        state=state,
        memory=GameMemory(),
    )

    for brief in (wolf_brief, villager_brief, wolf_speech_brief):
        lowered = brief.lower()
        # Win conditions are public knowledge — both factions stated to every seat.
        assert "village wins" in lowered
        assert "werewolves win" in lowered
        # Channel visibility: speeches and votes are public.
        assert "public" in lowered
        # Rules, not strategy — no coaching to deceive.
        for forbidden in ("bluff", "deceive", "pretend", "lie about", "hide your"):
            assert forbidden not in lowered, f"brief must not coach strategy: found {forbidden!r}"

    # Only a wolf is told its pack channel is private; a villager's brief never
    # mentions werewolf_chat (it has no such channel).
    assert "werewolf_chat" in wolf_brief
    assert "werewolf_chat" not in villager_brief


def test_brief_shows_all_players_remaining_speaking_budgets() -> None:
    """A DAY brief surfaces every living player's remaining speaking-bid budget,
    so agents can read who is eager to talk (low) vs. staying quiet (high).

    This is public information — each day's bids are broadcast in
    `DISCUSSION_RESOLVED`, so the running spend is common knowledge. The line
    must therefore be byte-identical in a wolf's and a villager's brief; if it
    were gated by role (like the pack line) it would risk leaking, and if it
    were caller-only it would not serve the user's "who talks too much" signal.
    """
    from social_deduction_bench.agents.decisions import _DAY_BRIEF, _format_brief

    state = GameState.initial(ROSTER, bid_budget=BID_BUDGET).with_bid_spent("Wolf1", 40).with_phase(Phase.DAY)
    wolf = _format_brief(_DAY_BRIEF, caller="Wolf1", role=Role.WEREWOLF.value, state=state, memory=GameMemory())
    vil = _format_brief(_DAY_BRIEF, caller="Vil1", role=Role.VILLAGER.value, state=state, memory=GameMemory())

    for brief in (wolf, vil):
        assert "Speaking-bid budget remaining" in brief
        assert "Wolf1 60" in brief  # spender's budget dropped 100 -> 60
        assert "Vil1 100" in brief

    # Public info ⇒ identical across roles (guards against gating/leaking).
    wolf_line = next(ln for ln in wolf.splitlines() if "Speaking-bid budget remaining" in ln)
    vil_line = next(ln for ln in vil.splitlines() if "Speaking-bid budget remaining" in ln)
    assert wolf_line == vil_line


def test_speaking_budget_line_is_day_only() -> None:
    """The speaking-bid budget is a day-phase signal (bids are only spent by day),
    so the line appears on a day brief and is omitted at night.

    Real-run waste: the budget line rendered on every NIGHT brief too, where no
    one bids — pure noise a small model still spent tokens parsing.
    """
    from social_deduction_bench.agents.decisions import _BID_BRIEF, _NIGHT_BRIEFS, _format_brief

    night_state = GameState.initial(ROSTER, bid_budget=BID_BUDGET)
    day_state = night_state.with_phase(Phase.DAY)

    night_brief = _format_brief(
        _NIGHT_BRIEFS[Role.WEREWOLF.value],
        caller="Wolf1",
        role=Role.WEREWOLF.value,
        state=night_state,
        memory=GameMemory(),
    )
    day_brief = _format_brief(
        _BID_BRIEF, caller="Wolf1", role=Role.WEREWOLF.value, state=day_state, memory=GameMemory()
    )
    assert "budget remaining" not in night_brief.lower()
    assert "budget remaining" in day_brief.lower()


def test_brief_explains_cognitive_tools_are_non_terminal() -> None:
    """Every brief names the cognitive tools and states they do not end the turn.

    Real-run waste: a model spent ~60s unsure whether `set_plan` would end its
    turn, because the cognitive tools were in the toolbelt but unnamed in the
    brief. Naming them + the non-terminal rule removes that ambiguity and is
    what gets `set_belief` / `set_plan` actually used.
    """
    from social_deduction_bench.agents.decisions import _DAY_BRIEF, _format_brief

    brief = _format_brief(
        _DAY_BRIEF, caller="Vil1", role=Role.VILLAGER.value, state=GameState.initial(ROSTER), memory=GameMemory()
    )
    lowered = brief.lower()
    assert "set_belief" in lowered
    assert "set_plan" in lowered
    assert "recall" in lowered
    assert "do not end your turn" in lowered


def test_brief_states_night_resolution_mechanics() -> None:
    """Briefs state how the night resolves — a public rule, not strategy.

    Real-run mislynch: a doctor's "I protected Alice" claim was read by the whole
    table (and the doctor itself) as impossible "because only Carol died", so the
    village exiled its own doctor. Stating that guarding a player the wolves did
    not target is normal — no visible effect, not an attack — removes that false
    tell. It is a mechanic, never a strategy hint.
    """
    from social_deduction_bench.agents.decisions import _DAY_BRIEF, _NIGHT_BRIEFS, _format_brief

    villager = _format_brief(
        _DAY_BRIEF, caller="Vil1", role=Role.VILLAGER.value, state=GameState.initial(ROSTER), memory=GameMemory()
    )
    doctor = _format_brief(
        _NIGHT_BRIEFS[Role.DOCTOR.value],
        caller="Doc1",
        role=Role.DOCTOR.value,
        state=GameState.initial(ROSTER),
        memory=GameMemory(),
    )
    for brief in (villager, doctor):
        lowered = brief.lower()
        assert "night mechanics" in lowered
        assert "guard" in lowered  # the doctor's protect mechanic is described
        # rule, not strategy — the night-mechanics line must not coach play
        for forbidden in ("bluff", "deceive", "pretend", "you should", "hide your"):
            assert forbidden not in lowered


def test_vote_brief_prompts_recording_a_suspicion() -> None:
    """The exile-vote brief asks the player to record a suspicion before voting,
    so the belief table (and the suspicion-accuracy metric it feeds) is populated.

    Real-run gap: `set_belief` was called zero times across a whole game, leaving
    villager memory empty. The vote is the one moment the player has heard the
    full discussion, so the prompt lands there. It prompts the *process* (record
    a read), never the answer.
    """
    from social_deduction_bench.agents.decisions import _DAY_BRIEF, _format_brief

    brief = _format_brief(
        _DAY_BRIEF, caller="Vil1", role=Role.VILLAGER.value, state=GameState.initial(ROSTER), memory=GameMemory()
    )
    lowered = brief.lower()
    assert "set_belief" in lowered
    assert "suspect" in lowered
    assert "submit_exile_vote" in lowered


def test_brief_memory_block_handles_empty_memory() -> None:
    """An agent with no plan / beliefs / events still gets a coherent block,
    not blank lines or a crash.

    Round 1 every agent starts with empty memory; the block must read
    cleanly ("none yet" / "nothing yet") rather than rendering empty
    segments that confuse a small model.
    """
    from social_deduction_bench.agents.decisions import _BID_BRIEF, _format_brief

    state = GameState.initial(ROSTER)
    brief = _format_brief(
        _BID_BRIEF,
        caller="Vil2",
        role=Role.VILLAGER.value,
        state=state,
        memory=GameMemory(),
    )
    lowered = brief.lower()
    assert "none yet" in lowered or "nothing yet" in lowered


def test_cognitive_toolbelt_drops_read_and_remember_keeps_structured_writes_and_recall() -> None:
    """The bound cognitive toolbelt excludes the redundant read tools AND the
    free-text `remember`, keeping the structured writes plus deep `recall`.

    Brief grounding makes `get_beliefs` / `get_plan` / `get_private_info` /
    `get_public_state` redundant — their content is in the brief. `remember` is
    dropped because it cost a whole extra ReAct iteration; the terminal action
    now carries an optional `note` that records the same thing in the
    committing call. `recall` stays for full-history dives beyond the brief's
    2-round window; `set_belief` / `set_plan` stay for structured writes.
    """
    from social_deduction_bench.agents.react import Commit

    captured_tools: list[str] = []

    async def fake_react_decide_async(
        *, cognitive_tools: object, decision_brief: str, caller: str, **_kw: object
    ) -> Commit:
        if isinstance(cognitive_tools, (list, tuple)):
            captured_tools.extend(getattr(t, "__name__", "") for t in cognitive_tools)
        trace_sink = _kw.get("trace_sink")
        if callable(trace_sink):
            trace_sink((), ())
        return Commit(tool="submit_kill_vote", value="Vil1")

    import social_deduction_bench.agents.decisions as decisions_mod

    original = decisions_mod.react_decide_async
    decisions_mod.react_decide_async = fake_react_decide_async  # type: ignore[assignment]
    try:
        source = ReActDecisionSource(roster=ROSTER, lms=_empty_lms())
        source.night_actions(GameState.initial(ROSTER))
    finally:
        decisions_mod.react_decide_async = original  # type: ignore[assignment]

    names = set(captured_tools)
    assert {"set_belief", "set_plan", "recall"} <= names
    assert "remember" not in names  # folded into the terminal's note-on-commit
    assert names.isdisjoint({"get_beliefs", "get_plan", "get_private_info", "get_public_state"})


def test_terminal_note_on_commit_records_only_on_valid_commit() -> None:
    """A terminal action may carry an optional `note`, folding "record + act"
    into one call instead of a separate `remember` iteration.

    Real-run waste: agents spent a whole extra ReAct iteration on `set_plan` /
    `remember` before acting. `note` lets a single committing call record the
    rationale. It must record exactly once on a valid commit, nothing when
    absent or blank, and nothing when the action was rejected (it never
    happened).
    """
    from social_deduction_bench.agents.decisions import _bind_terminal
    from social_deduction_bench.games.werewolf.tools import submit_exile_vote

    day = advance_phase(GameState.initial(ROSTER))

    # valid commit with a note -> exactly one note at the current round
    m = GameMemory()
    assert _bind_terminal(submit_exile_vote, day, "Vil1", m)(target="Wolf1", note="Wolf1 deflected hard").valid
    assert [n.text for n in m.notes] == ["Wolf1 deflected hard"]
    assert m.notes[0].round == day.round

    # valid commit without a note -> nothing recorded
    m_absent = GameMemory()
    _bind_terminal(submit_exile_vote, day, "Vil2", m_absent)(target="Wolf1")
    assert m_absent.notes == ()

    # blank note -> nothing recorded
    m_blank = GameMemory()
    _bind_terminal(submit_exile_vote, day, "Vil2", m_blank)(target="Wolf1", note="   ")
    assert m_blank.notes == ()

    # rejected call (dead target) -> nothing recorded even with a note supplied
    dead_day = day.with_player_killed("Vil3")
    m_rejected = GameMemory()
    result = _bind_terminal(submit_exile_vote, dead_day, "Vil1", m_rejected)(target="Vil3", note="should not stick")
    assert not result.valid
    assert m_rejected.notes == ()


# --- within-phase parallelism (T31+: async fan-out) -------------------------


def _slow_lm(answers: Sequence[Any], delay_s: float) -> DummyLM:
    """Build a `DummyLM` whose `aforward` awaits before returning.

    `asyncio.sleep` is the only thing in this chain that genuinely suspends
    the coroutine back to the event loop, so the gather call actually
    interleaves the seats — which is what we need to assert real concurrency
    instead of submission-order pseudo-concurrency that DummyLM produces by
    default (its bare `aforward` has no real awaits).
    """
    import asyncio as _asyncio

    lm = DummyLM(list(answers))  # type: ignore[arg-type]

    async def aforward(prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        await _asyncio.sleep(delay_s)
        return lm.forward(prompt=prompt, messages=messages, **kwargs)

    lm.aforward = aforward  # type: ignore[method-assign]
    return lm


def _night_only_scripts() -> dict[str, list[dict[str, Any]]]:
    """One commit per night-acting seat; villagers see no night calls."""
    return {
        "Wolf1": _commit_pair("submit_kill_vote", {"target": "Vil1"}),
        "Wolf2": _commit_pair("submit_kill_vote", {"target": "Vil1"}),
        "Seer1": _commit_pair("seer_inspect", {"target": "Wolf1"}),
        "Doc1": _commit_pair("doctor_protect", {"target": "Vil1"}),
        "Vil1": [],
        "Vil2": [],
        "Vil3": [],
    }


def test_night_actions_runs_seats_concurrently_under_real_io_delays() -> None:
    """Night decisions complete in roughly the time of the slowest seat.

    Within-phase parallelism is the whole point of the async surface: with
    each acting seat blocking ~0.2s on real I/O per ReAct iteration,
    sequential execution would be 4 seats * 2 iters * 0.2s = 1.6s. Parallel
    should be close to 2 iters * 0.2s = 0.4s. Crossing the midpoint (0.9s)
    proves real concurrency, not pseudo-concurrency from submission order.
    """
    import time as _time

    scripts = _night_only_scripts()
    lms: dict[str, DummyLM] = {name: _slow_lm(answers, 0.2) for name, answers in scripts.items()}

    source = ReActDecisionSource(roster=ROSTER, lms=lms)
    t0 = _time.monotonic()
    actions = source.night_actions(GameState.initial(ROSTER))
    elapsed = _time.monotonic() - t0

    assert actions.kill_votes == {"Wolf1": "Vil1", "Wolf2": "Vil1"}
    assert elapsed < 0.9, f"night_actions did not parallelize seats; elapsed={elapsed:.2f}s"


def test_trajectories_appended_in_roster_order_despite_completion_order() -> None:
    """Even when later-roster seats return first, trajectories are in roster order.

    Determinism (invariant #4): same seed + same per-seat scripts must produce
    the same `Trajectory` sequence regardless of which seat's LM happens to
    finish first. The async aggregator must collate by argument order
    (= roster order), not completion order.
    """

    scripts = _night_only_scripts()
    # Reverse-skew the delays: Doc1 fastest, Wolf1 slowest. Without explicit
    # roster-order collation, trajectories would land Doc1→Seer1→Wolf2→Wolf1.
    delays = {"Wolf1": 0.3, "Wolf2": 0.2, "Seer1": 0.1, "Doc1": 0.05}
    lms: dict[str, DummyLM] = {}
    for name, answers in scripts.items():
        lms[name] = _slow_lm(answers, delays.get(name, 0.0))

    source = ReActDecisionSource(roster=ROSTER, lms=lms)
    source.night_actions(GameState.initial(ROSTER))

    callers = [t.caller for t in source.trajectories]
    assert callers == ["Wolf1", "Wolf2", "Seer1", "Doc1"]
    # decision_seq is gap-free and in roster order.
    assert [t.decision_seq for t in source.trajectories] == [0, 1, 2, 3]


def test_pending_drafts_ordered_by_caller_roster_order_under_async_fanout() -> None:
    """`drain_drafts` returns BID drafts in roster order, not completion order.

    The CLI and the event log both depend on draft order being stable;
    interleaving by completion order would make replays non-deterministic
    and confuse spectators ("Doc1's bid came in before Wolf1's? wrong").
    """

    # Reverse-skew: Vil3 first to finish, Wolf1 last.
    delays = {"Wolf1": 0.3, "Wolf2": 0.25, "Seer1": 0.2, "Doc1": 0.15, "Vil1": 0.1, "Vil2": 0.05, "Vil3": 0.02}
    lms: dict[str, DummyLM] = {}
    for name, _role in ROSTER:
        lms[name] = _slow_lm(_commit_pair("submit_bid", {"amount": 1}), delays[name])

    source = ReActDecisionSource(roster=ROSTER, lms=lms)
    state = advance_phase(GameState.initial(ROSTER, bid_budget=BID_BUDGET))  # to Phase.DAY
    source.bids(state)

    drafts = source.drain_drafts()
    bid_drafts = [d for d in drafts if d.type == BID]
    bidders = [d.payload["bidder"] for d in bid_drafts]
    assert bidders == [name for name, _ in ROSTER]


def test_determinism_byte_identical_event_stream_across_per_seat_delay_orderings() -> None:
    """Two games with the same per-seat scripts but swapped per-seat LM
    delays produce byte-identical `EventStream` outputs.

    This is the load-bearing test for async correctness. Determinism
    (invariant #4) requires the recorded transcript to depend only on the
    seed + scripted decisions, never on wall-clock orderings. If async fan-out
    leaks any completion-order information into the log — through draft
    order, trajectory order, or vote aggregation — this test catches it.
    """

    def build_lms(delays: dict[str, float]) -> dict[str, DummyLM]:
        # Rebuild the per-seat scripts from scratch so neither source shares
        # a `DummyLM.answers` iterator with the other; the fixture function
        # returns fresh instances on every call.
        scripts = _per_seat_werewolf_sweep_lms()
        out: dict[str, DummyLM] = {}
        for name, original in scripts.items():
            answers = list(original.answers) if hasattr(original, "answers") else []
            out[name] = _slow_lm(answers, delays.get(name, 0.0))
        return out

    delays_a = {"Wolf1": 0.04, "Wolf2": 0.03, "Seer1": 0.02, "Doc1": 0.01, "Vil2": 0.01, "Vil3": 0.005}
    delays_b = {"Wolf1": 0.005, "Wolf2": 0.01, "Seer1": 0.02, "Doc1": 0.03, "Vil2": 0.04, "Vil3": 0.05}

    source_a = ReActDecisionSource(roster=ROSTER, lms=build_lms(delays_a))
    source_b = ReActDecisionSource(roster=ROSTER, lms=build_lms(delays_b))

    stream_a = run_game(ROSTER, seed=42, decisions=source_a)
    stream_b = run_game(ROSTER, seed=42, decisions=source_b)

    assert_streams_identical(stream_a, stream_b)


# --- two-phase night: wolf chat sub-phase (D2) ------------------------------


def _night_state() -> GameState:
    return GameState.initial(ROSTER)


def test_night_chat_runs_a_loop_per_wolf_and_stages_chat_drafts() -> None:
    """`night_chat` runs one ReAct loop per living wolf, each committing one
    `werewolf_chat` message that drains as a `WEREWOLF_CHAT` draft.

    Without a dedicated chat sub-phase, wolves stage chat during the same
    pass that commits the kill — so a wolf never sees its packmate's message
    before voting. `night_chat` is the first half of the two-phase night that
    makes coordination real.
    """
    lms: dict[str, DummyLM] = {name: DummyLM([]) for name, _ in ROSTER}
    lms["Wolf1"] = DummyLM(_commit_pair("werewolf_chat", {"message": "let's take Vil1"}))
    lms["Wolf2"] = DummyLM(_commit_pair("werewolf_chat", {"message": "agreed, Vil1"}))

    source = ReActDecisionSource(roster=ROSTER, lms=lms)
    source.night_chat(_night_state())

    drafts = source.drain_drafts()
    chat = [d for d in drafts if d.type == WEREWOLF_CHAT]
    speakers = {d.payload["speaker"] for d in chat}
    assert speakers == {"Wolf1", "Wolf2"}
    assert all(set(d.recipients) == {"Wolf1", "Wolf2"} for d in chat)


def test_wolf_observes_packmate_chat_before_voting() -> None:
    """After `night_chat` + `observe`, a wolf's memory holds its packmate's chat.

    This is the coordination fix: in the real run both wolves chatted into the
    void ("no reply recorded… I'll decide myself"). With the two-phase night,
    the engine drains the chat and observes it back into the wolves' memory
    before the kill vote, so wolf B can read wolf A's proposal.
    """
    lms: dict[str, DummyLM] = {name: DummyLM([]) for name, _ in ROSTER}
    lms["Wolf1"] = DummyLM(_commit_pair("werewolf_chat", {"message": "take Vil1 tonight"}))
    lms["Wolf2"] = DummyLM(_commit_pair("werewolf_chat", {"message": "ok"}))

    source = ReActDecisionSource(roster=ROSTER, lms=lms)
    state = _night_state()
    source.night_chat(state)

    # Simulate the engine: drain the chat drafts, log them, observe them back.
    log = EventLog()
    for draft in source.drain_drafts():
        log.append(
            round=state.round,
            phase=state.phase,
            type=draft.type,
            payload=draft.payload,
            recipients=draft.recipients,
        )
    source.observe(state, log.events)

    wolf2_chats = [e for e in source.memories["Wolf2"].events if e.type == WEREWOLF_CHAT]
    assert any(e.payload["speaker"] == "Wolf1" and e.payload["message"] == "take Vil1 tonight" for e in wolf2_chats)


def test_night_chat_skipped_when_only_one_wolf_alive() -> None:
    """A lone surviving wolf has no packmate, so the chat sub-phase is skipped.

    In the real run, after the second wolf died the survivor still ran a
    `werewolf_chat`, narrating to an empty pack ("Pack, ... anyone have
    concerns?") — a wasted decision and an LM call. With only one living wolf,
    `night_chat` must do nothing: no draft, and the wolf's LM is never called.
    """
    lms: dict[str, DummyLM] = {name: DummyLM([]) for name, _ in ROSTER}
    # Non-empty queue: a (pointless) solo chat loop would consume an answer and
    # populate history; we assert both stay untouched.
    lms["Wolf1"] = DummyLM(_commit_pair("werewolf_chat", {"message": "solo now"}))
    lms["Wolf2"] = DummyLM([])

    source = ReActDecisionSource(roster=ROSTER, lms=lms)
    state = _night_state().with_player_killed("Wolf2")
    source.night_chat(state)

    assert source.drain_drafts() == ()
    assert lms["Wolf1"].history == []
