"""Tests for `ReActDecisionSource`.

The adapter wires a per-player ReAct loop (from `react_decide`) into the
`run_game` driver. Per round, it walks alive players in roster order and runs
one decision-point loop per acting role; after each phase, the driver calls
`observe` with newly-appended events, which the adapter routes through
`observations_for` into each living player's `GameMemory`.

Tests use `DummyLM` to script each loop's LM responses in the order they will
be consumed. Each loop runs one terminal commit + one `finish` = two ReAct
iterations.
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
from social_deduction_bench.games.werewolf.events import (
    ABSTAIN,
    EXILE_RESOLVED,
    GAME_OVER,
    KILL_RESOLVED,
    SEER_INSPECT,
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


def _finish(thought: str = "done") -> dict[str, Any]:
    return {"next_thought": thought, "next_tool_name": "finish", "next_tool_args": {}}


def _commit_pair(tool: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [_step(tool, args), _finish()]


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


def _werewolf_sweep_script() -> list[dict[str, Any]]:
    """A scripted full-game LM transcript ending in a werewolf victory.

    Three villagers die over three nights; days all abstain (no exile). The
    village goes from 5 villagers + 2 wolves to 2 villagers + 2 wolves at
    parity — a werewolf win on the post-night terminal check.
    """
    answers: list[dict[str, Any]] = []

    answers += _commit_pair("submit_kill_vote", {"target": "Vil1"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil1"})
    answers += _commit_pair("seer_inspect", {"target": "Wolf1"})
    answers += _commit_pair("doctor_protect", {"target": "Seer1"})
    for _ in range(6):
        answers += _commit_pair("submit_exile_vote", {"target": ABSTAIN})

    answers += _commit_pair("submit_kill_vote", {"target": "Vil2"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil2"})
    answers += _commit_pair("seer_inspect", {"target": "Wolf2"})
    answers += _commit_pair("doctor_protect", {"target": "Vil3"})
    for _ in range(5):
        answers += _commit_pair("submit_exile_vote", {"target": ABSTAIN})

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
        assert len(seat_lms[acting].history) == 2, (
            f"acting seat {acting!r} should consume exactly 2 LM calls (commit + finish), "
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
        assert len(seat_lms[alive].history) == 2, (
            f"alive seat {alive!r} should consume 2 LM calls (commit + finish), got {len(seat_lms[alive].history)}"
        )


def _per_seat_werewolf_sweep_lms() -> dict[str, DummyLM]:
    """Per-seat scripted answers for the same werewolf-win sweep as the legacy fixture.

    Three nights of wolves killing Vil1/Vil2/Vil3; days in between are unanimous
    abstains. The seer inspects Wolf1/Wolf2/Doc1; the doctor protects Seer1/Vil3/Seer1.
    Villager seats only contribute their day-time abstain votes until they die.
    """
    wolf1 = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_kill_vote", {"target": "Vil3"})
    )
    wolf2 = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_kill_vote", {"target": "Vil3"})
    )
    seer1 = (
        _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("seer_inspect", {"target": "Wolf2"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("seer_inspect", {"target": "Doc1"})
    )
    doc1 = (
        _commit_pair("doctor_protect", {"target": "Seer1"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("doctor_protect", {"target": "Vil3"})
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("doctor_protect", {"target": "Seer1"})
    )
    vil2 = _commit_pair("submit_exile_vote", {"target": ABSTAIN})
    vil3 = _commit_pair("submit_exile_vote", {"target": ABSTAIN}) + _commit_pair(
        "submit_exile_vote", {"target": ABSTAIN}
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
