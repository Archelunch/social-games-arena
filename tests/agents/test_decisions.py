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

from collections.abc import Mapping
from typing import Any

import pytest
from dspy.utils.dummies import DummyLM

from social_deduction_bench.agents import GameMemory
from social_deduction_bench.agents.decisions import ReActDecisionSource
from social_deduction_bench.engine import EventLog, GameState, Phase, advance_phase
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


def test_observe_routes_public_event_to_every_alive_player() -> None:
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM([]))
    state = GameState.initial(ROSTER)

    log = EventLog()
    log.append(round=1, phase=Phase.NIGHT, type=KILL_RESOLVED, payload={"victim": "Vil1"})
    source.observe(state, log.events)

    for name in (n for n, _ in ROSTER):
        assert len(source.memories[name].events) == 1
        assert source.memories[name].events[0].type == KILL_RESOLVED


def test_observe_routes_private_event_to_named_recipients_only() -> None:
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM([]))
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
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM([]))
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
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))
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
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))

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
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))

    actions = source.night_actions(state)

    assert actions.doctor_protect is None
    assert actions.seer_inspect == "Wolf1"


def test_day_actions_collects_a_vote_from_every_alive_player() -> None:
    state = advance_phase(GameState.initial(ROSTER))

    targets = ["Wolf2", "Vil1", "Wolf1", "Wolf1", "Wolf1", "Wolf2", "Vil2"]
    answers: list[dict[str, Any]] = []
    for target in targets:
        answers.extend(_commit_pair("submit_exile_vote", {"target": target}))
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))

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
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))

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
    source = ReActDecisionSource(roster=swapped, lm=DummyLM(answers))
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
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))

    stream = run_game(ROSTER, seed=42, decisions=source)
    events = stream.log.events

    assert events[-1].type == GAME_OVER
    assert events[-1].payload["winner"] == "werewolves"


def test_no_recorded_event_violates_hidden_state_for_any_player() -> None:
    """Every event in each player's memory is either public or names them as a recipient."""
    answers = _werewolf_sweep_script()
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))
    run_game(ROSTER, seed=42, decisions=source)

    for name in (n for n, _ in ROSTER):
        for event in source.memories[name].events:
            assert event.recipients == () or name in event.recipients, (
                f"player {name!r} has private event {event.type!r} not addressed to them"
            )


def test_memories_property_is_a_read_only_mapping() -> None:
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM([]))
    assert isinstance(source.memories, Mapping)
    with pytest.raises(TypeError):
        source.memories["Wolf1"] = GameMemory()  # type: ignore[index]
