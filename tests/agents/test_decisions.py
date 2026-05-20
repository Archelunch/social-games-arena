"""Tests for `ReActDecisionSource` — the M4 `DecisionSource` adapter (T21).

The adapter wires a per-player ReAct loop (from `react_decide`) into the
`run_game` driver. Per round, it walks alive players in roster order and runs
one decision-point loop per acting role: at night, werewolves vote a kill, the
seer inspects, the doctor protects; in the day, every alive player votes (or
abstains) on the exile. After each phase, the driver calls the adapter's
`observe` hook with the newly-appended events; the adapter routes them through
`observations_for` into each player's `GameMemory` (invariant #2 — agents
never see hidden state).

Tests use `DummyLM` to script each loop's LM responses in the order they will
be consumed. Because each loop runs one `react_decide` (one terminal + one
`finish` = two ReAct iterations) and the adapter walks players in roster
order, the script's structure is predictable.
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

# --------------------------- fixtures -------------------------------------


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
    """One scripted ReAct iteration."""
    return {"next_thought": thought, "next_tool_name": tool, "next_tool_args": args}


def _finish(thought: str = "done") -> dict[str, Any]:
    return {"next_thought": thought, "next_tool_name": "finish", "next_tool_args": {}}


def _commit_pair(tool: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """A minimal commit: one terminal call, then `finish`."""
    return [_step(tool, args), _finish()]


# --------------------------- observe-routing tests ------------------------


def test_observe_routes_public_event_to_every_alive_player() -> None:
    """A public event broadcast lands in every alive player's memory.

    Invariant #2's read side: the adapter must consult `observations_for` to
    route events, and a public event has empty `recipients` so every player
    sees it. Pinning this guards against a refactor that accidentally drops
    public broadcasts from the routing layer.
    """
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM([]))
    state = GameState.initial(ROSTER)

    log = EventLog()
    log.append(round=1, phase=Phase.NIGHT, type=KILL_RESOLVED, payload={"victim": "Vil1"})
    source.observe(state, log.events)

    for name in (n for n, _ in ROSTER):
        assert len(source.memories[name].events) == 1
        assert source.memories[name].events[0].type == KILL_RESOLVED


def test_observe_routes_private_event_to_named_recipients_only() -> None:
    """A private event reaches only its named recipients (invariant #2).

    A `seer_inspect` is private to the seer — every other player's memory
    must stay empty for that event. This is the read-side closure of the
    engine's recipient-routing guarantee.
    """
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
    """Two `observe` calls with disjoint `new_events` accumulate, not re-stack.

    The driver passes a sliced view of just-appended events on each call. The
    adapter must trust that slice — re-pushing prior events would corrupt
    per-player history. Test by issuing two distinct events on two `observe`
    calls and confirming each memory holds exactly two events.
    """
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


# --------------------------- night_actions tests --------------------------


def test_night_actions_aggregates_per_role_terminals() -> None:
    """A scripted night yields one `NightActions` with the right shape.

    Walks alive players in roster order. Wolf1 + Wolf2 each commit a kill
    vote (both targeting Vil1); the seer inspects Wolf1; the doctor protects
    Vil1. The aggregated `NightActions` carries each contribution, keyed
    correctly.
    """
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})  # Wolf1
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})  # Wolf2
        + _commit_pair("seer_inspect", {"target": "Wolf1"})  # Seer1
        + _commit_pair("doctor_protect", {"target": "Vil1"})  # Doc1
    )
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))
    state = GameState.initial(ROSTER)

    actions = source.night_actions(state)

    assert actions.kill_votes == {"Wolf1": "Vil1", "Wolf2": "Vil1"}
    assert actions.seer_inspect == "Wolf1"
    assert actions.doctor_protect == "Vil1"


def test_night_actions_skips_dead_seer_returns_none() -> None:
    """When the seer is dead, no loop is run for them; `seer_inspect` is `None`.

    `NightActions.seer_inspect` is `str | None` — `None` means no inspection
    happened. The adapter must respect the alive filter before running a
    loop; running a dead seer's loop would (a) waste an LM call and (b) feed
    a tool result the resolver would reject anyway.
    """
    # Pre-kill the seer.
    state = GameState.initial(ROSTER).with_player_killed("Seer1")

    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})  # Wolf1
        + _commit_pair("submit_kill_vote", {"target": "Vil1"})  # Wolf2
        # no seer loop
        + _commit_pair("doctor_protect", {"target": "Vil1"})  # Doc1
    )
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))

    actions = source.night_actions(state)

    assert actions.seer_inspect is None
    assert actions.kill_votes == {"Wolf1": "Vil1", "Wolf2": "Vil1"}
    assert actions.doctor_protect == "Vil1"


def test_night_actions_skips_dead_doctor_returns_none() -> None:
    """When the doctor is dead, `doctor_protect` is `None` and no loop runs.

    Symmetric with the dead-seer case — the adapter must filter every acting
    role independently, not skip the whole night when one role is missing.
    """
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


# --------------------------- day_actions tests ----------------------------


def test_day_actions_collects_a_vote_from_every_alive_player() -> None:
    """A 7-player day yields seven exile-vote entries.

    Each alive player gets one ReAct loop with `submit_exile_vote` as its
    terminator. The resulting `DayActions.exile_votes` is keyed by every
    alive name.
    """
    state = advance_phase(GameState.initial(ROSTER))

    # Roster order: Wolf1, Wolf2, Seer1, Doc1, Vil1, Vil2, Vil3 — each row is
    # one player's exile-vote target. Order-sensitive: a regression here would
    # be caught by `test_roster_order_determines_lm_call_order` as well.
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
    """A player who commits `abstain` shows up as `abstain` in `DayActions`.

    `submit_exile_vote` accepts the ABSTAIN literal; the loop's `Commit.value`
    must be passed through without coercion. Day resolution treats abstains
    correctly only when it sees them spelled exactly.
    """
    state = advance_phase(GameState.initial(ROSTER))
    answers = (
        _commit_pair("submit_exile_vote", {"target": ABSTAIN})  # Wolf1
        + _commit_pair("submit_exile_vote", {"target": "Wolf2"})  # Wolf2
        + _commit_pair("submit_exile_vote", {"target": "Wolf1"})  # Seer1
        + _commit_pair("submit_exile_vote", {"target": "Wolf1"})  # Doc1
        + _commit_pair("submit_exile_vote", {"target": "Wolf1"})  # Vil1
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})  # Vil2
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})  # Vil3
    )
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))

    actions = source.day_actions(state)

    assert actions.exile_votes["Wolf1"] == ABSTAIN
    assert actions.exile_votes["Vil2"] == ABSTAIN
    assert actions.exile_votes["Vil3"] == ABSTAIN


# --------------------------- ordering / determinism -----------------------


def test_roster_order_determines_lm_call_order() -> None:
    """Adapter walks alive players in roster order, not role-grouped order.

    Pin this by swapping two werewolves in the roster: the first scripted
    answer must end up keyed against the first werewolf (by roster order),
    not by name. Determinism requires a fixed iteration order — invariant #4.
    """
    swapped: tuple[tuple[str, str], ...] = (
        ("Wolf2", Role.WEREWOLF.value),  # first now
        ("Wolf1", Role.WEREWOLF.value),
        ("Seer1", Role.SEER.value),
        ("Doc1", Role.DOCTOR.value),
        ("Vil1", Role.VILLAGER.value),
        ("Vil2", Role.VILLAGER.value),
        ("Vil3", Role.VILLAGER.value),
    )
    answers = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})  # First: Wolf2
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})  # Then: Wolf1
        + _commit_pair("seer_inspect", {"target": "Wolf1"})
        + _commit_pair("doctor_protect", {"target": "Vil1"})
    )
    source = ReActDecisionSource(roster=swapped, lm=DummyLM(answers))
    state = GameState.initial(swapped)

    actions = source.night_actions(state)

    assert actions.kill_votes == {"Wolf2": "Vil1", "Wolf1": "Vil2"}


# --------------------------- end-to-end with run_game ---------------------


def _werewolf_sweep_script() -> list[dict[str, Any]]:
    """A scripted full-game LM transcript ending in a werewolf victory.

    Plan: at each night the wolves kill one villager; doctor protects nobody
    relevant; seer inspects something; days all abstain (no exile). After
    three nights three villagers (Vil1, Vil2, Vil3) die — the village goes
    from 5 villagers + 2 wolves to 2 villagers + 2 wolves at parity, which
    is a werewolf win on the post-night terminal check.
    """
    answers: list[dict[str, Any]] = []

    # ------ Night 1: kill Vil1, doctor protects Doc1, seer inspects Wolf1 ------
    answers += _commit_pair("submit_kill_vote", {"target": "Vil1"})  # Wolf1
    answers += _commit_pair("submit_kill_vote", {"target": "Vil1"})  # Wolf2
    answers += _commit_pair("seer_inspect", {"target": "Wolf1"})  # Seer1
    answers += _commit_pair("doctor_protect", {"target": "Seer1"})  # Doc1 (no self-protect)
    # ------ Day 1: all alive abstain ------
    for _ in range(6):  # 7 - 1 killed = 6 alive
        answers += _commit_pair("submit_exile_vote", {"target": ABSTAIN})

    # ------ Night 2: kill Vil2, doctor protects a villager, seer inspects Wolf2 ------
    answers += _commit_pair("submit_kill_vote", {"target": "Vil2"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil2"})
    answers += _commit_pair("seer_inspect", {"target": "Wolf2"})
    answers += _commit_pair("doctor_protect", {"target": "Vil3"})  # no self-protect
    # ------ Day 2: all alive abstain ------
    for _ in range(5):
        answers += _commit_pair("submit_exile_vote", {"target": ABSTAIN})

    # ------ Night 3: kill Vil3, seer inspects, doctor protects Seer1 (no self-protect) ------
    answers += _commit_pair("submit_kill_vote", {"target": "Vil3"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil3"})
    answers += _commit_pair("seer_inspect", {"target": "Doc1"})
    answers += _commit_pair("doctor_protect", {"target": "Seer1"})

    return answers


def test_run_game_with_react_source_reaches_terminal_state() -> None:
    """`run_game` with a `ReActDecisionSource` and a scripted LM ends in game_over.

    The full integration test: the M4 adapter slots into `run_game` without
    modifying the driver beyond the `observe` Protocol hook. The final event
    is `GAME_OVER` and names the winning faction.
    """
    answers = _werewolf_sweep_script()
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))

    stream = run_game(ROSTER, seed=42, decisions=source)
    events = stream.log.events

    assert events[-1].type == GAME_OVER
    assert events[-1].payload["winner"] == "werewolves"


def test_no_recorded_event_violates_invariant_2_for_any_player() -> None:
    """Read-side hidden-state guard (invariant #2): every event in a player's memory
    is either public or names that player as a recipient.

    A full game ends; each player's `GameMemory` is the closure of what they
    saw. We assert the membership condition directly, mirroring
    `observations_for`'s contract. If routing leaks, this test catches it.
    """
    answers = _werewolf_sweep_script()
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM(answers))
    run_game(ROSTER, seed=42, decisions=source)

    for name in (n for n, _ in ROSTER):
        for event in source.memories[name].events:
            assert event.recipients == () or name in event.recipients, (
                f"player {name!r} has private event {event.type!r} not addressed to them"
            )


def test_memories_property_is_a_read_only_mapping() -> None:
    """`memories` exposes a `Mapping` view — direct mutation must be impossible.

    Callers (tests, telemetry, T22 wiring) read per-player memory state.
    Letting them swap a `GameMemory` in place would silently corrupt
    invariant #2's per-agent isolation.
    """
    source = ReActDecisionSource(roster=ROSTER, lm=DummyLM([]))
    assert isinstance(source.memories, Mapping)
    with pytest.raises(TypeError):
        source.memories["Wolf1"] = GameMemory()  # type: ignore[index]
