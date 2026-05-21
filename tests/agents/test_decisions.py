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
    BID,
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


_K_SPEAKERS = 3  # `K_DISCUSSION_SLOTS` — see `games.werewolf.config`.


def _day_dialogue_pairs(n_alive: int) -> list[dict[str, Any]]:
    """Bid + speech answers for one day, in the order `run_game` will call them.

    The loop walks the roster in order for `bids()` (one `submit_bid` commit per
    alive player), then walks the chosen `speakers` tuple (length
    `min(K, n_alive)`) for `speeches()`. The uniform-LM tests share one answer
    queue across seats, so the queue must hold every commit + finish pair in
    the order they will be consumed.
    """
    pairs: list[dict[str, Any]] = []
    for _ in range(n_alive):
        pairs += _commit_pair("submit_bid", {"amount": 0})
    for _ in range(min(_K_SPEAKERS, n_alive)):
        pairs += _commit_pair("speak", {"message": "no comment"})
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

    # Round 1: night (4 actors) -> day (6 alive: 6 bids, 3 speeches, 6 exile votes).
    answers += _commit_pair("submit_kill_vote", {"target": "Vil1"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil1"})
    answers += _commit_pair("seer_inspect", {"target": "Wolf1"})
    answers += _commit_pair("doctor_protect", {"target": "Seer1"})
    answers += _day_dialogue_pairs(6)
    for _ in range(6):
        answers += _commit_pair("submit_exile_vote", {"target": ABSTAIN})

    # Round 2: night (4 actors) -> day (5 alive: 5 bids, 3 speeches, 5 exile votes).
    answers += _commit_pair("submit_kill_vote", {"target": "Vil2"})
    answers += _commit_pair("submit_kill_vote", {"target": "Vil2"})
    answers += _commit_pair("seer_inspect", {"target": "Wolf2"})
    answers += _commit_pair("doctor_protect", {"target": "Vil3"})
    answers += _day_dialogue_pairs(5)
    for _ in range(5):
        answers += _commit_pair("submit_exile_vote", {"target": ABSTAIN})

    # Round 3 night ends the game on werewolf parity.
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


def _bid_then_maybe_speech(amount: int, *, speaks: bool) -> list[dict[str, Any]]:
    """One day's per-seat answers: a `submit_bid` commit and, when chosen, a `speak` commit."""
    out = _commit_pair("submit_bid", {"amount": amount})
    if speaks:
        out += _commit_pair("speak", {"message": "no comment"})
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
    wolf1 = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _bid_then_maybe_speech(10, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _bid_then_maybe_speech(10, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_kill_vote", {"target": "Vil3"})
    )
    wolf2 = (
        _commit_pair("submit_kill_vote", {"target": "Vil1"})
        + _bid_then_maybe_speech(9, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _bid_then_maybe_speech(9, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
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
                _finish(),
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
                _finish(),
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
    state = advance_phase(GameState.initial(ROSTER))
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
    state = advance_phase(GameState.initial(ROSTER)).with_player_killed("Vil1")
    seat_lms: dict[str, DummyLM] = {name: DummyLM(_commit_pair("submit_bid", {"amount": 1})) for name, _ in ROSTER}
    source = ReActDecisionSource(roster=ROSTER, lms=seat_lms)

    bids = source.bids(state)

    assert "Vil1" not in bids
    assert seat_lms["Vil1"].history == []


def test_speeches_emits_public_drafts_in_speaker_order() -> None:
    """`speeches(state, speakers)` runs one ReAct loop per speaker; each emits a public `SPEECH`."""
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

    speeches = source.speeches(state, ("Wolf1", "Seer1", "Vil2"))
    drafts = source.drain_drafts()
    speech_drafts = [d for d in drafts if d.type == SPEECH]

    assert speeches == (
        ("Wolf1", "I am the seer"),
        ("Seer1", "No, I am"),
        ("Vil2", "I trust Seer1"),
    )
    assert len(speech_drafts) == 3
    assert all(d.recipients == () for d in speech_drafts)
    assert [d.payload["speaker"] for d in speech_drafts] == ["Wolf1", "Seer1", "Vil2"]
    assert [d.payload["message"] for d in speech_drafts] == [
        "I am the seer",
        "No, I am",
        "I trust Seer1",
    ]


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
                _finish(),
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
                _finish(),
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
    night1_with_rejection = [
        _step("submit_kill_vote", {"target": "Wolf1"}, "bad"),  # self-target — rejected
        _step("submit_kill_vote", {"target": "Vil1"}, "retry"),
        _finish(),
    ]
    # The rest of Wolf1's day-1 / night-2 / day-2 / night-3 queue from the sweep.
    rest_of_wolf1 = (
        _bid_then_maybe_speech(10, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
        + _commit_pair("submit_kill_vote", {"target": "Vil2"})
        + _bid_then_maybe_speech(10, speaks=True)
        + _commit_pair("submit_exile_vote", {"target": ABSTAIN})
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
                _finish(),
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
                _finish(),
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

    # The first decision must be a night werewolf kill vote (roster order: Wolf1).
    first = trajectories[0]
    assert first.caller == "Wolf1"
    assert first.phase == Phase.NIGHT.value
    assert first.terminal_tool == "submit_kill_vote"
    assert first.round == 1


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
    speech_source.speeches(state, ("Wolf1", "Seer1", "Vil2"))

    speech_trajectories = speech_source.trajectories
    assert len(speech_trajectories) == 3
    assert tuple(t.caller for t in speech_trajectories) == ("Wolf1", "Seer1", "Vil2")
    assert all(t.terminal_tool == "speak" for t in speech_trajectories)
