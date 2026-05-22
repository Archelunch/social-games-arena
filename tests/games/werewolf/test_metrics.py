"""Tests for Werewolf metric extraction (T24).

`extract_game_metrics` turns one game's `EventStream` + `TrajectoryStream` into a
structured `GameMetrics`: winner, game length, per-seat win/survival, tokens, and
illegal-move counts. The engine event log is the source of truth for outcomes
(invariant #1) — winner and deaths come from events, never the manifest; the
trajectory sidecar only supplies agent-internal telemetry (tokens, tool calls).

Most cases build streams synthetically for precise control. One integration case
drives a real scripted wolves-win game so the extractor is proven against the
real trajectory shape (`"ok:"` / `"error:"` observation prefixes, real
`lm_calls`).
"""

from __future__ import annotations

from typing import Any

import pytest
from dspy.utils.dummies import DummyLM

from social_deduction_bench.agents.decisions import ReActDecisionSource
from social_deduction_bench.agents.trajectory import (
    LMCallRecord,
    ReActStep,
    Trajectory,
    TrajectoryStream,
)
from social_deduction_bench.engine import EventLog, Phase
from social_deduction_bench.engine.events import EventStream, StreamHeader
from social_deduction_bench.games.werewolf.events import (
    ABSTAIN,
    EXILE_RESOLVED,
    GAME_OVER,
    KILL_RESOLVED,
    TOOL_REJECTED,
)
from social_deduction_bench.games.werewolf.loop import run_game
from social_deduction_bench.games.werewolf.metrics import (
    aggregate_metrics,
    deceiver_detector_split,
    extract_game_metrics,
    extract_run_dir,
)
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.rating.manifest import RunManifest

# A small 4-seat roster for the synthetic cases: one of each role.
_PLAYERS: tuple[tuple[str, str], ...] = (
    ("Alice", Role.SEER.value),
    ("Bob", Role.DOCTOR.value),
    ("Carol", Role.VILLAGER.value),
    ("Dave", Role.WEREWOLF.value),
)


# --- synthetic stream builders -------------------------------------------


def _events(
    rows: list[tuple[int, Phase, str, dict[str, object], tuple[str, ...]]],
    *,
    players: tuple[tuple[str, str], ...] = _PLAYERS,
) -> EventStream:
    log = EventLog()
    for round_, phase, type_, payload, recipients in rows:
        log.append(round=round_, phase=phase, type=type_, payload=payload, recipients=recipients)
    return EventStream(header=StreamHeader(seed=7, game_id="g1", players=players), log=log)


def _trajectories(rows: list[Trajectory], *, players: tuple[tuple[str, str], ...] = _PLAYERS) -> TrajectoryStream:
    return TrajectoryStream(header=StreamHeader(seed=7, game_id="g1", players=players), trajectories=tuple(rows))


def _step(tool: str, observation: str) -> ReActStep:
    return ReActStep(iter=0, thought="t", tool=tool, args={}, observation=observation)


def _traj(
    *,
    seq: int,
    caller: str,
    role: str,
    terminal_tool: str = "submit_exile_vote",
    round_: int = 1,
    phase: str = "day",
    steps: tuple[ReActStep, ...] = (),
    lm_calls: tuple[LMCallRecord, ...] = (),
) -> Trajectory:
    return Trajectory(
        decision_seq=seq,
        round=round_,
        phase=phase,
        caller=caller,
        role=role,
        terminal_tool=terminal_tool,
        committed_value="x",
        react_trajectory=steps,
        lm_calls=lm_calls,
    )


def _lm(model: str, prompt: int, completion: int, cost: float | None = None) -> LMCallRecord:
    return LMCallRecord(model=model, prompt_tokens=prompt, completion_tokens=completion, latency_ms=1.0, cost_usd=cost)


def _villager_win_game() -> EventStream:
    """Carol killed at night, Dave (the wolf) exiled by day → villagers win, 1 round."""
    return _events(
        [
            (1, Phase.NIGHT, KILL_RESOLVED, {"victim": "Carol"}, ()),
            (1, Phase.DAY, EXILE_RESOLVED, {"ballots": {"Alice": "Dave"}, "exiled": "Dave"}, ()),
            (1, Phase.DAY, GAME_OVER, {"winner": "villagers"}, ()),
        ]
    )


# --- outcome extraction (engine event log = source of truth) -------------


def test_winner_rounds_and_player_count_come_from_events() -> None:
    metrics = extract_game_metrics(_villager_win_game(), _trajectories([]))
    assert metrics.winner == "villagers"
    assert metrics.rounds == 1
    assert metrics.n_players == 4


def test_game_with_no_game_over_event_fails_loud() -> None:
    # An incomplete transcript has no winner; computing metrics from it would
    # silently invent one. Fail loud (CLAUDE.md rule 11).
    stream = _events([(1, Phase.NIGHT, KILL_RESOLVED, {"victim": "Carol"}, ())])
    with pytest.raises(ValueError, match="game_over"):
        extract_game_metrics(stream, _trajectories([]))


def test_seat_win_follows_faction_vs_winner() -> None:
    metrics = extract_game_metrics(_villager_win_game(), _trajectories([]))
    # Villagers win → seer/doctor/villager won, werewolf lost.
    assert {s.name: s.faction for s in metrics.seats} == {
        "Alice": "villagers",
        "Bob": "villagers",
        "Carol": "villagers",
        "Dave": "werewolves",
    }
    assert {s.name: s.won for s in metrics.seats} == {
        "Alice": True,
        "Bob": True,
        "Carol": True,
        "Dave": False,
    }


def test_survived_is_false_for_night_victim_and_exiled_seat() -> None:
    metrics = extract_game_metrics(_villager_win_game(), _trajectories([]))
    by_name = {s.name: s for s in metrics.seats}
    assert by_name["Carol"].survived is False  # killed at night
    assert by_name["Dave"].survived is False  # exiled by day
    assert by_name["Alice"].survived is True
    assert by_name["Bob"].survived is True


def test_null_victim_and_null_exile_leave_everyone_alive() -> None:
    # Doctor save → kill_resolved victim None; tie → exiled None. No one dies.
    stream = _events(
        [
            (1, Phase.NIGHT, KILL_RESOLVED, {"victim": None}, ()),
            (1, Phase.DAY, EXILE_RESOLVED, {"ballots": {"Alice": ABSTAIN}, "exiled": None}, ()),
            (1, Phase.DAY, GAME_OVER, {"winner": "werewolves"}, ()),
        ]
    )
    metrics = extract_game_metrics(stream, _trajectories([]))
    assert all(s.survived for s in metrics.seats)


# --- token telemetry from the sidecar ------------------------------------


def test_tokens_sum_per_seat_and_do_not_leak_across_seats() -> None:
    trajs = _trajectories(
        [
            _traj(seq=0, caller="Alice", role=Role.SEER.value, lm_calls=(_lm("m", 100, 10), _lm("m", 50, 5))),
            _traj(seq=1, caller="Dave", role=Role.WEREWOLF.value, lm_calls=(_lm("m", 7, 3),)),
        ]
    )
    metrics = extract_game_metrics(_villager_win_game(), trajs)
    by_name = {s.name: s for s in metrics.seats}
    assert (by_name["Alice"].prompt_tokens, by_name["Alice"].completion_tokens) == (150, 15)
    assert (by_name["Dave"].prompt_tokens, by_name["Dave"].completion_tokens) == (7, 3)
    assert (by_name["Carol"].prompt_tokens, by_name["Carol"].completion_tokens) == (0, 0)
    assert metrics.total_prompt_tokens == 157
    assert metrics.total_completion_tokens == 18
    assert metrics.total_lm_calls == 3


def test_cost_is_none_when_all_calls_are_uncosted_else_summed() -> None:
    none_cost = extract_game_metrics(
        _villager_win_game(),
        _trajectories([_traj(seq=0, caller="Alice", role=Role.SEER.value, lm_calls=(_lm("m", 1, 1),))]),
    )
    assert none_cost.total_cost_usd is None

    with_cost = extract_game_metrics(
        _villager_win_game(),
        _trajectories(
            [
                _traj(seq=0, caller="Alice", role=Role.SEER.value, lm_calls=(_lm("m", 1, 1, cost=0.02),)),
                _traj(seq=1, caller="Dave", role=Role.WEREWOLF.value, lm_calls=(_lm("m", 1, 1, cost=0.01),)),
            ]
        ),
    )
    assert with_cost.total_cost_usd == pytest.approx(0.03)


# --- illegal-move counting (engine/sidecar agreement) --------------------


def test_illegal_moves_equal_tool_rejected_events_and_exclude_cognitive_errors() -> None:
    """Illegal game-action attempts (sidecar) equal the engine's TOOL_REJECTED count.

    The engine emits a private `TOOL_REJECTED` event for every rejected
    game-action call (T29). Illegal moves are taken from the engine event log (the
    referee is the authority — invariant #1), NOT from trajectory `"error:"`
    observations. Two other `"error:"` shapes must NOT be counted as illegal:
    a *cognitive*-tool error (e.g. a bad `set_belief`) and a tool *execution*
    fault (`"error: execution error in ..."` — the tool fn raised), neither of
    which produces a `TOOL_REJECTED` event. Only the engine-recorded rejection is.
    """
    stream = _events(
        [
            (1, Phase.NIGHT, KILL_RESOLVED, {"victim": "Carol"}, ()),
            (1, Phase.NIGHT, TOOL_REJECTED, {"tool": "submit_kill_vote", "args": {}, "reason": "dead"}, ("Dave",)),
            (1, Phase.DAY, EXILE_RESOLVED, {"ballots": {}, "exiled": None}, ()),
            (1, Phase.DAY, GAME_OVER, {"winner": "werewolves"}, ()),
        ]
    )
    trajs = _trajectories(
        [
            _traj(
                seq=0,
                caller="Dave",
                role=Role.WEREWOLF.value,
                terminal_tool="submit_kill_vote",
                round_=1,
                phase="night",
                steps=(
                    _step("set_belief", "error: confidence must be one of ..."),  # cognitive error — not a move
                    _step("submit_kill_vote", "error: target 'Carol' is dead and cannot be acted on"),  # rejected
                    _step("submit_kill_vote", "error: execution error in submit_kill_vote: ValueError()"),  # fault
                    _step("submit_kill_vote", "ok: submit_kill_vote committed with {'target': 'Bob'}"),
                ),
            )
        ]
    )
    metrics = extract_game_metrics(stream, trajs)
    tool_rejected_events = sum(1 for e in stream.log.events if e.type == TOOL_REJECTED)
    # Three game-action `"error:"` observations exist across the trajectory (one
    # rejection, one execution fault) plus one cognitive error, but only the single
    # engine-recorded TOOL_REJECTED event is an illegal move. A naive "count every
    # error: step" would report 2 here — this is the regression guard.
    assert metrics.total_illegal_moves == tool_rejected_events == 1
    # Three game-action attempts (rejected + faulted + ok); the cognitive error is excluded.
    assert metrics.total_tool_calls == 3
    assert metrics.illegal_move_rate == pytest.approx(1 / 3)


def test_tool_usage_counts_game_actions_only_and_is_sorted() -> None:
    # Feed game-action tools in NON-alphabetical insertion order so the `sorted(...)`
    # in tool_usage is load-bearing: a plain Counter would preserve insertion order
    # (submit_exile_vote, speak, accuse) and fail the ordered-tuple assertion below.
    trajs = _trajectories(
        [
            _traj(
                seq=0,
                caller="Dave",
                role=Role.WEREWOLF.value,
                steps=(
                    _step("recall", "ok: <memory dump>"),  # cognitive — excluded
                    _step("set_belief", "ok: belief set"),  # cognitive — excluded
                    _step("submit_exile_vote", "ok: submit_exile_vote committed with ..."),
                    _step("speak", "ok: speak committed with ..."),
                    _step("accuse", "ok: accuse committed with ..."),
                ),
            )
        ]
    )
    metrics = extract_game_metrics(_villager_win_game(), trajs)
    # Exact tuple in alphabetical order — pins both the game-action filter and the sort.
    assert metrics.tool_usage == (("accuse", 1), ("speak", 1), ("submit_exile_vote", 1))


def test_illegal_move_rate_is_zero_when_no_game_actions() -> None:
    # Only cognitive tool calls → denominator 0 → rate 0.0, not a ZeroDivisionError.
    trajs = _trajectories([_traj(seq=0, caller="Alice", role=Role.SEER.value, steps=(_step("recall", "ok: ..."),))])
    metrics = extract_game_metrics(_villager_win_game(), trajs)
    assert metrics.total_tool_calls == 0
    assert metrics.illegal_move_rate == 0.0


# --- model identity: manifest vs. inference ------------------------------


def test_model_inferred_from_trajectory_lm_calls_when_no_manifest() -> None:
    trajs = _trajectories(
        [_traj(seq=0, caller="Alice", role=Role.SEER.value, lm_calls=(_lm("openrouter/qwen/qwen3.5-9b", 1, 1),))]
    )
    metrics = extract_game_metrics(_villager_win_game(), trajs, manifest=None)
    by_name = {s.name: s for s in metrics.seats}
    assert by_name["Alice"].model == "openrouter/qwen/qwen3.5-9b"
    # A seat with no LM call and no manifest cannot be resolved.
    assert by_name["Carol"].model == "unknown"


def test_manifest_models_override_inference_and_cover_silent_seats() -> None:
    manifest = RunManifest(
        game_id="g1",
        seed=7,
        players=_PLAYERS,
        models=(("Alice", "model-A"), ("Bob", "model-B"), ("Carol", "model-C"), ("Dave", "model-D")),
        model_arg="mixed",
        temperature=0.7,
        max_tokens=8000,
        max_iters=6,
        reasoning=False,
        git_sha=None,
        created_at="2026-05-22T00:00:00+00:00",
        winner="villagers",
        rounds=1,
    )
    # Alice's trajectory says one model, but the manifest is authoritative.
    trajs = _trajectories([_traj(seq=0, caller="Alice", role=Role.SEER.value, lm_calls=(_lm("ignored", 1, 1),))])
    metrics = extract_game_metrics(_villager_win_game(), trajs, manifest=manifest)
    by_name = {s.name: s for s in metrics.seats}
    assert by_name["Alice"].model == "model-A"
    assert by_name["Carol"].model == "model-C"  # silent seat resolved via manifest


def test_engine_outcomes_win_over_a_conflicting_manifest() -> None:
    """Invariant #1: winner/rounds come from the event log, never the manifest.

    A manifest is derived metadata and could be stale or wrong. The extractor must
    read outcomes only from the engine event stream. This manifest deliberately
    LIES (werewolves win in 99 rounds); the events say villagers win in 1 round.
    """
    lying_manifest = RunManifest(
        game_id="g1",
        seed=7,
        players=_PLAYERS,
        models=(("Alice", "m"), ("Bob", "m"), ("Carol", "m"), ("Dave", "m")),
        model_arg="m",
        temperature=0.7,
        max_tokens=8000,
        max_iters=6,
        reasoning=False,
        git_sha=None,
        created_at="2026-05-22T00:00:00+00:00",
        winner="werewolves",  # contradicts the events (villagers win)
        rounds=99,  # contradicts the events (1 round)
    )
    metrics = extract_game_metrics(_villager_win_game(), _trajectories([]), manifest=lying_manifest)
    assert metrics.winner == "villagers"
    assert metrics.rounds == 1
    # And the werewolf seat lost, per the engine outcome (not the manifest's claim).
    assert {s.name: s.won for s in metrics.seats}["Dave"] is False


# --- determinism & aggregation -------------------------------------------


def test_same_inputs_yield_equal_game_metrics() -> None:
    trajs = [_traj(seq=0, caller="Alice", role=Role.SEER.value, lm_calls=(_lm("m", 1, 1),))]
    a = extract_game_metrics(_villager_win_game(), _trajectories(list(trajs)))
    b = extract_game_metrics(_villager_win_game(), _trajectories(list(trajs)))
    assert a == b


def test_extract_run_dir_reads_the_triplet_and_matches_in_memory(tmp_path) -> None:
    from social_deduction_bench.agents.trajectory import write_jsonl as write_trajectories
    from social_deduction_bench.engine import write_jsonl as write_events

    stream = _villager_win_game()
    trajs = _trajectories([_traj(seq=0, caller="Alice", role=Role.SEER.value, lm_calls=(_lm("m", 5, 2),))])
    write_events(stream, tmp_path / "events.jsonl")
    write_trajectories(trajs, tmp_path / "trajectories.jsonl")
    # No manifest.json on disk → backfill via trajectory inference (the existing-runs path).
    from_disk = extract_run_dir(tmp_path)
    in_memory = extract_game_metrics(stream, trajs)
    assert from_disk == in_memory


def test_extract_run_dir_reads_manifest_when_present(tmp_path) -> None:
    from social_deduction_bench.agents.trajectory import write_jsonl as write_trajectories
    from social_deduction_bench.engine import write_jsonl as write_events
    from social_deduction_bench.rating.manifest import write_json as write_manifest

    stream = _villager_win_game()
    # Alice's trajectory says one model; the on-disk manifest must override it.
    trajs = _trajectories([_traj(seq=0, caller="Alice", role=Role.SEER.value, lm_calls=(_lm("inferred", 5, 2),))])
    write_events(stream, tmp_path / "events.jsonl")
    write_trajectories(trajs, tmp_path / "trajectories.jsonl")
    manifest = RunManifest(
        game_id="g1",
        seed=7,
        players=_PLAYERS,
        models=(("Alice", "from-manifest"), ("Bob", "from-manifest"), ("Carol", "from-manifest"), ("Dave", "x")),
        model_arg="from-manifest",
        temperature=0.7,
        max_tokens=8000,
        max_iters=6,
        reasoning=False,
        git_sha=None,
        created_at="2026-05-22T00:00:00+00:00",
        winner="villagers",
        rounds=1,
    )
    write_manifest(manifest, tmp_path / "manifest.json")
    metrics = extract_run_dir(tmp_path)
    assert {s.name: s.model for s in metrics.seats}["Alice"] == "from-manifest"


def _two_round_werewolf_win_game() -> EventStream:
    return _events(
        [
            (1, Phase.NIGHT, KILL_RESOLVED, {"victim": "Carol"}, ()),
            (1, Phase.DAY, EXILE_RESOLVED, {"ballots": {}, "exiled": None}, ()),
            (2, Phase.NIGHT, KILL_RESOLVED, {"victim": "Bob"}, ()),
            (2, Phase.DAY, GAME_OVER, {"winner": "werewolves"}, ()),
        ]
    )


def test_aggregate_metrics_over_two_games() -> None:
    uniform = (("Alice", "m"), ("Bob", "m"), ("Carol", "m"), ("Dave", "m"))
    manifest_kwargs: dict[str, Any] = dict(
        seed=7,
        players=_PLAYERS,
        models=uniform,
        model_arg="m",
        temperature=0.7,
        max_tokens=8000,
        max_iters=6,
        reasoning=False,
        git_sha=None,
        created_at="2026-05-22T00:00:00+00:00",
    )
    g1 = extract_game_metrics(
        _villager_win_game(),
        _trajectories([]),
        manifest=RunManifest(game_id="g1", winner="villagers", rounds=1, **manifest_kwargs),
    )
    g2 = extract_game_metrics(
        _two_round_werewolf_win_game(),
        _trajectories([]),
        manifest=RunManifest(game_id="g2", winner="werewolves", rounds=2, **manifest_kwargs),
    )

    agg = aggregate_metrics([g1, g2])
    assert agg.n_games == 2
    assert agg.mean_game_length == pytest.approx(1.5)

    by_model = {m.model: m for m in agg.models}
    m = by_model["m"]
    # 4 seats x 2 games = 8 seat-games; villagers(3) win g1 + werewolf(1) wins g2 = 4 wins.
    assert m.games == 8
    assert m.wins == 4
    assert m.win_rate == pytest.approx(0.5)

    by_role = {r.role: r for r in m.roles}
    # The seer wins g1 (villagers) and loses g2 (werewolves) → 1/2.
    assert by_role[Role.SEER.value].games == 2
    assert by_role[Role.SEER.value].wins == 1
    assert by_role[Role.SEER.value].win_rate == pytest.approx(0.5)
    # The werewolf loses g1 and wins g2 → 1/2.
    assert by_role[Role.WEREWOLF.value].wins == 1
    # Per-model role buckets are emitted in sorted order (deterministic output).
    role_order = tuple(r.role for r in m.roles)
    assert role_order == tuple(sorted(role_order))


def test_aggregate_models_are_sorted_by_name() -> None:
    # Two models assigned so first-seen order ("zeta" before "alpha") differs from
    # sorted order — makes the `sorted(model_games)` in aggregate_metrics load-bearing.
    manifest = RunManifest(
        game_id="g1",
        seed=7,
        players=_PLAYERS,
        models=(("Alice", "zeta"), ("Bob", "zeta"), ("Carol", "alpha"), ("Dave", "alpha")),
        model_arg="mixed",
        temperature=0.7,
        max_tokens=8000,
        max_iters=6,
        reasoning=False,
        git_sha=None,
        created_at="2026-05-22T00:00:00+00:00",
        winner="villagers",
        rounds=1,
    )
    game = extract_game_metrics(_villager_win_game(), _trajectories([]), manifest=manifest)
    agg = aggregate_metrics([game])
    assert tuple(m.model for m in agg.models) == ("alpha", "zeta")


# --- integration: a real scripted wolves-win game ------------------------

_ROSTER: tuple[tuple[str, str], ...] = (
    ("Wolf1", Role.WEREWOLF.value),
    ("Wolf2", Role.WEREWOLF.value),
    ("Seer1", Role.SEER.value),
    ("Doc1", Role.DOCTOR.value),
    ("Vil1", Role.VILLAGER.value),
    ("Vil2", Role.VILLAGER.value),
    ("Vil3", Role.VILLAGER.value),
)
_K_SPEAKERS = 3


def _commit(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"next_thought": "t", "next_tool_name": tool, "next_tool_args": args}


def _day_pairs(n_alive: int) -> list[dict[str, Any]]:
    pairs = [_commit("submit_bid", {"amount": 0}) for _ in range(n_alive)]
    pairs += [_commit("speak", {"message": "no comment"}) for _ in range(min(_K_SPEAKERS, n_alive))]
    pairs += [_commit("pass_turn", {}) for _ in range(n_alive)]
    return pairs


def _wolves_win_script() -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    answers += [_commit("werewolf_chat", {"message": "take Vil1"}), _commit("werewolf_chat", {"message": "ok"})]
    answers += [_commit("submit_kill_vote", {"target": "Vil1"}), _commit("submit_kill_vote", {"target": "Vil1"})]
    answers += [_commit("seer_inspect", {"target": "Wolf1"}), _commit("doctor_protect", {"target": "Seer1"})]
    answers += _day_pairs(6)
    answers += [_commit("submit_exile_vote", {"target": ABSTAIN}) for _ in range(6)]
    answers += [_commit("werewolf_chat", {"message": "now Vil2"}), _commit("werewolf_chat", {"message": "ok"})]
    answers += [_commit("submit_kill_vote", {"target": "Vil2"}), _commit("submit_kill_vote", {"target": "Vil2"})]
    answers += [_commit("seer_inspect", {"target": "Wolf2"}), _commit("doctor_protect", {"target": "Vil3"})]
    answers += _day_pairs(5)
    answers += [_commit("submit_exile_vote", {"target": ABSTAIN}) for _ in range(5)]
    answers += [_commit("werewolf_chat", {"message": "finish"}), _commit("werewolf_chat", {"message": "Vil3"})]
    answers += [_commit("submit_kill_vote", {"target": "Vil3"}), _commit("submit_kill_vote", {"target": "Vil3"})]
    answers += [_commit("seer_inspect", {"target": "Doc1"}), _commit("doctor_protect", {"target": "Seer1"})]
    return answers


def test_extract_from_a_real_scripted_wolves_win_game() -> None:
    # `_wolves_win_script()` is one global answer queue in run_game call order, so
    # all seats must share a single DummyLM cursor (the `_uniform_lms` pattern from
    # test_decisions.py). A per-seat copy would make every seat restart at index 0.
    shared = DummyLM(_wolves_win_script())
    source = ReActDecisionSource(roster=_ROSTER, lms={name: shared for name, _ in _ROSTER})
    stream = run_game(_ROSTER, seed=42, decisions=source)
    trajs = TrajectoryStream(header=stream.header, trajectories=source.trajectories)

    metrics = extract_game_metrics(stream, trajs)
    assert metrics.winner == "werewolves"
    assert metrics.rounds == 3
    assert metrics.n_players == 7

    by_name = {s.name: s for s in metrics.seats}
    assert by_name["Wolf1"].won is True
    assert by_name["Vil1"].won is False
    assert by_name["Vil1"].survived is False  # killed night 1

    # Model inference reads the real lm_calls; tool_usage is non-empty and game-action only.
    wolf1_model = next(c.model for t in source.trajectories if t.caller == "Wolf1" for c in t.lm_calls)
    assert by_name["Wolf1"].model == wolf1_model
    usage = dict(metrics.tool_usage)
    assert usage  # at least some committed game actions
    assert not ({"set_belief", "recall", "finish"} & usage.keys())


# --- deceiver/detector split (T25) ---------------------------------------
#
# WEREWOLF_DESIGN.md §10's first-class metric: werewolf win rate (deceiver) vs.
# villager exile accuracy (detector). Exile accuracy is "when the village resolves
# an exile, is the target a wolf"; ties / no-exile are NOT counted (indecision is
# not a detection failure), and a game with zero resolved exiles has
# `exile_accuracy is None` (no decision to judge), distinct from 0.0.


def _villager_exiled_game() -> EventStream:
    """4-seat: Carol (a villager) exiled by day → a *wrong* exile; werewolves win."""
    return _events(
        [
            (1, Phase.NIGHT, KILL_RESOLVED, {"victim": "Bob"}, ()),
            (1, Phase.DAY, EXILE_RESOLVED, {"ballots": {"Alice": "Carol"}, "exiled": "Carol"}, ()),
            (2, Phase.NIGHT, KILL_RESOLVED, {"victim": "Alice"}, ()),
            (2, Phase.DAY, GAME_OVER, {"winner": "werewolves"}, ()),
        ]
    )


def _two_exile_mixed_game() -> EventStream:
    """7-seat realizable game with two resolved exiles: one villager (wrong), one wolf (right).

    On `_ROSTER` (2 wolves, 5 villagers): Vil2 exiled day 1 (wrong), Wolf1 exiled
    day 2 (right), wolves reach parity by night 3. exiles_total=2, exiles_correct=1.
    """
    return _events(
        [
            (1, Phase.NIGHT, KILL_RESOLVED, {"victim": "Vil1"}, ()),
            (1, Phase.DAY, EXILE_RESOLVED, {"ballots": {}, "exiled": "Vil2"}, ()),
            (2, Phase.NIGHT, KILL_RESOLVED, {"victim": "Vil3"}, ()),
            (2, Phase.DAY, EXILE_RESOLVED, {"ballots": {}, "exiled": "Wolf1"}, ()),
            (3, Phase.NIGHT, KILL_RESOLVED, {"victim": "Seer1"}, ()),
            (3, Phase.DAY, GAME_OVER, {"winner": "werewolves"}, ()),
        ],
        players=_ROSTER,
    )


def _mk_manifest(
    *,
    game_id: str,
    players: tuple[tuple[str, str], ...],
    models: tuple[tuple[str, str], ...],
    winner: str,
    rounds: int,
) -> RunManifest:
    return RunManifest(
        game_id=game_id,
        seed=7,
        players=players,
        models=models,
        model_arg="faction",
        temperature=0.7,
        max_tokens=8000,
        max_iters=6,
        reasoning=False,
        git_sha=None,
        created_at="2026-05-22T00:00:00+00:00",
        winner=winner,
        rounds=rounds,
    )


def _faction_models_4(*, wolf: str, village: str) -> tuple[tuple[str, str], ...]:
    return (("Alice", village), ("Bob", village), ("Carol", village), ("Dave", wolf))


def _faction_models_7(*, wolf: str, village: str) -> tuple[tuple[str, str], ...]:
    return tuple((name, wolf if role == Role.WEREWOLF.value else village) for name, role in _ROSTER)


# --- per-game exile accuracy (read from EXILE_RESOLVED, invariant #1) -----


def test_exile_accuracy_counts_a_resolved_wolf_exile_as_correct() -> None:
    # _villager_win_game exiles Dave, the wolf → 1 resolved exile, 1 correct.
    metrics = extract_game_metrics(_villager_win_game(), _trajectories([]))
    assert metrics.exiles_total == 1
    assert metrics.exiles_correct == 1
    assert metrics.exile_accuracy == pytest.approx(1.0)


def test_exile_accuracy_counts_a_resolved_villager_exile_as_incorrect() -> None:
    # Exiling Carol (a villager) is a detection miss: resolved but not a wolf.
    metrics = extract_game_metrics(_villager_exiled_game(), _trajectories([]))
    assert metrics.exiles_total == 1
    assert metrics.exiles_correct == 0
    assert metrics.exile_accuracy == pytest.approx(0.0)


def test_exile_accuracy_is_none_when_no_resolved_exile() -> None:
    # _two_round_werewolf_win_game has only a tie (exiled None). Zero decisions to
    # judge → exile_accuracy is None, NOT 0.0 (which would read as "exiled, missed").
    metrics = extract_game_metrics(_two_round_werewolf_win_game(), _trajectories([]))
    assert metrics.exiles_total == 0
    assert metrics.exiles_correct == 0
    assert metrics.exile_accuracy is None


def test_exile_accuracy_over_two_resolved_exiles_one_each() -> None:
    metrics = extract_game_metrics(_two_exile_mixed_game(), _trajectories([], players=_ROSTER))
    assert metrics.exiles_total == 2
    assert metrics.exiles_correct == 1
    assert metrics.exile_accuracy == pytest.approx(0.5)


# --- aggregate deceiver/detector split -----------------------------------


def test_split_attributes_deceiver_to_wolf_model_and_detector_to_village_model() -> None:
    # Faction cross-play: model A is the wolves, model B is the village, across two
    # 4-seat games. g1: villagers win, exile the wolf (B detects right). g2: wolves
    # win, exile a villager (B detects wrong).
    g1 = extract_game_metrics(
        _villager_win_game(),
        _trajectories([]),
        manifest=_mk_manifest(
            game_id="g1",
            players=_PLAYERS,
            models=_faction_models_4(wolf="A", village="B"),
            winner="villagers",
            rounds=1,
        ),
    )
    g2 = extract_game_metrics(
        _villager_exiled_game(),
        _trajectories([]),
        manifest=_mk_manifest(
            game_id="g2",
            players=_PLAYERS,
            models=_faction_models_4(wolf="A", village="B"),
            winner="werewolves",
            rounds=2,
        ),
    )
    report = deceiver_detector_split([g1, g2])
    by_model = {m.model: m for m in report.models}

    # Deceiver: A is the wolf model in both games; wolves won only g2.
    assert by_model["A"].wolf_games == 2
    assert by_model["A"].wolf_wins == 1
    assert by_model["A"].wolf_win_rate == pytest.approx(0.5)
    # A never sat as the village → no detector decisions attributed to it.
    assert by_model["A"].village_exiles == 0
    assert by_model["A"].exile_accuracy is None

    # Detector: B is the village model in both; 2 resolved exiles, 1 hit a wolf.
    assert by_model["B"].village_exiles == 2
    assert by_model["B"].village_correct_exiles == 1
    assert by_model["B"].exile_accuracy == pytest.approx(0.5)
    # B never sat as the wolves.
    assert by_model["B"].wolf_games == 0

    # Population: wolves won 1 of 2 games; pooled exile accuracy 1/2.
    assert report.n_games == 2
    assert report.wolf_win_rate == pytest.approx(0.5)
    assert report.total_exiles == 2
    assert report.total_correct_exiles == 1
    assert report.exile_accuracy == pytest.approx(0.5)


def test_split_pools_exile_accuracy_across_decisions_not_mean_of_rates() -> None:
    # g1: 2 exiles, 1 correct (rate 0.5). g2: 1 exile, 0 correct (rate 0.0).
    # Mean-of-rates would be 0.25; pooling across the 3 decisions gives 1/3. Both
    # games share village model B so its detector record pools too.
    g1 = extract_game_metrics(
        _two_exile_mixed_game(),
        _trajectories([], players=_ROSTER),
        manifest=_mk_manifest(
            game_id="g1",
            players=_ROSTER,
            models=_faction_models_7(wolf="A", village="B"),
            winner="werewolves",
            rounds=3,
        ),
    )
    g2 = extract_game_metrics(
        _villager_exiled_game(),
        _trajectories([]),
        manifest=_mk_manifest(
            game_id="g2",
            players=_PLAYERS,
            models=_faction_models_4(wolf="A", village="B"),
            winner="werewolves",
            rounds=2,
        ),
    )
    report = deceiver_detector_split([g1, g2])
    assert report.exile_accuracy == pytest.approx(1 / 3)
    assert report.total_exiles == 3
    assert report.total_correct_exiles == 1
    by_model = {m.model: m for m in report.models}
    assert by_model["B"].village_exiles == 3
    assert by_model["B"].village_correct_exiles == 1
    assert by_model["B"].exile_accuracy == pytest.approx(1 / 3)


def test_split_skips_per_model_detector_when_village_faction_is_mixed() -> None:
    # The village faction holds two different models (B1, B2) → no single village
    # model owns the exile decision, so it is excluded from per-model detector
    # attribution, but still counts toward the population pool.
    mixed_village = (("Alice", "B1"), ("Bob", "B2"), ("Carol", "B1"), ("Dave", "A"))
    game = extract_game_metrics(
        _villager_win_game(),  # exiles the wolf → 1 resolved exile, 1 correct
        _trajectories([]),
        manifest=_mk_manifest(
            game_id="g1",
            players=_PLAYERS,
            models=mixed_village,
            winner="villagers",
            rounds=1,
        ),
    )
    report = deceiver_detector_split([game])
    # Population still sees the exile.
    assert report.total_exiles == 1
    assert report.total_correct_exiles == 1
    assert report.exile_accuracy == pytest.approx(1.0)
    # But no model is credited with the detection (village was mixed).
    assert sum(m.village_exiles for m in report.models) == 0
    # The wolf faction was unambiguous (only A) → A still gets its deceiver record.
    by_model = {m.model: m for m in report.models}
    assert by_model["A"].wolf_games == 1
    assert by_model["A"].wolf_wins == 0  # villagers won


def test_split_skips_attribution_for_unidentified_model_faction() -> None:
    # No manifest and no LM calls → every seat's model resolves to the "unknown"
    # sentinel. An unidentified faction must NOT produce a per-model row labelled
    # "unknown" (it isn't a real model); it is skipped for per-model attribution
    # like a mixed faction, but still counts toward the population pool.
    game = extract_game_metrics(_villager_win_game(), _trajectories([]), manifest=None)
    report = deceiver_detector_split([game])
    assert report.models == ()
    assert report.total_exiles == 1
    assert report.total_correct_exiles == 1
    assert report.exile_accuracy == pytest.approx(1.0)


def test_split_models_are_sorted_by_name() -> None:
    # village="alpha", wolf="zeta" so first-seen order differs from sorted order.
    game = extract_game_metrics(
        _villager_win_game(),
        _trajectories([]),
        manifest=_mk_manifest(
            game_id="g1",
            players=_PLAYERS,
            models=_faction_models_4(wolf="zeta", village="alpha"),
            winner="villagers",
            rounds=1,
        ),
    )
    report = deceiver_detector_split([game])
    assert tuple(m.model for m in report.models) == ("alpha", "zeta")


def test_split_over_no_games_is_safe() -> None:
    report = deceiver_detector_split([])
    assert report.n_games == 0
    assert report.wolf_win_rate == 0.0
    assert report.exile_accuracy is None
    assert report.total_exiles == 0
    assert report.total_correct_exiles == 0
    assert report.models == ()
