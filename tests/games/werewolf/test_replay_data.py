"""Tests for the per-game replay payload builder (game-replay site page).

`build_replay` merges a recorded game's four on-disk artifacts (events,
trajectories, manifest, memories) into one JSON-ready dict the static replay
screen consumes. These tests pin the contract the frontend relies on:

- determinism: a fixed game dir is a pure input, so two builds are byte-equal
  (invariant #4 applied to a derived artifact);
- no provenance leak: wall-clock / git_sha never enter the payload, so a
  re-record with a new sha/timestamp would not change it;
- the engine event log stays the source of truth for winner/rounds/visibility
  (invariant #1/#2): each event's `public` flag mirrors `Event.is_public` and
  the per-event `actor` is derived from the payload the engine wrote;
- the seq<->decision_seq join: every decision anchors to the public event that
  represents it, even when an earlier rejected attempt produced an event first;
- memory reconstruction: replaying the trajectory's `set_belief`/`set_plan`
  calls reproduces `memories.json`'s final belief/plan state (the oracle).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from social_deduction_bench.agents.trajectory import (
    LMCallRecord,
    ReActStep,
    Trajectory,
    TrajectoryStream,
)
from social_deduction_bench.agents.trajectory import write_jsonl as write_trajectories
from social_deduction_bench.engine import EventLog, EventStream, Phase, StreamHeader
from social_deduction_bench.engine import write_jsonl as write_events
from social_deduction_bench.games.werewolf.replay_data import build_replay, write_replays
from social_deduction_bench.rating.manifest import RunManifest
from social_deduction_bench.rating.manifest import write_json as write_manifest

# `write_game_dir` / `two_game_run` are auto-injected fixtures from tests/conftest.py.
GameDirWriter = Callable[..., Path]

_PLAYERS: tuple[tuple[str, str], ...] = (
    ("Alice", "villager"),
    ("Bob", "doctor"),
    ("Carol", "werewolf"),
    ("Dave", "seer"),
    ("Eve", "werewolf"),
    ("Grace", "villager"),
)
_MODELS: tuple[tuple[str, str], ...] = (
    ("Alice", "v-model"),
    ("Bob", "v-model"),
    ("Carol", "w-model"),
    ("Dave", "v-model"),
    ("Eve", "w-model"),
    ("Grace", "v-model"),
)


def _step(tool: str, args: dict[str, object], observation: str) -> ReActStep:
    return ReActStep(iter=0, thought="t", tool=tool, args=args, observation=observation)


def _lm() -> LMCallRecord:
    return LMCallRecord(model="w-model", prompt_tokens=100, completion_tokens=20, latency_ms=12.5, cost_usd=0.001)


def _traj(
    decision_seq: int,
    round_: int,
    phase: str,
    caller: str,
    role: str,
    terminal_tool: str,
    committed_value: object,
    steps: tuple[ReActStep, ...],
) -> Trajectory:
    return Trajectory(
        decision_seq=decision_seq,
        round=round_,
        phase=phase,
        caller=caller,
        role=role,
        terminal_tool=terminal_tool,
        committed_value=committed_value,
        react_trajectory=steps,
        lm_calls=(_lm(),),
    )


def _write_game(game_dir: Path) -> Path:
    """Write a synthetic 6-seat one-round game exercising every join branch.

    Covers: a terminal werewolf_chat vs an intermediate one (same speaker, two
    events, disambiguated by the committed message); a doctor whose first attempt
    is rejected (the decision must anchor to the successful protect, not the
    rejection); a seer whose decision carries a `set_belief`; a day speak that
    carries a `set_plan`; and the four referee resolutions that belong to no
    decision (kill_resolved / discussion_resolved / exile_resolved / game_over).
    """
    log = EventLog()
    P = Phase
    # round 1 night
    log.append(
        round=1,
        phase=P.NIGHT,
        type="werewolf_chat",
        payload={"message": "open", "speaker": "Carol"},
        recipients=("Carol", "Eve"),
    )  # 0
    log.append(
        round=1,
        phase=P.NIGHT,
        type="tool_rejected",
        payload={"tool": "doctor_protect", "args": {"target": "Bob"}, "reason": "no self"},
        recipients=("Bob",),
    )  # 1
    log.append(
        round=1,
        phase=P.NIGHT,
        type="werewolf_chat",
        payload={"message": "kill Grace", "speaker": "Carol"},
        recipients=("Carol", "Eve"),
    )  # 2
    log.append(round=1, phase=P.NIGHT, type="doctor_protect", payload={"target": "Alice"}, recipients=("Bob",))  # 3
    log.append(
        round=1,
        phase=P.NIGHT,
        type="seer_inspect",
        payload={"target": "Carol", "faction": "werewolves"},
        recipients=("Dave",),
    )  # 4
    log.append(
        round=1,
        phase=P.NIGHT,
        type="kill_ballots",
        payload={"ballots": {"Carol": "Grace", "Eve": "Grace"}},
        recipients=("Carol", "Eve"),
    )  # 5
    log.append(round=1, phase=P.NIGHT, type="kill_resolved", payload={"victim": "Grace"})  # 6
    # round 1 day
    log.append(round=1, phase=P.DAY, type="bid", payload={"amount": 20, "bidder": "Alice"}, recipients=("Alice",))  # 7
    log.append(
        round=1, phase=P.DAY, type="discussion_resolved", payload={"bids": {"Alice": 20}, "speakers": ["Alice"]}
    )  # 8
    log.append(round=1, phase=P.DAY, type="speech", payload={"message": "hello", "speaker": "Alice"})  # 9
    log.append(
        round=1, phase=P.DAY, type="accusation", payload={"accuser": "Carol", "target": "Dave", "reason": "r"}
    )  # 10
    log.append(
        round=1, phase=P.DAY, type="exile_resolved", payload={"ballots": {"Alice": "Carol"}, "exiled": "Carol"}
    )  # 11
    log.append(round=1, phase=P.DAY, type="game_over", payload={"winner": "villagers"})  # 12
    header = StreamHeader(seed=42, game_id="g-test", players=_PLAYERS)
    write_events(EventStream(header=header, log=log), game_dir / "events.jsonl")

    trajs = (
        _traj(
            0,
            1,
            "night",
            "Carol",
            "werewolf",
            "werewolf_chat",
            "open",
            (_step("werewolf_chat", {"message": "open"}, "ok"),),
        ),
        _traj(
            1,
            1,
            "night",
            "Bob",
            "doctor",
            "doctor_protect",
            "Alice",
            (
                _step("doctor_protect", {"target": "Bob"}, "error: no self"),
                _step("doctor_protect", {"target": "Alice"}, "ok: protected"),
            ),
        ),
        _traj(
            2,
            1,
            "night",
            "Dave",
            "seer",
            "seer_inspect",
            "Carol",
            (
                _step(
                    "set_belief",
                    {"player": "Carol", "guess": "werewolf", "confidence": "high", "evidence": "seer"},
                    "ok: belief set",
                ),
                _step("seer_inspect", {"target": "Carol"}, "ok"),
            ),
        ),
        _traj(
            3,
            1,
            "night",
            "Carol",
            "werewolf",
            "submit_kill_vote",
            "Grace",
            (
                _step("werewolf_chat", {"message": "kill Grace"}, "ok"),
                _step("submit_kill_vote", {"target": "Grace"}, "ok"),
            ),
        ),
        _traj(4, 1, "day", "Alice", "villager", "submit_bid", 20, (_step("submit_bid", {"amount": 20}, "ok"),)),
        _traj(
            5,
            1,
            "day",
            "Alice",
            "villager",
            "speak",
            "hello",
            (
                _step("set_plan", {"text": "win"}, "ok: plan set"),
                _step("speak", {"message": "hello"}, "ok"),
            ),
        ),
        _traj(
            6,
            1,
            "day",
            "Carol",
            "werewolf",
            "accuse",
            "Dave",
            (_step("accuse", {"target": "Dave", "reason": "r"}, "ok"),),
        ),
        _traj(
            7,
            1,
            "day",
            "Alice",
            "villager",
            "submit_exile_vote",
            "Carol",
            (_step("submit_exile_vote", {"target": "Carol"}, "ok"),),
        ),
    )
    write_trajectories(TrajectoryStream(header=header, trajectories=trajs), game_dir / "trajectories.jsonl")

    write_manifest(
        RunManifest(
            game_id="g-test",
            seed=42,
            players=_PLAYERS,
            models=_MODELS,
            model_arg="tournament",
            temperature=1.0,
            max_tokens=8000,
            max_iters=9,
            reasoning=False,
            git_sha="deadbeefcafe",
            created_at="2026-01-01T00:00:00+00:00",
            winner="villagers",
            rounds=1,
        ),
        game_dir / "manifest.json",
    )

    memories = {
        "game_id": "g-test",
        "memories": {
            "Dave": {
                "plan": "",
                "beliefs": {"Carol": {"guess": "werewolf", "confidence": "high", "evidence": "seer"}},
                "notes": [{"round": 1, "text": "inspected Carol"}],
            },
            "Alice": {"plan": "win", "beliefs": {}, "notes": [{"round": 1, "text": "spoke up"}]},
        },
    }
    (game_dir / "memories.json").write_text(json.dumps(memories), encoding="utf-8")
    return game_dir


@pytest.fixture
def game(tmp_path: Path) -> Path:
    return _write_game(tmp_path)


# --- determinism & no provenance leak ------------------------------------


def test_build_replay_is_deterministic_for_a_fixed_game_dir(game: Path) -> None:
    a = json.dumps(build_replay(game), sort_keys=True)
    b = json.dumps(build_replay(game), sort_keys=True)
    assert a == b


def test_replay_payload_never_embeds_wallclock_or_git_sha(game: Path) -> None:
    blob = json.dumps(build_replay(game))
    # A re-record with a different sha/timestamp must not change the replay,
    # so neither the manifest's created_at nor git_sha may leak in.
    assert "deadbeefcafe" not in blob
    assert "2026-01-01" not in blob
    assert "git_sha" not in blob
    assert "created_at" not in blob


# --- header / roster: engine + manifest merge ----------------------------


def test_winner_and_rounds_come_from_the_event_log_not_the_manifest(game: Path) -> None:
    out = build_replay(game)
    assert out["game_id"] == "g-test"
    assert out["seed"] == 42
    assert out["winner"] == "villagers"
    assert out["rounds"] == 1


def test_players_merge_role_from_header_and_model_from_manifest(game: Path) -> None:
    by_name = {p["name"]: p for p in build_replay(game)["players"]}
    assert by_name["Carol"] == {"name": "Carol", "role": "werewolf", "model": "w-model"}
    assert by_name["Alice"]["role"] == "villager"
    assert by_name["Alice"]["model"] == "v-model"


def test_unique_faction_models_label_the_matchup(game: Path) -> None:
    out = build_replay(game)
    assert out["wolf_model"] == "w-model"
    assert out["village_model"] == "v-model"


# --- per-event visibility & actor (engine = source of truth) -------------


def test_public_flag_mirrors_event_recipients(game: Path) -> None:
    events = build_replay(game)["events"]
    assert events[0]["public"] is False  # werewolf_chat is private to the pack
    assert events[9]["public"] is True  # speech is broadcast


def test_actor_is_derived_from_payload_or_recipient(game: Path) -> None:
    events = build_replay(game)["events"]
    assert events[0]["actor"] == "Carol"  # speaker
    assert events[3]["actor"] == "Bob"  # doctor_protect: the sole recipient
    assert events[4]["actor"] == "Dave"  # seer_inspect: the sole recipient
    assert events[7]["actor"] == "Alice"  # bid: bidder
    assert events[9]["actor"] == "Alice"  # speech: speaker
    assert events[10]["actor"] == "Carol"  # accusation: accuser


def test_referee_resolutions_have_no_actor(game: Path) -> None:
    by_seq = {e["seq"]: e for e in build_replay(game)["events"]}
    for seq in (5, 6, 8, 11, 12):  # kill_ballots, kill_resolved, discussion_resolved, exile_resolved, game_over
        assert by_seq[seq]["actor"] is None


# --- the seq <-> decision_seq join ---------------------------------------


def test_decisions_anchor_to_the_event_that_represents_them(game: Path) -> None:
    anchor = {t["decision_seq"]: t["anchor_seq"] for t in build_replay(game)["trajectories"]}
    assert anchor[0] == 0  # Carol's terminal werewolf_chat -> the "open" event, not the later "kill Grace"
    assert anchor[2] == 4  # seer
    assert anchor[4] == 7  # bid
    assert anchor[5] == 9  # speak
    assert anchor[6] == 10  # accuse


def test_rejected_attempt_does_not_become_the_anchor(game: Path) -> None:
    anchor = {t["decision_seq"]: t["anchor_seq"] for t in build_replay(game)["trajectories"]}
    # Bob's first protect was rejected (event seq 1); the decision must anchor to
    # the successful doctor_protect (event seq 3), never the tool_rejected.
    assert anchor[1] == 3


def test_vote_decisions_anchor_to_their_phase_resolution(game: Path) -> None:
    anchor = {t["decision_seq"]: t["anchor_seq"] for t in build_replay(game)["trajectories"]}
    assert anchor[3] == 6  # submit_kill_vote -> last night event (kill_resolved)
    assert anchor[7] == 11  # submit_exile_vote -> exile_resolved


# --- memory reconstruction timeline & oracle -----------------------------


def test_memory_timeline_snapshots_beliefs_and_plan_at_their_anchor(game: Path) -> None:
    snaps = build_replay(game)["memory_timeline"]
    dave = [s for s in snaps if s["player"] == "Dave"]
    assert dave
    assert dave[-1]["at_seq"] == 4
    assert dave[-1]["beliefs"]["Carol"]["guess"] == "werewolf"
    alice = [s for s in snaps if s["player"] == "Alice"]
    assert alice
    assert alice[-1]["plan"] == "win"


def test_reconstructed_final_state_matches_memories_json(game: Path) -> None:
    out = build_replay(game)
    final = out["final_memories"]
    # The authoritative end state ships verbatim (it is the only source of notes).
    assert final["Dave"]["notes"] == [{"round": 1, "text": "inspected Carol"}]
    # And the reconstructed timeline's final belief/plan per player equals it.
    last: dict[str, dict] = {}
    for snap in out["memory_timeline"]:
        last[snap["player"]] = snap
    assert last["Dave"]["beliefs"] == final["Dave"]["beliefs"]
    assert last["Alice"]["plan"] == final["Alice"]["plan"]


# --- phase segmentation --------------------------------------------------


def test_phases_segment_the_timeline_by_round_and_phase(game: Path) -> None:
    phases = build_replay(game)["phases"]
    assert [(p["round"], p["phase"]) for p in phases] == [(1, "night"), (1, "day")]
    night, day = phases
    assert (night["first_seq"], night["last_seq"]) == (0, 6)
    assert (day["first_seq"], day["last_seq"]) == (7, 12)
    assert night["label"] == "Night 1"
    assert day["label"] == "Day 1"


# --- write_replays: one file per game, skip torn dirs --------------------


def test_write_replays_emits_one_json_per_completed_game(tmp_path: Path, two_game_run: Path) -> None:
    out = tmp_path / "site"
    written = write_replays([two_game_run], out)
    assert written == 2
    files = sorted(p.name for p in (out / "games").glob("*.json"))
    assert files == ["g0000-A-vs-B.json", "g0001-B-vs-A.json"]
    payload = json.loads((out / "games" / "g0000-A-vs-B.json").read_text())
    assert payload["game_id"] == "g0000-A-vs-B"
    assert payload["winner"] == "werewolves"
    assert {p["name"] for p in payload["players"]} == {"Alice", "Bob", "Carol", "Dave", "Eve"}


def test_write_replays_skips_torn_dirs(tmp_path: Path, write_game_dir: GameDirWriter) -> None:
    run = tmp_path / "run"
    write_game_dir(run, "g0000-A-vs-B", wolf_model="A", village_model="B", winner="werewolves")
    write_game_dir(run, "g0001-A-vs-B", wolf_model="A", village_model="B", winner="werewolves", with_game_over=False)
    out = tmp_path / "site"
    assert write_replays([run], out) == 1
    assert [p.name for p in (out / "games").glob("*.json")] == ["g0000-A-vs-B.json"]


def test_write_replays_files_never_leak_provenance(tmp_path: Path, two_game_run: Path) -> None:
    out = tmp_path / "site"
    write_replays([two_game_run], out)
    blob = (out / "games" / "g0000-A-vs-B.json").read_text()
    # The fixture manifest carries a git_sha + created_at; neither may reach the page.
    assert "deadbeefcafe" not in blob
    assert "git_sha" not in blob
    assert "created_at" not in blob


def test_write_replays_is_byte_stable(tmp_path: Path, two_game_run: Path) -> None:
    write_replays([two_game_run], tmp_path / "one")
    write_replays([two_game_run], tmp_path / "two")
    first = (tmp_path / "one" / "games" / "g0000-A-vs-B.json").read_bytes()
    second = (tmp_path / "two" / "games" / "g0000-A-vs-B.json").read_bytes()
    assert first == second


# --- real recorded game (skips when the run dir is absent) ---------------


def _g0000() -> Path | None:
    root = Path(__file__).resolve().parents[3]
    matches = sorted((root / "games" / "run2").glob("g0000-*")) if (root / "games" / "run2").is_dir() else []
    return matches[0] if matches else None


@pytest.mark.skipif(_g0000() is None, reason="games/run2 recordings not present")
def test_build_replay_on_a_real_recording() -> None:
    out = build_replay(_g0000())  # type: ignore[arg-type]
    assert out["winner"] == "werewolves"
    assert out["events"][0]["type"] == "werewolf_chat"
    assert out["events"][0]["actor"] == "Carol"
    # Bob's doctor decision (decision_seq 2) rejected its first protect (event seq 2)
    # then succeeded (event seq 6); it must anchor to the success.
    bob = next(t for t in out["trajectories"] if t["decision_seq"] == 2)
    assert bob["caller"] == "Bob"
    assert bob["anchor_seq"] == 6
    # Every resolution is actorless; every speaker/recipient event has an actor.
    for e in out["events"]:
        if e["type"] in {"kill_resolved", "exile_resolved", "game_over", "discussion_resolved", "kill_ballots"}:
            assert e["actor"] is None


@pytest.mark.skipif(_g0000() is None, reason="games/run2 recordings not present")
def test_real_recording_reconstructs_its_memories_json() -> None:
    game_dir = _g0000()
    out = build_replay(game_dir)  # type: ignore[arg-type]
    final = out["final_memories"]
    last: dict[str, dict] = {}
    for snap in out["memory_timeline"]:
        last[snap["player"]] = snap
    for player, mem in final.items():
        recon = last.get(player, {"beliefs": {}, "plan": ""})
        assert recon["beliefs"] == mem["beliefs"], player
        assert recon["plan"] == mem["plan"], player
