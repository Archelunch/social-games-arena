"""Per-game replay payload for the static site's game-replay screen.

`build_replay` merges a recorded game's four artifacts — the engine event log,
the agent trajectory sidecar, the run manifest, and the memory dump — into one
deterministic JSON-ready dict that the replay page fetches on demand (one file
per game, lazy-loaded; never inlined into the aggregate `data.json`).

Design choices that keep the frontend a set of pure lookups:

- **The engine event log is the source of truth** (invariant #1). Winner, round
  count, who-is-alive, and every event's `public` flag come from the events, not
  the manifest (which the metrics layer already treats as untrusted). The
  manifest supplies only the seat->model map; when it is absent we fall back to
  the model each seat used in its own trajectory.
- **Per-event `actor`** is derived from the payload the engine wrote (`speaker` /
  `bidder` / `accuser` / `defender`, or the sole recipient of a private role
  action). Referee resolutions (kill/exile/discussion/game_over and the wolves'
  ballot record) belong to no single player -> `actor=None`.
- **The seq<->decision_seq join.** Each decision's terminal tool emits one event;
  we anchor the decision to that event's `seq` so the drawer can show "this
  seat's decision as of the cursor" by a single `anchor_seq <= cursor` scan. A
  terminal that resolves collectively (a kill / exile vote, or a pass) has no
  owned event, so it anchors to its phase's last event (the resolution). A
  rejected first attempt produced an earlier `tool_rejected` event, but the
  decision still anchors to the *successful* terminal event.
- **Memory over time.** Replaying the trajectory's successful `set_belief` /
  `set_plan` calls reproduces the belief/plan state at each anchor, so the drawer
  can show how an agent's read of the table evolved. Notes are not emitted as
  tool calls, so they ship verbatim in `final_memories` (the authoritative end
  state) and the frontend filters them by round.

Determinism (invariant #4): `build_replay` is a pure function of the on-disk
files, so two builds of the same dir are byte-identical. No wall-clock, `git_sha`,
or absolute path enters the payload (those manifest fields are never read).
Telemetry that is *intentionally* non-deterministic (`latency_ms`, `cost_usd`)
rides along for display only and never gates the contract.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from social_deduction_bench.agents.trajectory import Trajectory, TrajectoryStream
from social_deduction_bench.agents.trajectory import read_jsonl as read_trajectories
from social_deduction_bench.engine.events import Event, EventStream
from social_deduction_bench.engine.events import read_jsonl as read_events
from social_deduction_bench.games.werewolf.events import (
    ACCUSATION,
    BID,
    DEFENSE,
    DOCTOR_PROTECT,
    GAME_OVER,
    SEER_INSPECT,
    SPEECH,
    WEREWOLF_CHAT,
)
from social_deduction_bench.games.werewolf.roles import Role
from social_deduction_bench.games.werewolf.site_data import iter_game_dirs
from social_deduction_bench.rating.manifest import RunManifest
from social_deduction_bench.rating.manifest import read_json as read_manifest

logger = logging.getLogger(__name__)

_WOLF_ROLE = Role.WEREWOLF.value

# The single event type a terminal tool commits, when it owns one. Vote and pass
# terminals resolve collectively and own no event (they anchor to the phase's
# resolution), so they are deliberately absent.
_TERMINAL_EVENT_TYPE: dict[str, str] = {
    "speak": SPEECH,
    "accuse": ACCUSATION,
    "defend": DEFENSE,
    "submit_bid": BID,
    "werewolf_chat": WEREWOLF_CHAT,
    "seer_inspect": SEER_INSPECT,
    "doctor_protect": DOCTOR_PROTECT,
}

# Private role actions whose actor is the sole recipient (the seer/doctor/caller),
# since their payload names the target, not the author.
_RECIPIENT_ACTOR_TYPES = frozenset({SEER_INSPECT, DOCTOR_PROTECT, "tool_rejected"})


def _event_actor(event: Event) -> str | None:
    """The player who caused `event`, or None for a referee resolution.

    Public dialogue names its author in the payload; private role actions name
    only their target, so their author is the sole recipient. Resolutions
    (kill/exile/discussion/game_over and the pack's ballot record) have no author.
    """
    payload = event.payload
    for key in ("speaker", "bidder", "accuser", "defender"):
        actor = payload.get(key)
        if isinstance(actor, str):
            return actor
    if event.type in _RECIPIENT_ACTOR_TYPES and len(event.recipients) == 1:
        return event.recipients[0]
    return None


def _model_map(manifest: RunManifest | None, trajectories: TrajectoryStream) -> dict[str, str]:
    """Seat -> model, from the manifest when present, else inferred per trajectory.

    A seat that never called an LM (e.g. a player killed before acting) has no
    inferable model and is simply omitted, leaving its `model` None.
    """
    if manifest is not None:
        return {seat: model for seat, model in manifest.models}
    inferred: dict[str, str] = {}
    for traj in trajectories.trajectories:
        if traj.caller not in inferred and traj.lm_calls:
            inferred[traj.caller] = traj.lm_calls[0].model
    return inferred


def _unique_model(roles: dict[str, str], models: dict[str, str], *, wolf: bool) -> str | None:
    """The single model staffing one faction, or None when mixed/unknown."""
    seats = {models.get(name) for name, role in roles.items() if (role == _WOLF_ROLE) is wolf}
    seats.discard(None)
    return next(iter(seats)) if len(seats) == 1 else None


def _phases(events: tuple[Event, ...]) -> list[dict[str, object]]:
    """Contiguous (round, phase) blocks with their seq span and a display label."""
    blocks: list[dict[str, object]] = []
    for event in events:
        phase = event.phase.value
        if blocks and blocks[-1]["round"] == event.round and blocks[-1]["phase"] == phase:
            blocks[-1]["last_seq"] = event.seq
            continue
        label = f"{'Night' if phase == 'night' else 'Day'} {event.round}"
        blocks.append(
            {"round": event.round, "phase": phase, "first_seq": event.seq, "last_seq": event.seq, "label": label}
        )
    return blocks


def _anchor_seqs(events: tuple[Event, ...], trajectories: tuple[Trajectory, ...]) -> dict[int, int]:
    """Map each decision_seq to the event seq that represents it.

    Owned terminals consume their matching event in order (werewolf_chat, which a
    seat can emit several times in one night, is disambiguated by the committed
    message). Collective terminals (votes, pass) anchor to their phase's last
    event — the resolution everyone sees.
    """
    phase_last: dict[tuple[int, str], int] = {}
    owned: dict[tuple[int, str, str, str], list[Event]] = {}
    for event in events:
        # game_over is a meta event, not a phase resolution a vote produces, so a
        # collective terminal anchors to the substantive resolution before it.
        if event.type != GAME_OVER:
            phase_last[(event.round, event.phase.value)] = event.seq
        actor = _event_actor(event)
        if actor is not None:
            owned.setdefault((event.round, event.phase.value, event.type, actor), []).append(event)

    anchors: dict[int, int] = {}
    for traj in trajectories:
        owned_type = _TERMINAL_EVENT_TYPE.get(traj.terminal_tool)
        if owned_type is not None:
            bucket = owned.get((traj.round, traj.phase, owned_type, traj.caller), [])
            chosen = _take_owned_event(bucket, owned_type, traj.committed_value)
            if chosen is not None:
                anchors[traj.decision_seq] = chosen.seq
                continue
        anchors[traj.decision_seq] = phase_last.get((traj.round, traj.phase), -1)
    return anchors


def _take_owned_event(bucket: list[Event], owned_type: str, committed_value: object) -> Event | None:
    """Pop the event this decision committed, consuming it so a later sibling can't reuse it.

    werewolf_chat is the only type a seat emits more than once per phase, so when
    several remain we match the committed message; otherwise the first unconsumed
    event in seq order is this decision's.
    """
    if not bucket:
        return None
    if owned_type == WEREWOLF_CHAT and len(bucket) > 1:
        for event in bucket:
            if event.payload.get("message") == committed_value:
                bucket.remove(event)
                return event
    return bucket.pop(0)


def _memory_timeline(trajectories: tuple[Trajectory, ...], anchors: dict[int, int]) -> list[dict[str, object]]:
    """Cumulative belief/plan snapshots, one per decision that mutated memory.

    Only successful tool calls (observation prefixed "ok") count; a rejected
    `set_belief` is still recorded in the trajectory but never changed the agent's
    real memory. Each snapshot is keyed by the decision's anchor seq so the drawer
    can pick the latest state at or before the cursor.
    """
    running: dict[str, dict[str, Any]] = {}
    snapshots: list[dict[str, object]] = []
    for traj in trajectories:
        state = running.setdefault(traj.caller, {"plan": "", "beliefs": {}})
        mutated = False
        for step in traj.react_trajectory:
            if not step.observation.startswith("ok"):
                continue
            if step.tool == "set_belief" and {"player", "guess", "confidence", "evidence"} <= step.args.keys():
                state["beliefs"][step.args["player"]] = {
                    "guess": step.args["guess"],
                    "confidence": step.args["confidence"],
                    "evidence": step.args["evidence"],
                }
                mutated = True
            elif step.tool == "set_plan" and "text" in step.args:
                state["plan"] = step.args["text"]
                mutated = True
        if mutated:
            snapshots.append(
                {
                    "at_seq": anchors.get(traj.decision_seq, -1),
                    "player": traj.caller,
                    "plan": state["plan"],
                    "beliefs": {subject: dict(row) for subject, row in state["beliefs"].items()},
                }
            )
    return snapshots


def _winner(events: tuple[Event, ...]) -> str:
    for event in events:
        if event.type == "game_over":
            winner = event.payload.get("winner")
            if isinstance(winner, str):
                return winner
    raise ValueError("transcript has no game_over event")


def build_replay(game_dir: Path) -> dict[str, Any]:
    """Merge one recorded game's artifacts into the replay payload.

    Reads `events.jsonl` and `trajectories.jsonl` (required), `manifest.json` and
    `memories.json` (optional). The return is the JSON document boundary, so it is
    typed `dict[str, Any]` rather than a domain model.
    """
    events_stream: EventStream = read_events(game_dir / "events.jsonl")
    trajectories_stream: TrajectoryStream = read_trajectories(game_dir / "trajectories.jsonl")
    events = events_stream.log.events
    trajectories = trajectories_stream.trajectories

    manifest_path = game_dir / "manifest.json"
    manifest = read_manifest(manifest_path) if manifest_path.exists() else None
    models = _model_map(manifest, trajectories_stream)
    roles = {name: role for name, role in events_stream.header.players}

    memories_path = game_dir / "memories.json"
    final_memories: dict[str, Any] = {}
    if memories_path.exists():
        final_memories = json.loads(memories_path.read_text(encoding="utf-8")).get("memories", {})

    anchors = _anchor_seqs(events, trajectories)

    return {
        "game_id": events_stream.header.game_id,
        "seed": events_stream.header.seed,
        "winner": _winner(events),
        "rounds": max((event.round for event in events), default=0),
        "wolf_model": _unique_model(roles, models, wolf=True),
        "village_model": _unique_model(roles, models, wolf=False),
        "players": [
            {"name": name, "role": role, "model": models.get(name)} for name, role in events_stream.header.players
        ],
        "phases": _phases(events),
        "events": [
            {
                "seq": event.seq,
                "round": event.round,
                "phase": event.phase.value,
                "type": event.type,
                "payload": dict(event.payload),
                "recipients": list(event.recipients),
                "public": event.is_public,
                "actor": _event_actor(event),
            }
            for event in events
        ],
        "trajectories": [
            {**traj.to_json_dict(), "anchor_seq": anchors.get(traj.decision_seq, -1)} for traj in trajectories
        ],
        "memory_timeline": _memory_timeline(trajectories, anchors),
        "final_memories": final_memories,
    }


def write_replays(run_dirs: Sequence[Path], out_dir: Path) -> int:
    """Write one `out_dir/games/<game_id>.json` per completed game; return the count.

    Globs the same `g*/` dirs the aggregate build reads. A torn / in-progress dir
    (no `game_over`, half-written file) is logged and skipped rather than aborting
    the build — the run dir is written live by a parallel sweep. Each file is
    `sort_keys=True` so a fixed game dir yields byte-identical output (invariant #4);
    the files are compact (no indent) because they are fetched by the page, not read
    by hand, and the full per-game trajectories make them large.
    """
    games_out = out_dir / "games"
    games_out.mkdir(parents=True, exist_ok=True)
    written = 0
    for game_dir in iter_game_dirs(run_dirs):
        try:
            payload = build_replay(game_dir)
        except (OSError, ValueError, KeyError) as e:
            logger.warning("site: skipping unreadable game dir %s: %s", game_dir, e)
            continue
        (games_out / f"{payload['game_id']}.json").write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        written += 1
    return written
