"""Werewolf per-game and aggregate metric extraction (T24).

`extract_game_metrics` turns one game's `EventStream` + `TrajectoryStream` into a
structured `GameMetrics`: winner, game length, per-seat win/survival, token
telemetry, and illegal-move counts. The engine event log is the source of truth
for outcomes (invariant #1) — winner and deaths come from events, never the
manifest; the trajectory sidecar only supplies agent-internal telemetry (tokens,
tool calls). The optional `RunManifest` is consulted ONLY for per-seat model
identity, since a seat that never calls an LM has none in its trajectories.

A "game action" is one of `WEREWOLF_TOOL_REQUIREMENTS`' canonical tools; cognitive
and control tools (`set_belief`, `recall`, `finish`, ...) are excluded from move
metrics. Illegal moves are counted from the engine's `TOOL_REJECTED` events — the
referee is the authority on what is illegal (invariant #1) — not from trajectory
`"error:"` observations, which also cover tool *execution* faults the engine never
rejected (e.g. a tool fn raising on malformed LLM args).

Determinism: all serialized/stored collections have a deterministic order (seats
in roster order; `tool_usage` and aggregate buckets sorted), so identical inputs
yield equal value types (invariant #4).
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from social_deduction_bench.agents.trajectory import Trajectory, TrajectoryStream
from social_deduction_bench.agents.trajectory import read_jsonl as read_trajectories
from social_deduction_bench.engine import EventStream
from social_deduction_bench.engine import read_jsonl as read_events
from social_deduction_bench.games.werewolf.events import EXILE_RESOLVED, GAME_OVER, KILL_RESOLVED, TOOL_REJECTED
from social_deduction_bench.games.werewolf.roles import Faction, faction_of
from social_deduction_bench.games.werewolf.tools import WEREWOLF_TOOL_REQUIREMENTS
from social_deduction_bench.rating.manifest import RunManifest
from social_deduction_bench.rating.manifest import read_json as read_manifest

_GAME_ACTIONS: frozenset[str] = frozenset(WEREWOLF_TOOL_REQUIREMENTS)

# Sentinel for a seat whose model identity could not be resolved (no manifest and
# no LM call to infer from). Never a real model name, so per-model rollups must not
# attribute a faction record to it.
_UNKNOWN_MODEL = "unknown"


@dataclass(frozen=True, slots=True)
class SeatMetrics:
    """One seat's per-game telemetry: outcome, survival, tokens, and tool counts."""

    name: str
    role: str
    model: str
    faction: str
    won: bool
    survived: bool
    prompt_tokens: int
    completion_tokens: int
    lm_calls: int
    tool_calls: int
    illegal_moves: int
    cost_usd: float | None


@dataclass(frozen=True, slots=True)
class GameMetrics:
    """One game's full metric record: outcome plus per-seat and game-level totals."""

    game_id: str
    seed: int
    winner: str
    rounds: int
    n_players: int
    seats: tuple[SeatMetrics, ...]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_lm_calls: int
    total_tool_calls: int
    total_illegal_moves: int
    illegal_move_rate: float
    total_cost_usd: float | None
    tool_usage: tuple[tuple[str, int], ...]
    exiles_total: int
    exiles_correct: int
    exile_accuracy: float | None


@dataclass(frozen=True, slots=True)
class RoleStats:
    """A model's win record while playing one role, across seat-games."""

    role: str
    games: int
    wins: int
    win_rate: float


@dataclass(frozen=True, slots=True)
class ModelStats:
    """A model's aggregate record across seat-games (and its per-role breakdown)."""

    model: str
    games: int
    wins: int
    win_rate: float
    prompt_tokens: int
    completion_tokens: int
    illegal_moves: int
    tool_calls: int
    roles: tuple[RoleStats, ...]


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Cross-game rollup over seat-games, bucketed by model."""

    n_games: int
    mean_game_length: float
    mean_illegal_move_rate: float
    models: tuple[ModelStats, ...]


@dataclass(frozen=True, slots=True)
class FactionSplit:
    """One model's deceiver (wolf) and detector (village exile) record."""

    model: str
    wolf_games: int
    wolf_wins: int
    wolf_win_rate: float
    village_exiles: int
    village_correct_exiles: int
    exile_accuracy: float | None  # None when village_exiles == 0


@dataclass(frozen=True, slots=True)
class DeceiverDetectorReport:
    """Cross-game deceiver-vs-detector split: wolf win rate vs. village exile accuracy."""

    n_games: int
    wolf_win_rate: float
    exile_accuracy: float | None  # pooled across ALL resolved exiles; None when total == 0
    total_exiles: int
    total_correct_exiles: int
    models: tuple[FactionSplit, ...]  # sorted by model name


def _dead_seats(events: EventStream) -> set[str]:
    """Return the set of seats that died: night victims plus day exiles (skip None)."""
    dead: set[str] = set()
    for event in events.log.events:
        if event.type == KILL_RESOLVED:
            victim = event.payload.get("victim")
            if victim is not None:
                dead.add(victim)  # type: ignore[arg-type]
        elif event.type == EXILE_RESOLVED:
            exiled = event.payload.get("exiled")
            if exiled is not None:
                dead.add(exiled)  # type: ignore[arg-type]
    return dead


def _exile_accuracy(events: EventStream) -> tuple[int, int]:
    """Return (resolved_exiles, correct_exiles) read from `EXILE_RESOLVED` events.

    Detection accuracy is "when the village resolves an exile, is the target a
    wolf". A resolved exile is an `EXILE_RESOLVED` event whose `exiled` is not None
    (ties / no-exile are indecision, not a detection failure, so they are not
    counted). It is correct when the exiled seat's role is a werewolf, per the
    roster in the stream header (invariant #1: outcomes from the event log).
    """
    roles = dict(events.header.players)
    resolved = 0
    correct = 0
    for event in events.log.events:
        if event.type == EXILE_RESOLVED:
            exiled = event.payload.get("exiled")
            if exiled is not None:
                resolved += 1
                if faction_of(roles[exiled]) is Faction.WEREWOLVES:  # type: ignore[index]
                    correct += 1
    return resolved, correct


def _illegal_by_seat(events: EventStream) -> Counter[str]:
    """Per-seat illegal-move count, taken from the engine's `TOOL_REJECTED` events.

    The engine (referee) is the authority on what is illegal (invariant #1): a move
    it rejected is recorded as a private `TOOL_REJECTED` event addressed to the
    caller. A tool that *raises* during execution yields an `"error:"` trajectory
    observation but no rejection event — that is a tool fault, not an illegal move,
    and must not inflate the rate.
    """
    counts: Counter[str] = Counter()
    for event in events.log.events:
        if event.type == TOOL_REJECTED:
            for name in event.recipients:
                counts[name] += 1
    return counts


def _winner_and_rounds(events: EventStream) -> tuple[str, int]:
    """Return the winning faction and game length (max round) from the event log.

    Winner is read from the (terminal) `GAME_OVER` event; the same `next(...)`
    first-match convention the CLI uses. Fail loud (invariant #1, CLAUDE.md rule
    11) when there is no `GAME_OVER` event: an incomplete transcript has no winner
    and computing one would silently invent it.
    """
    game_over = next((e for e in events.log.events if e.type == GAME_OVER), None)
    if game_over is None:
        raise ValueError("event stream has no game_over event: cannot determine winner")
    winner: str = game_over.payload["winner"]  # type: ignore[assignment]
    rounds = max((e.round for e in events.log.events), default=0)
    return winner, rounds


def extract_game_metrics(
    events: EventStream,
    trajectories: TrajectoryStream,
    manifest: RunManifest | None = None,
) -> GameMetrics:
    """Extract one game's `GameMetrics` from its event stream and trajectory sidecar.

    Outcomes (winner, deaths) come from the engine event log; per-seat telemetry
    from the trajectory sidecar; per-seat model identity from the manifest when
    present, else inferred from the seat's first LM call.
    """
    winner, rounds = _winner_and_rounds(events)
    dead = _dead_seats(events)
    illegal_by_seat = _illegal_by_seat(events)
    manifest_models = dict(manifest.models) if manifest is not None else None

    trajs_by_caller: dict[str, list[Trajectory]] = {}
    for traj in trajectories.trajectories:
        trajs_by_caller.setdefault(traj.caller, []).append(traj)

    seats: list[SeatMetrics] = []
    tool_usage: Counter[str] = Counter()
    for name, role in events.header.players:
        seat_trajs = trajs_by_caller.get(name, [])

        prompt_tokens = 0
        completion_tokens = 0
        lm_call_count = 0
        costs: list[float] = []
        inferred_model: str | None = None
        for traj in seat_trajs:
            for call in traj.lm_calls:
                prompt_tokens += call.prompt_tokens
                completion_tokens += call.completion_tokens
                lm_call_count += 1
                if call.cost_usd is not None:
                    costs.append(call.cost_usd)
                if inferred_model is None:
                    inferred_model = call.model

        tool_calls = 0
        for traj in seat_trajs:
            for step in traj.react_trajectory:
                if step.tool not in _GAME_ACTIONS:
                    continue
                tool_calls += 1
                # tool_usage = successful commits; rejected/faulted attempts ("error:")
                # still count toward tool_calls (the rate denominator) but not usage.
                if not step.observation.startswith("error:"):
                    tool_usage[step.tool] += 1
        illegal_moves = illegal_by_seat.get(name, 0)

        if manifest_models is not None:
            model = manifest_models.get(name, _UNKNOWN_MODEL)
        else:
            model = inferred_model if inferred_model is not None else _UNKNOWN_MODEL

        seats.append(
            SeatMetrics(
                name=name,
                role=role,
                model=model,
                faction=faction_of(role).value,
                won=faction_of(role).value == winner,
                survived=name not in dead,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                lm_calls=lm_call_count,
                tool_calls=tool_calls,
                illegal_moves=illegal_moves,
                cost_usd=sum(costs) if costs else None,
            )
        )

    total_prompt_tokens = sum(s.prompt_tokens for s in seats)
    total_completion_tokens = sum(s.completion_tokens for s in seats)
    total_lm_calls = sum(s.lm_calls for s in seats)
    total_tool_calls = sum(s.tool_calls for s in seats)
    total_illegal_moves = sum(s.illegal_moves for s in seats)
    seat_costs = [s.cost_usd for s in seats if s.cost_usd is not None]
    exiles_total, exiles_correct = _exile_accuracy(events)

    return GameMetrics(
        game_id=events.header.game_id,
        seed=events.header.seed,
        winner=winner,
        rounds=rounds,
        n_players=len(events.header.players),
        seats=tuple(seats),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_lm_calls=total_lm_calls,
        total_tool_calls=total_tool_calls,
        total_illegal_moves=total_illegal_moves,
        illegal_move_rate=total_illegal_moves / total_tool_calls if total_tool_calls else 0.0,
        total_cost_usd=sum(seat_costs) if seat_costs else None,
        tool_usage=tuple(sorted(tool_usage.items())),
        exiles_total=exiles_total,
        exiles_correct=exiles_correct,
        exile_accuracy=exiles_correct / exiles_total if exiles_total else None,
    )


def aggregate_metrics(games: Sequence[GameMetrics]) -> AggregateMetrics:
    """Roll up per-game metrics over seat-games, bucketed by model then role.

    Granularity is seat-games: each seat a model occupied in each game is one
    data point. `win_rate` guards against an empty bucket; model and role
    breakdowns are sorted for deterministic output.
    """
    model_games: Counter[str] = Counter()
    model_wins: Counter[str] = Counter()
    model_prompt: Counter[str] = Counter()
    model_completion: Counter[str] = Counter()
    model_illegal: Counter[str] = Counter()
    model_tool_calls: Counter[str] = Counter()
    role_games: dict[str, Counter[str]] = {}
    role_wins: dict[str, Counter[str]] = {}

    for game in games:
        for seat in game.seats:
            model_games[seat.model] += 1
            model_wins[seat.model] += 1 if seat.won else 0
            model_prompt[seat.model] += seat.prompt_tokens
            model_completion[seat.model] += seat.completion_tokens
            model_illegal[seat.model] += seat.illegal_moves
            model_tool_calls[seat.model] += seat.tool_calls
            role_games.setdefault(seat.model, Counter())[seat.role] += 1
            role_wins.setdefault(seat.model, Counter())[seat.role] += 1 if seat.won else 0

    models: list[ModelStats] = []
    for model in sorted(model_games):
        games_n = model_games[model]
        wins_n = model_wins[model]
        roles = tuple(
            RoleStats(
                role=role,
                games=role_games[model][role],
                wins=role_wins[model][role],
                win_rate=role_wins[model][role] / role_games[model][role] if role_games[model][role] else 0.0,
            )
            for role in sorted(role_games[model])
        )
        models.append(
            ModelStats(
                model=model,
                games=games_n,
                wins=wins_n,
                win_rate=wins_n / games_n if games_n else 0.0,
                prompt_tokens=model_prompt[model],
                completion_tokens=model_completion[model],
                illegal_moves=model_illegal[model],
                tool_calls=model_tool_calls[model],
                roles=roles,
            )
        )

    n_games = len(games)
    mean_game_length = sum(g.rounds for g in games) / n_games if n_games else 0.0
    # Equal weight per game (mean of per-game rates), NOT a pooled
    # total_illegal / total_tool_calls. A pooled rate is derivable per model from
    # ModelStats.illegal_moves / ModelStats.tool_calls when T26+ rating needs it.
    mean_illegal_move_rate = sum(g.illegal_move_rate for g in games) / n_games if n_games else 0.0

    return AggregateMetrics(
        n_games=n_games,
        mean_game_length=mean_game_length,
        mean_illegal_move_rate=mean_illegal_move_rate,
        models=tuple(models),
    )


def _unique_faction_model(seats: Sequence[SeatMetrics], faction_value: str) -> str | None:
    """Return the single model staffing a faction, or None when it is mixed/absent.

    Used to attribute a game-level faction outcome to a model: only meaningful when
    one model holds the whole faction (so a per-model record is unambiguous). The
    `_UNKNOWN_MODEL` sentinel is not a real model, so a faction staffed by it is
    treated as unattributable (None) rather than crediting a bogus "unknown" row.
    """
    models = {s.model for s in seats if s.faction == faction_value}
    if len(models) != 1:
        return None
    (model,) = models
    return None if model == _UNKNOWN_MODEL else model


def deceiver_detector_split(games: Sequence[GameMetrics]) -> DeceiverDetectorReport:
    """Split games into the deceiver (wolf win rate) and detector (exile accuracy) axes.

    WEREWOLF_DESIGN.md §10's first-class metric. The deceiver axis is a game-level
    outcome (did the wolves win) attributed to the unique werewolf-faction model;
    the detector axis pools village exile decisions attributed to the unique
    village-faction model. Population `exile_accuracy` POOLS across decisions
    (Σcorrect / Σtotal), deliberately unlike `aggregate_metrics.mean_illegal_move_rate`
    (mean of per-game rates): each exile is the unit of detection, so a multi-exile
    game must outweigh a single-exile game. Output models are sorted by name.
    """
    wolf_games: Counter[str] = Counter()
    wolf_wins: Counter[str] = Counter()
    village_exiles: Counter[str] = Counter()
    village_correct: Counter[str] = Counter()

    wolf_value = Faction.WEREWOLVES.value
    village_value = Faction.VILLAGERS.value
    population_wolf_wins = 0
    for game in games:
        if game.winner == wolf_value:
            population_wolf_wins += 1
        wolf_model = _unique_faction_model(game.seats, wolf_value)
        if wolf_model is not None:
            wolf_games[wolf_model] += 1
            if game.winner == wolf_value:
                wolf_wins[wolf_model] += 1
        village_model = _unique_faction_model(game.seats, village_value)
        if village_model is not None:
            village_exiles[village_model] += game.exiles_total
            village_correct[village_model] += game.exiles_correct

    n_games = len(games)
    total_exiles = sum(g.exiles_total for g in games)
    total_correct_exiles = sum(g.exiles_correct for g in games)

    model_names = sorted(set(wolf_games) | set(village_exiles))
    models = tuple(
        FactionSplit(
            model=model,
            wolf_games=wolf_games[model],
            wolf_wins=wolf_wins[model],
            wolf_win_rate=wolf_wins[model] / wolf_games[model] if wolf_games[model] else 0.0,
            village_exiles=village_exiles[model],
            village_correct_exiles=village_correct[model],
            exile_accuracy=village_correct[model] / village_exiles[model] if village_exiles[model] else None,
        )
        for model in model_names
    )

    return DeceiverDetectorReport(
        n_games=n_games,
        wolf_win_rate=population_wolf_wins / n_games if n_games else 0.0,
        exile_accuracy=total_correct_exiles / total_exiles if total_exiles else None,
        total_exiles=total_exiles,
        total_correct_exiles=total_correct_exiles,
        models=models,
    )


def extract_run_dir(path: Path) -> GameMetrics:
    """Extract `GameMetrics` from a run directory's `events`/`trajectories`/`manifest`.

    Reads `events.jsonl` and `trajectories.jsonl`; reads `manifest.json` when it
    exists (else `manifest=None`, falling back to trajectory-inferred models).
    """
    events = read_events(path / "events.jsonl")
    trajectories = read_trajectories(path / "trajectories.jsonl")
    manifest_path = path / "manifest.json"
    manifest = read_manifest(manifest_path) if manifest_path.exists() else None
    return extract_game_metrics(events, trajectories, manifest)
