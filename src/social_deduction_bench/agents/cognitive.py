"""Game-agnostic cognitive tools: read your memory, write notes, manage beliefs and your plan."""

from social_deduction_bench.agents.memory import Confidence, GameMemory
from social_deduction_bench.engine import GameState


def _validate_caller(state: GameState, caller: str) -> None:
    """Raise `KeyError` if `caller` is not a real player."""
    state.player(caller)


def recall(
    state: GameState,
    memory: GameMemory,
    caller: str,
    last_n_rounds: int | None = None,
) -> str:
    """Read your own memory: events you have observed and notes you have written, in chronological order.

    `last_n_rounds` keeps only the most recent N rounds; omit for the full history.
    """
    _validate_caller(state, caller)
    return memory.recall(last_n_rounds=last_n_rounds)


def remember(state: GameState, memory: GameMemory, caller: str, note: str) -> str:
    """Write a free-text note to your own memory; you can read it back later with `recall`.

    `note` is the text to record; it cannot be empty.
    """
    _validate_caller(state, caller)
    memory.remember(note, round_=state.round)
    return f"ok: noted at R{state.round}"


def get_beliefs(state: GameState, memory: GameMemory, caller: str) -> str:
    """Read your structured suspicion table: one row per player you have an opinion on."""
    _validate_caller(state, caller)
    if not memory.beliefs:
        return "beliefs: (none)"
    lines = ["beliefs:"]
    for player in sorted(memory.beliefs):
        belief = memory.beliefs[player]
        evidence = belief.evidence or "(no evidence)"
        lines.append(f"  {player}: {belief.guess} ({belief.confidence}) -- {evidence}")
    return "\n".join(lines)


def set_belief(
    state: GameState,
    memory: GameMemory,
    caller: str,
    player: str,
    guess: str,
    confidence: Confidence,
    evidence: str,
) -> str:
    """Record or update your suspicion about one player; overwrites any prior row for the same player.

    `player` is the player you are suspecting. `guess` is what you think they are.
    `evidence` is a short free-text justification you can re-read later.
    """
    _validate_caller(state, caller)
    memory.set_belief(player, guess, confidence, evidence)
    return f"ok: belief set for {player}"


def get_plan(state: GameState, memory: GameMemory, caller: str) -> str:
    """Read your persistent strategic plan."""
    _validate_caller(state, caller)
    return f"plan: {memory.plan}" if memory.plan else "plan: (none)"


def set_plan(state: GameState, memory: GameMemory, caller: str, text: str) -> str:
    """Overwrite your persistent strategic plan; it survives across decision points within this game.

    `text` is the plan to record; it cannot be empty.
    """
    _validate_caller(state, caller)
    memory.set_plan(text)
    return "ok: plan set"
