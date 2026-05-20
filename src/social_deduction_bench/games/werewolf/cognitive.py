"""Werewolf-specific cognitive tools: read the public game state and your own private info."""

from social_deduction_bench.agents.memory import GameMemory
from social_deduction_bench.engine import GameState
from social_deduction_bench.games.werewolf.events import SEER_INSPECT
from social_deduction_bench.games.werewolf.roles import Role, faction_of


def get_public_state(state: GameState, memory: GameMemory, caller: str) -> str:
    """Read the public game position: current round, current phase, who is alive, and who is dead."""
    state.player(caller)
    alive_names = state.alive_names()
    dead_names = tuple(p.name for p in state.players if not p.alive)
    alive_str = ", ".join(alive_names) if alive_names else "(none)"
    dead_str = ", ".join(dead_names) if dead_names else "(none)"
    return f"round={state.round} phase={state.phase}\nalive: {alive_str}\ndead: {dead_str}"


def get_private_info(state: GameState, memory: GameMemory, caller: str) -> str:
    """Read your own private information: your role and faction, plus role-specific knowledge.

    Werewolves additionally see the names of their living fellow werewolves.
    The seer additionally sees their inspection history (which players they
    inspected on which round, and what faction came back).
    Villagers and the doctor see only their own role and faction.
    """
    me = state.player(caller)
    lines = [f"you={caller} role={me.role} faction={faction_of(me.role)}"]

    if me.role == Role.WEREWOLF.value:
        partners = sorted(
            p.name for p in state.players if p.role == Role.WEREWOLF.value and p.alive and p.name != caller
        )
        partners_str = ", ".join(partners) if partners else "(none)"
        lines.append(f"partners: {partners_str}")

    if me.role == Role.SEER.value:
        inspects = [
            f"R{event.round} {event.payload['target']}={event.payload['faction']}"
            for event in memory.events
            if event.type == SEER_INSPECT
        ]
        inspects_str = ", ".join(inspects) if inspects else "(none)"
        lines.append(f"inspections: {inspects_str}")

    return "\n".join(lines)
