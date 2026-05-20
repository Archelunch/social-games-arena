"""Werewolf-specific cognitive tools (WEREWOLF_DESIGN.md §6.2).

Two readers carry Werewolf rules:

  - `get_public_state` renders the public game position: round, phase,
    alive list, dead list. It deliberately reads only `name` and
    `alive` from `state.players` — every player's `role` is omitted,
    which is the read-side enforcement of invariant #2 ("agents never
    read hidden state") at this layer. A regression that pasted
    `state.players` into the rendering would publish every role.

  - `get_private_info` renders the caller's own private view: own
    role/faction, plus a role-specific section. Werewolves see the
    list of other werewolves (the pack knows each other,
    WEREWOLF_DESIGN.md §5). The seer sees their own inspection
    history, derived from the `SEER_INSPECT` private events in their
    memory — already routed by `observations_for` (T05), so the
    cognitive layer does not re-filter for recipients. Villagers and
    the doctor get only the self-line; no team or history section
    leaks adjacent role names.

The four other game-agnostic cognitive tools (`recall`, `remember`,
`get_beliefs`, `set_belief`, `get_plan`, `set_plan`) live in
`agents/cognitive.py` — ONUW and Secret Hitler reuse them unchanged.
"""

from social_deduction_bench.agents import GameMemory
from social_deduction_bench.engine import GameState
from social_deduction_bench.games.werewolf.events import SEER_INSPECT
from social_deduction_bench.games.werewolf.roles import Role, faction_of


def get_public_state(state: GameState, memory: GameMemory, caller: str) -> str:
    """Render the public position: round, phase, alive list, dead list.

    Roles are never surfaced — this is the read-side guard for
    invariant #2. `alive_names` preserves the engine's player order;
    the dead list uses the same order over `state.players`.
    """
    state.player(caller)  # fail loud on unknown caller (mirrors `state.player`)
    alive_names = state.alive_names()
    dead_names = tuple(p.name for p in state.players if not p.alive)
    alive_str = ", ".join(alive_names) if alive_names else "(none)"
    dead_str = ", ".join(dead_names) if dead_names else "(none)"
    return f"round={state.round} phase={state.phase}\nalive: {alive_str}\ndead: {dead_str}"


def get_private_info(state: GameState, memory: GameMemory, caller: str) -> str:
    """Render the caller's own private view.

    Layout:

      ``you={caller} role={role} faction={faction}``

    plus, if the caller is a werewolf:

      ``partners: {other living werewolves, sorted by name, or '(none)'}``
      — dead pack members are excluded so the LLM coordinates only
      with who is still around to act this round.

    plus, if the caller is the seer:

      ``inspections: R{n} {target}={faction}, ...``   (memory order)

    Villagers and the doctor get the self-line only. The seer's
    history is filtered out of `memory.events` — `observations_for`
    (T05) has already restricted the events to ones routed to this
    caller, so the filter here is by event type, not by recipient.
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
