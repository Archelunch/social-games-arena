"""Game-agnostic cognitive tools (WEREWOLF_DESIGN.md §6.2).

Cognitive tools are the read-only / private-scratchpad calls a ReAct
agent makes between game actions. These six are pure delegations to
`GameMemory` and carry no game-specific knowledge; ONUW and Secret
Hitler reuse them unchanged. The two werewolf-specific readers
(`get_public_state`, `get_private_info`) live in `games/werewolf/`.

All six share the signature `(state, memory, caller, ...)`:

  - `state` is the engine snapshot; we use `state.player(caller)` to
    fail loud (KeyError) on an unrecognized name. A dead caller is
    allowed — they read as a spectator.
  - `memory` is the caller's own per-agent `GameMemory`.
  - The return type is always `str` so a DSPy ReAct loop can surface
    every tool call as an Observation without per-tool plumbing.

Writers (`remember`, `set_belief`, `set_plan`) propagate the
`ValueError`s `GameMemory` / `Note` / `Belief` raise on invalid input
— fail loud at the agents/ boundary; the cognitive layer does not
silently catch and substitute an error string.

Sorting / sentinels: `get_beliefs` sorts rows by subject name (a
deterministic render is part of invariant #4 at the read layer);
`get_plan` / `get_beliefs` use the literal `(none)` for empty
sections so the LLM and the test suite can distinguish "no data"
from a missing line.
"""

from social_deduction_bench.agents.memory import Confidence, GameMemory
from social_deduction_bench.engine import GameState


def _validate_caller(state: GameState, caller: str) -> None:
    """Raise `KeyError` if `caller` is not a real player.

    Delegates to `state.player` so the fail-loud shape matches every
    other engine read (T05/T17). Dead callers pass — the cognitive
    layer never gates on alive.
    """
    state.player(caller)


def recall(
    state: GameState,
    memory: GameMemory,
    caller: str,
    last_n_rounds: int | None = None,
) -> str:
    """Render the caller's Tier 0 memory: events + notes, optionally last-N-rounds filtered.

    Pure forward to `GameMemory.recall`. The `query` argument
    WEREWOLF_DESIGN.md §6.2 mentions lands with Tier 1 (when the
    storage layer grows the filter); adding a no-op now would be the
    speculative parameter T19 already rejected.
    """
    _validate_caller(state, caller)
    return memory.recall(last_n_rounds=last_n_rounds)


def remember(state: GameState, memory: GameMemory, caller: str, note: str) -> str:
    """Write a free-text note to the caller's memory, tagged with `state.round`.

    The cognitive layer is where the engine clock binds — `GameMemory`
    is clock-agnostic (T19) so the LLM does not need to thread the
    round into its tool argument.
    """
    _validate_caller(state, caller)
    memory.remember(note, round_=state.round)
    return f"ok: noted at R{state.round}"


def get_beliefs(state: GameState, memory: GameMemory, caller: str) -> str:
    """Render the structured suspicion table, one row per subject, sorted by name.

    Empty table → `"beliefs: (none)"`. Non-empty → header `"beliefs:"`
    plus indented rows `"  {player}: {guess} ({confidence}) -- {evidence}"`,
    with empty `evidence` rendered as `(no evidence)` so a row's shape
    stays parseable.
    """
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
    """Overwrite the suspicion row for `player`.

    Validation lives in `Belief.__post_init__` (blank player/guess,
    confidence outside `{low, medium, high}`); we let `ValueError`
    propagate.
    """
    _validate_caller(state, caller)
    memory.set_belief(player, guess, confidence, evidence)
    return f"ok: belief set for {player}"


def get_plan(state: GameState, memory: GameMemory, caller: str) -> str:
    """Render the caller's persistent strategic plan.

    Default (`memory.plan == ""`) renders as `"plan: (none)"`; a set
    plan renders as `"plan: <text>"`.
    """
    _validate_caller(state, caller)
    return f"plan: {memory.plan}" if memory.plan else "plan: (none)"


def set_plan(state: GameState, memory: GameMemory, caller: str, text: str) -> str:
    """Overwrite the caller's persistent strategic plan.

    Blank text is rejected by `GameMemory.set_plan` (WEREWOLF_DESIGN.md
    §9 frames the plan as meaningful strategic text); we let
    `ValueError` propagate.
    """
    _validate_caller(state, caller)
    memory.set_plan(text)
    return "ok: plan set"
