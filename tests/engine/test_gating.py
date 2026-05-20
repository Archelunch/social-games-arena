"""Tests for engine-level tool gating (T17).

`available_tools(state, caller, registry)` is the dual of `validate_tool_call`:
the latter rejects an illegal call, the former lists which calls are legal
*right now*. The DSPy ReAct agent (T21) consumes this menu to choose its next
move; the menu is also the defense-in-depth layer that hides illegal options
before the agent ever tries them.

These tests encode the *intent* — why each behavior matters for the benchmark
invariants (CLAUDE.md rule 8). The output is sorted because invariant #4
requires byte-identical replay across runs; dead callers get `()` because
invariant #1 says only living, engine-known players act; an unknown caller
fails loud because silently degrading would mask a routing bug in the agent
integration layer.
"""

import dataclasses

import pytest

from social_deduction_bench.engine import (
    GameState,
    Phase,
    ToolRequirement,
    available_tools,
)

# Canonical 4-player roster reused across tests: (name, role). Round 1, NIGHT
# by GameState.initial; tests that need DAY apply .with_phase.
PLAYERS = [("Alice", "villager"), ("Bob", "werewolf"), ("Cara", "seer"), ("Dan", "doctor")]


def _state() -> GameState:
    """A fresh round-1 NIGHT state with the canonical roster."""
    return GameState.initial(PLAYERS)


# A small synthetic registry covering every (phase, role) combination we want
# to assert on. Kept distinct from the Werewolf registry so the engine tests
# verify the *primitive*, not a particular game's catalog.
_NIGHT_SEER = ToolRequirement(phase=Phase.NIGHT, role="seer", requires_target=True)
_DAY_ANY = ToolRequirement(phase=Phase.DAY)
_NIGHT_ANY = ToolRequirement(phase=Phase.NIGHT)
_ANY_PHASE_VILLAGER = ToolRequirement(role="villager")
_UNCONSTRAINED = ToolRequirement()


def test_returns_tools_matching_role_and_phase() -> None:
    """A fully-constrained tool is exposed only when role *and* phase match.

    The happy path for a night-ability tool — if this fails, the agent never
    sees its primary move.
    """
    registry = {"seer_inspect": _NIGHT_SEER}

    assert available_tools(_state(), "Cara", registry) == ("seer_inspect",)


def test_omits_tools_with_wrong_phase() -> None:
    """A DAY-only tool is hidden during NIGHT for an otherwise-eligible caller.

    Phase is the primary axis driving the §4 game loop. A phase leak would let
    the agent surface an out-of-phase tool even before `validate_tool_call`
    triggers the rejection at call time.
    """
    registry = {"submit_exile_vote": _DAY_ANY}

    assert available_tools(_state(), "Alice", registry) == ()


def test_omits_tools_with_wrong_role() -> None:
    """A role-locked tool is hidden from callers of the wrong role.

    Invariant #2 — agents only see what their role allows. An over-broad menu
    is itself a hidden-state leak: showing `seer_inspect` to a werewolf would
    reveal that the Seer role exists at all (a real game keeps roles hidden
    from inspection).
    """
    registry = {"seer_inspect": _NIGHT_SEER}

    assert available_tools(_state(), "Bob", registry) == ()


def test_phase_only_requirement_matches_any_role() -> None:
    """A `role=None` requirement is exposed to every alive player in the right phase.

    Pins the "no constraint" semantics of `None` on the role gate; matches the
    Werewolf day vote tools (`submit_bid`, `speak`, `submit_exile_vote`).
    """
    day = _state().with_phase(Phase.DAY)
    registry = {"submit_bid": _DAY_ANY}

    for name, _ in PLAYERS:
        assert available_tools(day, name, registry) == ("submit_bid",)


def test_role_only_requirement_matches_any_phase() -> None:
    """A `phase=None` requirement is exposed in NIGHT *and* DAY.

    Pins the dual `None` semantic on the phase gate. No current Werewolf tool
    uses it, but a future cross-phase tool must work without a code change.
    """
    night = _state()
    day = night.with_phase(Phase.DAY)
    registry = {"cross_phase_tool": _ANY_PHASE_VILLAGER}

    assert available_tools(night, "Alice", registry) == ("cross_phase_tool",)
    assert available_tools(day, "Alice", registry) == ("cross_phase_tool",)


def test_unconstrained_requirement_is_always_available_to_alive_players() -> None:
    """A `ToolRequirement()` with no gates is exposed in any state.

    The filter must be a *positive* match (gate is None → satisfied), not a
    "default to forbidden" — otherwise role/phase = None would lock the menu.
    """
    night = _state()
    day = night.with_phase(Phase.DAY)
    registry = {"free_tool": _UNCONSTRAINED}

    for state in (night, day):
        for name, _ in PLAYERS:
            assert available_tools(state, name, registry) == ("free_tool",)


def test_dead_caller_has_no_available_tools() -> None:
    """A dead player gets `()` regardless of role/phase.

    Invariant #1 — dead players never act. The menu must reflect that
    directly, not rely solely on the call-time gate; an agent integration
    layer reading the menu must see "nothing available", not a list of moves
    the engine will reject seconds later.
    """
    state = _state().with_player_killed("Cara")
    registry = {
        "seer_inspect": _NIGHT_SEER,
        "free_tool": _UNCONSTRAINED,
        "any_night": _NIGHT_ANY,
    }

    assert available_tools(state, "Cara", registry) == ()


def test_unknown_caller_raises_keyerror() -> None:
    """An unknown caller name raises `KeyError`, matching `state.player`.

    Fail loud — silently degrading to `()` would mask a routing bug (the
    agent integration layer named a non-existent player). Consistent with the
    rest of the engine's identity-lookup contract.
    """
    with pytest.raises(KeyError):
        available_tools(_state(), "Mallory", {"seer_inspect": _NIGHT_SEER})


def test_empty_registry_returns_empty_tuple() -> None:
    """A registry with no tools yields an empty menu.

    Boundary correctness — a future game may legitimately declare no tools in
    some phase, and the primitive must handle that without raising.
    """
    assert available_tools(_state(), "Alice", {}) == ()


def test_state_not_mutated() -> None:
    """The state is byte-identical before and after the call.

    Invariant #3 — `available_tools` is a pure read; it never mutates state.
    Whole-object equality (not just field-by-field), consistent with every
    other engine purity test.
    """
    state = _state()
    snapshot = dataclasses.replace(state)
    registry = {"seer_inspect": _NIGHT_SEER, "any_night": _NIGHT_ANY}

    available_tools(state, "Cara", registry)

    assert state == snapshot


def test_output_is_sorted_tuple() -> None:
    """The returned names are sorted ascending in a `tuple`.

    Invariant #4 — replay requires byte-identical output across runs. A `set`
    or insertion-ordered output would diverge across Python minor versions or
    dict-iteration changes; a sorted tuple is the only deterministic shape.
    """
    # Insertion order deliberately reversed so a "return dict order" bug fails.
    registry = {
        "zeta": _UNCONSTRAINED,
        "alpha": _UNCONSTRAINED,
        "mu": _UNCONSTRAINED,
    }

    result = available_tools(_state(), "Alice", registry)

    assert isinstance(result, tuple)
    assert result == ("alpha", "mu", "zeta")


def test_deterministic_across_independent_states() -> None:
    """Building the same state twice and querying both yields identical tuples.

    Determinism gate aligned with T08's harness philosophy: same input → same
    output, regardless of object identity. A future implementation that
    accidentally hashed object id would fail this.
    """
    registry = {
        "seer_inspect": _NIGHT_SEER,
        "free_tool": _UNCONSTRAINED,
    }

    a = available_tools(_state(), "Cara", registry)
    b = available_tools(_state(), "Cara", registry)

    assert a == b
