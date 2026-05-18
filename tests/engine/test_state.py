"""Tests for the engine game-state model (T03).

These encode benchmark invariant #1: the engine is the single source of truth,
and every game position is an immutable snapshot. T04 appends one frozen
snapshot per event so games stay replayable — a mutable `GameState` could be
retroactively edited, silently corrupting the recorded stream. If the state
model ever becomes mutable, or derivation stops being pure, these fail loudly.
"""

from dataclasses import FrozenInstanceError

import pytest

from social_deduction_bench.engine import GameState, Phase

# A canonical 4-player setup reused across tests: (name, role) pairs.
PLAYERS = [("Alice", "villager"), ("Bob", "werewolf"), ("Cara", "seer"), ("Dan", "doctor")]


def test_initial_state_has_all_players_alive_round_one_night() -> None:
    """`initial()` pins the canonical start contract.

    Every game must begin from the same position — all players alive, round 1,
    NIGHT. The first night is round 1 because the WEREWOLF_DESIGN.md §4 loop
    increments the round before the first night.
    """
    state = GameState.initial(PLAYERS)

    assert state.round == 1
    assert state.phase is Phase.NIGHT
    assert [p.name for p in state.players] == ["Alice", "Bob", "Cara", "Dan"]
    assert [p.role for p in state.players] == ["villager", "werewolf", "seer", "doctor"]
    assert all(p.alive for p in state.players)


def test_initial_state_rejects_duplicate_player_names() -> None:
    """Duplicate names must be rejected at construction.

    `name` is the lookup key and the observation-routing address (T05). A
    duplicate makes `player()` ambiguous and would misroute private events to
    the wrong agent — a hidden-state-leak vector, the Critical class.
    """
    with pytest.raises(ValueError, match="duplicate"):
        GameState.initial([("Alice", "villager"), ("Alice", "werewolf")])


def test_initial_state_accepts_empty_roster() -> None:
    """The engine state model does not enforce a player count.

    `GameState` is the game-agnostic core; player-count rules (the 7-player
    Werewolf default) are a game-config concern (M2), not an engine invariant.
    An empty roster is a structurally valid snapshot.
    """
    state = GameState.initial([])

    assert state.players == ()
    assert state.alive_names() == ()
    assert state.round == 1
    assert state.phase is Phase.NIGHT


def test_game_state_is_frozen() -> None:
    """`GameState` fields cannot be reassigned in place.

    A mutable snapshot could be edited after T04 records it, breaking the
    append-only stream's replayability guarantee.
    """
    state = GameState.initial(PLAYERS)
    with pytest.raises(FrozenInstanceError):
        state.round = 5  # type: ignore[misc]  # assigning to a frozen field is the point
    with pytest.raises(FrozenInstanceError):
        state.phase = Phase.DAY  # type: ignore[misc]


def test_player_state_is_frozen() -> None:
    """`PlayerState.alive` cannot be flipped in place — same guarantee per player."""
    player = GameState.initial(PLAYERS).players[0]
    with pytest.raises(FrozenInstanceError):
        player.alive = False  # type: ignore[misc]


def test_players_container_is_immutable() -> None:
    """`players` must be a tuple, not a list.

    A frozen dataclass around a mutable list still leaks: callers could append
    or kill players in place. Immutability must reach the container.
    """
    assert isinstance(GameState.initial(PLAYERS).players, tuple)


def test_with_player_killed_returns_new_state_original_unchanged() -> None:
    """Killing a player derives a new state; the original snapshot is untouched.

    A death in a later round must not corrupt an earlier snapshot — derivation
    is the only legal way to evolve frozen state.
    """
    before = GameState.initial(PLAYERS)
    after = before.with_player_killed("Bob")

    assert after is not before
    assert after.player("Bob").alive is False
    assert before.player("Bob").alive is True


def test_killed_is_idempotent() -> None:
    """Killing an already-dead player must not raise or change anything else.

    Kill resolution (T11) and tie-breaks may double-apply a death; doing so
    must be a safe no-op.
    """
    state = GameState.initial(PLAYERS).with_player_killed("Bob")
    twice = state.with_player_killed("Bob")

    # Full structural equality — a buggy rebuild that touched another player's
    # fields would slip past an alive-names-only check.
    assert twice == state


def test_with_phase_and_advanced_round_do_not_mutate_original() -> None:
    """Phase and round derivation are pure — originals feeding T06 stay intact."""
    before = GameState.initial(PLAYERS)

    assert before.with_phase(Phase.DAY).phase is Phase.DAY
    assert before.phase is Phase.NIGHT

    assert before.advanced_round().round == 2
    assert before.round == 1


def test_with_player_killed_only_affects_target() -> None:
    """Killing one player must leave every other player's `alive` flag intact.

    Guards against a buggy rebuild that resets other players when reconstructing
    the tuple.
    """
    state = GameState.initial(PLAYERS).with_player_killed("Bob")

    assert state.player("Bob").alive is False
    assert state.player("Alice").alive is True
    assert state.player("Cara").alive is True
    assert state.player("Dan").alive is True


def test_with_player_killed_unknown_name_raises() -> None:
    """Killing an unknown name must raise, not silently no-op.

    Invariant #3: illegal moves are rejected, never silently applied. If this
    guard regressed to a no-op, an illegal kill (e.g. a dead or non-existent
    target) would be swallowed instead of surfaced as an error.
    """
    with pytest.raises(KeyError):
        GameState.initial(PLAYERS).with_player_killed("Nobody")


def test_player_lookup_returns_correct_player() -> None:
    """`player()` resolves a name to the exact `PlayerState`.

    Observation routing (T05) and tool validation (T07) address players by
    name; lookup must be exact.
    """
    cara = GameState.initial(PLAYERS).player("Cara")

    assert cara.name == "Cara"
    assert cara.role == "seer"


def test_player_lookup_unknown_name_raises() -> None:
    """An unknown name must raise, not return `None`.

    Fail loud (CLAUDE.md rule 11): a silent `None` would let an illegal tool
    target (T07) slip through as a no-op instead of an error observation.
    """
    with pytest.raises(KeyError):
        GameState.initial(PLAYERS).player("Nobody")


def test_alive_players_excludes_dead() -> None:
    """`alive_players()` returns only the living.

    Night/day acting sets and win checks (M2) operate on the living only; a
    stale dead player in the set would corrupt vote counts.
    """
    state = GameState.initial(PLAYERS).with_player_killed("Bob")

    alive = state.alive_players()
    assert {p.name for p in alive} == {"Alice", "Cara", "Dan"}
    assert all(p.alive for p in alive)


def test_alive_names_matches_alive_players() -> None:
    """`alive_names()` — the cheap form for routing — agrees with `alive_players()`."""
    state = GameState.initial(PLAYERS).with_player_killed("Dan")

    assert state.alive_names() == tuple(p.name for p in state.alive_players())
    assert state.is_alive("Alice") is True
    assert state.is_alive("Dan") is False


def test_equal_states_compare_equal() -> None:
    """Two states built identically compare equal, including the nested players.

    T08's determinism harness compares two runs structurally; frozen-dataclass
    equality must hold through the player tuple for that comparison to work.
    """
    a = GameState.initial(PLAYERS).with_player_killed("Bob").advanced_round()
    b = GameState.initial(PLAYERS).with_player_killed("Bob").advanced_round()

    assert a == b
    assert a != GameState.initial(PLAYERS)
