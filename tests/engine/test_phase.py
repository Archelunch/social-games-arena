"""Tests for the engine phase state machine (T06).

These encode benchmark invariant #1 (the engine — not a game — owns phase and
round progression) and invariant #4 (the transition is a pure derivation over
`GameState`: no RNG, no clock, so the same input always yields the same next
position and the stream stays replayable). The terminal-detection hook is the
game-agnostic seam a game's win condition plugs into; the engine forms no
opinion on what ends a game.
"""

from social_deduction_bench.engine import (
    GameState,
    Phase,
    advance_phase,
    is_terminal,
)

# A canonical 4-player setup reused across tests: (name, role) pairs.
PLAYERS = [("Alice", "villager"), ("Bob", "werewolf"), ("Cara", "seer"), ("Dan", "doctor")]


def test_night_advances_to_day_same_round() -> None:
    """NIGHT -> DAY stays in the same round.

    A round's day follows its own night — the WEREWOLF_DESIGN.md loop resolves
    night then day within one round. Advancing the round here would mislabel
    every recorded day event.
    """
    night = GameState.initial(PLAYERS)
    day = advance_phase(night)

    assert day.phase is Phase.DAY
    assert day.round == night.round


def test_day_advances_to_night_next_round() -> None:
    """DAY -> NIGHT opens the next round.

    The DAY->NIGHT edge is the *only* place a round advances. If it stopped
    incrementing, every later round would collide on one round number.
    """
    day = GameState.initial(PLAYERS).with_phase(Phase.DAY)
    night = advance_phase(day)

    assert night.phase is Phase.NIGHT
    assert night.round == day.round + 1


def test_full_cycle_from_initial() -> None:
    """A full cycle from `initial()` reproduces the round-1 convention.

    The first night is round 1 (WEREWOLF_DESIGN.md §4 increments the round
    before the first night). This test pins that convention end to end: if
    `initial()` is ever changed back to round 0, it fails loudly.
    """
    night1 = GameState.initial(PLAYERS)
    assert (night1.round, night1.phase) == (1, Phase.NIGHT)

    day1 = advance_phase(night1)
    assert (day1.round, day1.phase) == (1, Phase.DAY)

    night2 = advance_phase(day1)
    assert (night2.round, night2.phase) == (2, Phase.NIGHT)

    day2 = advance_phase(night2)
    assert (day2.round, day2.phase) == (2, Phase.DAY)


def test_advance_phase_does_not_mutate_input() -> None:
    """The transition is a pure derivation — the input snapshot is untouched.

    Invariant #4: an earlier recorded position (T04 stream) must stay intact
    after the game moves on, or replay is corrupted.
    """
    before = GameState.initial(PLAYERS)
    after = advance_phase(before)

    assert after is not before
    assert before.phase is Phase.NIGHT
    assert before.round == 1


def test_advance_phase_preserves_players() -> None:
    """The transition touches only phase/round, never the roster.

    Every player's name, role, and alive flag must survive byte-identical; a
    buggy rebuild that reconstructed the player tuple could silently flip one.
    """
    before = GameState.initial(PLAYERS)
    after = advance_phase(before)

    assert after.players == before.players


def test_advance_phase_carries_player_deaths() -> None:
    """A death survives the phase transition — the dead are not resurrected.

    Night/day resolution (T11/T12) kills players; that outcome must persist
    across the following phase change, or a killed werewolf would reappear.
    """
    state = GameState.initial(PLAYERS).with_player_killed("Bob")
    after = advance_phase(state)

    assert after.player("Bob").alive is False
    assert after.player("Alice").alive is True


def test_repeated_advance_is_deterministic() -> None:
    """Same input -> identical output, and two independent runs agree.

    Invariant #4: the transition is deterministic. T08's determinism harness
    relies on this — a four-step cycle from two separate `initial()` states
    must compare structurally equal at every step. The cycle is also asserted
    to visit the expected *distinct* positions, so an `advance_phase` that
    degenerated to identity would fail here, not pass vacuously.
    """
    a = GameState.initial(PLAYERS)
    b = GameState.initial(PLAYERS)
    visited: list[tuple[int, Phase]] = []

    for _ in range(4):
        a = advance_phase(a)
        b = advance_phase(b)
        assert a == b
        visited.append((a.round, a.phase))

    assert visited == [
        (1, Phase.DAY),
        (2, Phase.NIGHT),
        (2, Phase.DAY),
        (3, Phase.NIGHT),
    ]


def test_is_terminal_delegates_to_check() -> None:
    """`is_terminal` returns exactly what the supplied check returns.

    Invariant #1: the engine owns no win condition — it only relays the game's
    verdict. A stub returning True/False must pass straight through.
    """
    state = GameState.initial(PLAYERS)

    assert is_terminal(state, lambda _state: True) is True
    assert is_terminal(state, lambda _state: False) is False


def test_is_terminal_passes_state_unchanged_to_check() -> None:
    """The game's win check receives the actual `GameState`, by identity.

    The seam must hand the real position to the game — not a copy, not `None`
    — or a win condition would be evaluated against the wrong state.
    """
    state = GameState.initial(PLAYERS)
    seen: list[GameState] = []

    def recording_check(s: GameState) -> bool:
        seen.append(s)
        return False

    is_terminal(state, recording_check)

    assert len(seen) == 1
    assert seen[0] is state
