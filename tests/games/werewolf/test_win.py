"""Tests for Werewolf win-condition checks (T13).

`winner`/`is_game_over` decide when a game ends and who won (WEREWOLF_DESIGN.md
§2). These tests encode the rules and the benchmark invariants:

- villagers win when both werewolves are dead;
- werewolves win at parity — `#werewolves_alive >= #non_werewolves_alive`;
- the check ORDER matters: when the last werewolf dies, `0 >= 0` parity is also
  true, so the villager win must be tested first or the werewolves would be
  wrongly credited a win at the moment they are wiped out;
- #1 — `winner` is a pure read over `GameState`; it never mutates it;
- `is_game_over` is the `TerminalCheck` the engine's `is_terminal` seam (T06)
  consumes — the engine owns no win condition, it only relays this verdict.
"""

from social_deduction_bench.engine import GameState, is_terminal
from social_deduction_bench.games.werewolf.win import Winner, is_game_over, winner

ROSTER = (
    ("Wolf1", "werewolf"),
    ("Wolf2", "werewolf"),
    ("Seer", "seer"),
    ("Doc", "doctor"),
    ("Vil1", "villager"),
    ("Vil2", "villager"),
    ("Vil3", "villager"),
)


def _state_with_dead(*dead: str) -> GameState:
    """A fresh 7-player game with the named players killed."""
    state = GameState.initial(ROSTER)
    for name in dead:
        state = state.with_player_killed(name)
    return state


def test_villagers_win_when_both_werewolves_are_dead() -> None:
    """With zero werewolves alive, the villagers have won.

    The villagers' win condition (WEREWOLF_DESIGN.md §2): eliminate every
    werewolf. Five villagers still alive, no wolves -> a clean villager win.
    """
    state = _state_with_dead("Wolf1", "Wolf2")

    assert winner(state) is Winner.VILLAGERS


def test_werewolves_win_at_parity() -> None:
    """Werewolves win once they reach parity with the rest.

    2 werewolves vs 2 villagers: `2 >= 2` — the werewolves can no longer be
    out-voted, so the game is theirs (WEREWOLF_DESIGN.md §2).
    """
    state = _state_with_dead("Seer", "Doc", "Vil3")

    assert winner(state) is Winner.WEREWOLVES


def test_werewolves_win_when_they_outnumber_the_village() -> None:
    """Werewolves win when they strictly outnumber the rest.

    2 werewolves vs 1 villager: `2 >= 1`. Parity is a `>=`, so an outright
    majority is also a werewolf win.
    """
    state = _state_with_dead("Seer", "Doc", "Vil2", "Vil3")

    assert winner(state) is Winner.WEREWOLVES


def test_game_continues_while_villagers_outnumber_werewolves() -> None:
    """A full 7-player game is not yet terminal — `winner` is None.

    2 werewolves vs 5 villagers: the wolves are alive but below parity, so the
    game must continue. A premature winner here would end every game at setup.
    """
    state = GameState.initial(ROSTER)

    assert winner(state) is None


def test_game_continues_at_three_to_two_village_lead() -> None:
    """Mid-game, 2 werewolves vs 3 villagers is still not terminal.

    `2 >= 3` is false and a werewolf is still alive — neither side has won.
    Pins that the parity threshold is not crossed one step early.
    """
    state = _state_with_dead("Vil2", "Vil3")

    assert winner(state) is None


def test_last_werewolf_death_is_a_villager_win_not_a_parity_win() -> None:
    """When the last werewolf dies into an empty board, villagers win.

    With everyone dead, `#werewolves_alive (0) >= #non_werewolves_alive (0)` is
    true — the werewolf parity rule would fire. The villager check must run
    first, so a board with zero werewolves is always a villager win. This is
    the check-order correctness test.
    """
    state = _state_with_dead("Wolf1", "Wolf2", "Seer", "Doc", "Vil1", "Vil2", "Vil3")

    assert winner(state) is Winner.VILLAGERS


def test_is_game_over_is_true_at_a_terminal_position() -> None:
    """`is_game_over` is True exactly when `winner` names a side.

    The terminal hook the game loop branches on must agree with `winner`.
    """
    assert is_game_over(_state_with_dead("Wolf1", "Wolf2")) is True


def test_is_game_over_is_false_mid_game() -> None:
    """`is_game_over` is False while the game is still in progress.

    A full 7-player board is not terminal; the loop must keep running.
    """
    assert is_game_over(GameState.initial(ROSTER)) is False


def test_is_game_over_satisfies_the_engine_terminal_seam() -> None:
    """`is_game_over` plugs into the engine's `is_terminal` seam (T06).

    Invariant #1: the engine owns no win condition — it only relays the game's
    `TerminalCheck`. Passing `is_game_over` through `is_terminal` must return
    the same verdict, confirming the seam contract holds.
    """
    terminal = _state_with_dead("Wolf1", "Wolf2")
    ongoing = GameState.initial(ROSTER)

    assert is_terminal(terminal, is_game_over) is True
    assert is_terminal(ongoing, is_game_over) is False


def test_winner_does_not_mutate_the_state() -> None:
    """`winner` is a pure read — the input snapshot is untouched.

    Invariant #1: the win check inspects state, never derives a new one; a
    recorded position must survive the query byte-identical.
    """
    state = GameState.initial(ROSTER)
    before = GameState.initial(ROSTER)

    winner(state)

    assert state == before


def test_winner_string_values_are_stable() -> None:
    """`Winner` string values are pinned literals.

    The winning side is written into the public `game_over` event payload
    (T14) and into every transcript; a rename would break recorded replays and
    the rating pipeline that reads them.
    """
    assert Winner.WEREWOLVES.value == "werewolves"
    assert Winner.VILLAGERS.value == "villagers"
