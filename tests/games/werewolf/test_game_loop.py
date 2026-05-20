"""Integration test for the full Werewolf game loop (T14).

This is the M2 capstone: it wires T09-T13 together and drives a complete
7-player game from setup to a terminal state. It encodes the benchmark
invariants end to end:

- #4 — the whole game is deterministic and replayable: `assert_deterministic`
  (the T08 harness) confirms two runs of the same seeded scripted game produce
  byte-identical transcripts; a tied kill vote resolves differently under a
  different seed.
- #5 — the transcript is an append-only event stream that round-trips losslessly
  through JSONL.
- #2 — no private event ever ships with empty recipients, and a plain villager's
  observation view never contains a seer/doctor/pack-chat event.

Decisions are *scripted* — M2 has no agents. The `DecisionSource` Protocol is
the seam an M4 DSPy agent later implements; the loop itself does not change.
"""

import pytest

from social_deduction_bench.engine import Event, EventStream, GameState, assert_deterministic, observations_for
from social_deduction_bench.games.werewolf.config import PRIVATE_EVENT_TYPES
from social_deduction_bench.games.werewolf.day import DayActions
from social_deduction_bench.games.werewolf.events import GAME_OVER, KILL_RESOLVED
from social_deduction_bench.games.werewolf.loop import run_game
from social_deduction_bench.games.werewolf.night import NightActions
from social_deduction_bench.games.werewolf.scripted import ScriptedDecisions

ROSTER = (
    ("Wolf1", "werewolf"),
    ("Wolf2", "werewolf"),
    ("Seer", "seer"),
    ("Doc", "doctor"),
    ("Vil1", "villager"),
    ("Vil2", "villager"),
    ("Vil3", "villager"),
)


def _werewolf_win_script() -> ScriptedDecisions:
    """A scripted 7-player game the werewolves win at parity on round 2.

    Round 1 night: kill Vil1 (seer inspects Wolf1, doctor protects Vil2 — the
    protect misses, so Vil1 dies). Round 1 day: the village mis-exiles Vil2.
    Round 2 night: kill Vil3 — leaving 2 werewolves vs 2 villagers, a werewolf
    win. Each call returns a *fresh* source (its per-round cursors are state).
    """
    return ScriptedDecisions(
        nights=[
            NightActions(
                kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"},
                seer_inspect="Wolf1",
                doctor_protect="Vil2",
            ),
            NightActions(
                kill_votes={"Wolf1": "Vil3", "Wolf2": "Vil3"},
                seer_inspect="Wolf2",
                doctor_protect="Seer",
            ),
        ],
        days=[
            DayActions(
                exile_votes={
                    "Wolf1": "Vil2",
                    "Wolf2": "Vil2",
                    "Seer": "Vil2",
                    "Doc": "Vil2",
                    "Vil2": "Wolf1",
                    "Vil3": "Wolf1",
                }
            ),
        ],
    )


def _tied_first_night_script() -> ScriptedDecisions:
    """A scripted game whose round-1 night kill vote ties between Seer and Doc.

    The werewolves split 1-1, so the victim is decided by the engine seed. Days
    are all-abstain (no exile, no living-voter concern); the later nights kill
    fixed villagers so the game terminates as a werewolf win regardless of
    which of Seer/Doc the tie removed.
    """
    return ScriptedDecisions(
        nights=[
            NightActions(kill_votes={"Wolf1": "Seer", "Wolf2": "Doc"}),
            NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"}),
            NightActions(kill_votes={"Wolf1": "Vil2", "Wolf2": "Vil2"}),
        ],
        days=[DayActions(exile_votes={}), DayActions(exile_votes={})],
    )


def _villager_win_script() -> ScriptedDecisions:
    """A scripted game the villagers win by exiling both werewolves.

    Round 1: kill Vil1, then exile Wolf1. Round 2: the lone remaining werewolf
    kills Vil2, then the village exiles Wolf2 — leaving zero werewolves, a
    villager win. The game terminates *after a day exile*, exercising the
    loop's post-exile terminal check (the werewolf-win script ends after a
    night, so this is the complementary integration path).
    """
    return ScriptedDecisions(
        nights=[
            NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"}),
            NightActions(kill_votes={"Wolf2": "Vil2"}),
        ],
        days=[
            DayActions(
                exile_votes={
                    "Seer": "Wolf1",
                    "Doc": "Wolf1",
                    "Vil2": "Wolf1",
                    "Vil3": "Wolf1",
                    "Wolf1": "Seer",
                    "Wolf2": "Seer",
                }
            ),
            DayActions(exile_votes={"Seer": "Wolf2", "Doc": "Wolf2", "Vil3": "Wolf2", "Wolf2": "Seer"}),
        ],
    )


class _StallingDecisions:
    """A `DecisionSource` that never lets the game end.

    The doctor always protects the werewolves' kill target, so the kill is
    always suppressed; days are all-abstain. Nobody ever dies — used to prove
    `run_game`'s `max_rounds` safety stop fires.
    """

    def night_actions(self, state: GameState, /) -> NightActions:
        return NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"}, doctor_protect="Vil1")

    def day_actions(self, state: GameState, /) -> DayActions:
        return DayActions(exile_votes={})

    def observe(self, state: GameState, new_events: tuple[Event, ...], /) -> None:
        """No-op: this scripted source ignores the transcript."""


def test_scripted_game_reaches_a_terminal_state() -> None:
    """A scripted 7-player game runs to a terminal state with a named winner.

    The core T14 acceptance criterion: the loop wires night/day resolution and
    the win check into a complete game. The final event is `GAME_OVER` and it
    names the winning faction.
    """
    stream = run_game(ROSTER, seed=42, decisions=_werewolf_win_script())
    events = stream.log.events

    assert events[-1].type == GAME_OVER
    assert events[-1].payload["winner"] == "werewolves"
    assert events[-1].recipients == ()  # the result is a public broadcast


def test_scripted_villager_win_game_ends_after_an_exile() -> None:
    """A scripted game the villagers win by exiling the last werewolf on a day.

    The werewolf-win script terminates after a night kill; this complementary
    script terminates after a day exile, exercising the loop's post-exile
    terminal check — without it, a game won on a day would run one phase too far.
    """
    stream = run_game(ROSTER, seed=42, decisions=_villager_win_script())
    events = stream.log.events

    assert events[-1].type == GAME_OVER
    assert events[-1].payload["winner"] == "villagers"


def test_scripted_game_is_deterministic() -> None:
    """Two runs of the same seeded scripted game produce identical transcripts.

    Invariant #4, verified through the T08 determinism harness — this is what
    makes recorded games replayable and post-hoc metrics trustworthy. A fresh
    `ScriptedDecisions` is built per run so the comparison is of the loop, not
    of shared mutable cursors.
    """
    assert_deterministic(lambda seed: run_game(ROSTER, seed, _werewolf_win_script()), seed=42)


def test_tied_kill_vote_resolves_differently_under_different_seeds() -> None:
    """A tied werewolf kill vote is broken by the seed — different seeds diverge.

    Invariant #4 at the integration level: the only stochastic point in a game
    is seed-derived, so two seeds over the same tied script produce different
    transcripts. Both still run to a terminal `GAME_OVER`.
    """
    stream_a = run_game(ROSTER, seed=0, decisions=_tied_first_night_script())
    stream_b = run_game(ROSTER, seed=1, decisions=_tied_first_night_script())

    def _first_kill_victim(stream: EventStream) -> object:
        return next(e.payload["victim"] for e in stream.log.events if e.type == KILL_RESOLVED)

    # The divergence is specifically the tied round-1 kill: the seed picks the victim.
    assert _first_kill_victim(stream_a) != _first_kill_victim(stream_b)
    assert stream_a.log.events[-1].type == GAME_OVER
    assert stream_b.log.events[-1].type == GAME_OVER


def test_no_private_event_is_logged_with_empty_recipients() -> None:
    """Every declared-private event in the transcript carries recipients.

    Invariant #2, end to end: the loop appends each draft through the
    private-event guard, so a `seer_inspect`/`doctor_protect`/`werewolf_chat`
    event can never reach the log as an empty-recipient broadcast.
    """
    stream = run_game(ROSTER, seed=42, decisions=_werewolf_win_script())

    private = [e for e in stream.log.events if e.type in PRIVATE_EVENT_TYPES]
    assert private  # the scripted game does emit private events — guard is non-vacuous
    for event in private:
        assert event.recipients != ()


def test_transcript_round_trips_through_jsonl() -> None:
    """The game transcript serializes and reloads losslessly (invariant #5).

    A game is only replayable and debuggable if its append-only event stream
    survives a JSONL write/read round-trip byte-identically.
    """
    stream = run_game(ROSTER, seed=42, decisions=_werewolf_win_script())
    lines = list(stream.to_jsonl_lines())
    restored = EventStream.from_jsonl_lines(lines)

    assert list(restored.to_jsonl_lines()) == lines


def test_plain_villager_never_observes_a_private_event() -> None:
    """A plain villager's observation view contains no private event (invariant #2).

    Routing a full game's transcript for a non-seer, non-doctor, non-werewolf
    player must yield zero `seer_inspect`/`doctor_protect`/`werewolf_chat`
    events — agents never read hidden state.
    """
    stream = run_game(ROSTER, seed=42, decisions=_werewolf_win_script())

    # Vil3 is a plain villager — no role grants it any private channel.
    observed = observations_for(stream.log.events, "Vil3")

    assert observed  # routing is non-vacuous: the villager still sees the public events
    assert all(e.type not in PRIVATE_EVENT_TYPES for e in observed)


def test_run_game_raises_when_a_script_never_terminates() -> None:
    """A non-terminating game hits the `max_rounds` safety stop and fails loud.

    `run_game` must not loop forever on a script that never reaches a terminal
    state; the `max_rounds` guard raises `RuntimeError` rather than hanging.
    """
    with pytest.raises(RuntimeError, match="round"):
        run_game(ROSTER, seed=42, decisions=_StallingDecisions(), max_rounds=3)
