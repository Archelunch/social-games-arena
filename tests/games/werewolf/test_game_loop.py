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
from social_deduction_bench.games.werewolf.events import (
    ACCUSATION,
    BID,
    DEFENSE,
    DISCUSSION_RESOLVED,
    EXILE_RESOLVED,
    GAME_OVER,
    KILL_BALLOTS,
    KILL_RESOLVED,
    SPEECH,
    WEREWOLF_CHAT,
    EventDraft,
)
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

    def night_chat(self, state: GameState, /) -> None:
        """No-op: the stalling source stages no chat."""

    def night_actions(self, state: GameState, /) -> NightActions:
        return NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"}, doctor_protect="Vil1")

    def day_actions(self, state: GameState, /) -> DayActions:
        return DayActions(exile_votes={})

    def bids(self, state: GameState, /) -> dict[str, int]:
        return {}

    def next_speech(self, state: GameState, speaker: str, /) -> str:
        return ""

    def next_reaction(self, state: GameState, reactor: str, /) -> None:
        """No-op: the stalling source stages no reaction."""

    def drain_drafts(self) -> tuple[EventDraft, ...]:
        return ()

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


def _dialogue_werewolf_win_script() -> ScriptedDecisions:
    """A scripted werewolf-win game enriched with night chat, bids, and speeches.

    Same round-by-round terminal outcomes as `_werewolf_win_script` so the win
    condition is still reached deterministically. Round 1 adds two werewolf
    chat lines, a bid map (`Wolf1`=9, `Seer`=5, others=0), and one scripted
    speech per top bidder (the bid resolver picks the top 3 in
    `K_DISCUSSION_SLOTS` order).
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
        night_chats=[
            (("Wolf1", "hunt Vil1"), ("Wolf2", "agreed")),
            (("Wolf1", "now Vil3"),),
        ],
        day_bids=[
            # Vil1 died on night 1 — bids cover the 6 living players only.
            {"Wolf1": 9, "Wolf2": 0, "Seer": 5, "Doc": 0, "Vil2": 0, "Vil3": 0},
        ],
        day_speeches=[
            (("Wolf1", "I am the seer; Vil2 is a werewolf"), ("Seer", "no I am, exile Wolf1")),
        ],
        day_reactions=[
            # Day 1: Vil2 (the accused) accuses back, Doc defends the real Seer;
            # the other living players (Wolf1, Wolf2, Seer, Vil3) pass.
            (
                ("Vil2", "accuse", "Wolf1", "Wolf1's seer claim is a lie"),
                ("Doc", "defend", "Seer", "the real seer is Seer, not Wolf1"),
            ),
        ],
    )


def test_dialogue_script_emits_all_new_event_types_with_correct_recipients() -> None:
    """A dialogue-rich scripted game logs WEREWOLF_CHAT, BID, DISCUSSION_RESOLVED, SPEECH events.

    Pins the new event vocabulary end to end: chat is private to the living
    werewolf pack, bids are private to each bidder, the resolved discussion
    is public with the chosen speaker tuple, and each speech is public.
    """
    stream = run_game(ROSTER, seed=42, decisions=_dialogue_werewolf_win_script())
    events = stream.log.events

    # Exactly 3 chats from the script: 2 on night 1, 1 on night 2. Both wolves
    # are alive across both nights in this script (day 1 exiles Vil2, not a
    # wolf), so every chat is private to the full pack {Wolf1, Wolf2}.
    # Pinning the count catches a double-emit or stray-emit regression.
    chats = [e for e in events if e.type == WEREWOLF_CHAT]
    assert len(chats) == 3
    n1_chats = [c for c in chats if c.round == 1]
    n2_chats = [c for c in chats if c.round == 2]
    assert len(n1_chats) == 2
    assert len(n2_chats) == 1
    assert all(set(c.recipients) == {"Wolf1", "Wolf2"} for c in chats)

    bids = [e for e in events if e.type == BID]
    assert len(bids) == 6  # one per alive day-1 player (Vil1 died on night 1)
    for event in bids:
        bidder = event.payload["bidder"]
        assert isinstance(bidder, str)
        assert event.recipients == (bidder,)

    discussion = [e for e in events if e.type == DISCUSSION_RESOLVED]
    assert len(discussion) == 1
    assert discussion[0].recipients == ()  # public
    speakers = discussion[0].payload["speakers"]
    # Wolf1=9 and Seer=5 take the top two slots; the third comes from the
    # zero-bidder tie group via seeded shuffle. K_DISCUSSION_SLOTS=3 is pinned
    # in test_config — assert exactly that count here so a drift in K shows up.
    assert isinstance(speakers, list)
    assert len(speakers) == 3
    assert speakers[0] == "Wolf1"
    assert speakers[1] == "Seer"
    assert speakers[2] in {"Wolf2", "Doc", "Vil2", "Vil3"}

    # The script supplies only 2 speeches (the third slot is intentionally
    # silent — `ScriptedDecisions.speeches` returns the script verbatim and
    # the loop tolerates a prefix-shorter return). Speech events match the
    # script length.
    speeches = [e for e in events if e.type == SPEECH]
    assert len(speeches) == 2
    assert all(e.recipients == () for e in speeches)
    assert [e.payload["speaker"] for e in speeches] == ["Wolf1", "Seer"]


def _all_accuse_day1_script() -> ScriptedDecisions:
    """The dialogue win script, but day 1 has ALL six living players accuse.

    Each living seat accuses (a non-self living target) with a caller-unique
    reason, so the `ACCUSATION` events land in exactly the order the loop drove
    the reaction round — i.e. the seeded reaction order. Lets a test read that
    order off the transcript and assert it is seed-derived, not fixed.
    """
    reactors = ["Wolf1", "Wolf2", "Seer", "Doc", "Vil2", "Vil3"]
    script = _dialogue_werewolf_win_script()
    script.day_reactions = [
        tuple((n, "accuse", ("Wolf2" if n == "Wolf1" else "Wolf1"), f"{n} reason") for n in reactors)
    ]
    return script


def _day1_accuser_order(seed: int) -> list[str]:
    stream = run_game(ROSTER, seed=seed, decisions=_all_accuse_day1_script())
    return [str(e.payload["accuser"]) for e in stream.log.events if e.type == ACCUSATION and e.round == 1]


def test_reaction_order_is_seed_derived_not_fixed() -> None:
    """The reaction round's order comes from the engine seed, not roster/alphabetical.

    Invariant #4 with a fairness twist: the per-day reaction order is a seeded
    `rng.shuffle`, so (a) a different seed yields a different order and (b) the
    order is not the trivial sorted/seat order — otherwise a fixed seat would get
    a permanent last-mover information edge that confounds cross-play ratings.
    Both assertions FAIL if the seeding is dropped (e.g. plain `sorted(...)`),
    which the prior same-seed-twice determinism tests could not catch. The seeds
    42 and 1 were hand-verified to diverge (6 reactors → 720 orders, collision
    negligible and pinned here).
    """
    order_42 = _day1_accuser_order(42)
    order_1 = _day1_accuser_order(1)

    assert sorted(order_42) == sorted(order_1)  # same set of reactors both runs
    assert order_42 != order_1  # the seed actually changes the order (kills de-seeding)
    assert order_42 != sorted(order_42)  # not the trivial alphabetical/seat order


def test_reaction_order_replays_identically_for_the_same_seed() -> None:
    """Same seed -> identical reaction order, every time (invariant #4).

    The reaction-order shuffle is a recorded stochastic point; two runs over the
    same seed must produce the same accuser sequence or recorded games diverge on
    replay.
    """
    assert _day1_accuser_order(42) == _day1_accuser_order(42)


def test_dialogue_script_emits_public_reaction_events() -> None:
    """The day reaction round logs public `ACCUSATION` / `DEFENSE` events.

    Every living player reacts once after the statements; here Vil2 accuses and
    Doc defends (the rest pass, emitting nothing). Both events are broadcast
    (empty recipients) so the whole table — and the suspicion metric — can read
    who accused or defended whom and why.
    """
    stream = run_game(ROSTER, seed=42, decisions=_dialogue_werewolf_win_script())
    events = stream.log.events

    accusations = [e for e in events if e.type == ACCUSATION]
    defenses = [e for e in events if e.type == DEFENSE]

    assert len(accusations) == 1
    assert accusations[0].recipients == ()  # public
    assert accusations[0].payload == {"accuser": "Vil2", "target": "Wolf1", "reason": "Wolf1's seer claim is a lie"}

    assert len(defenses) == 1
    assert defenses[0].recipients == ()  # public
    assert defenses[0].payload == {"defender": "Doc", "defended": "Seer", "reason": "the real seer is Seer, not Wolf1"}


def test_reaction_events_fall_between_speeches_and_the_exile_within_a_round() -> None:
    """Reactions are logged after the statements and before the exile vote resolves.

    Pins the day sub-phase order (bid -> statements -> reaction -> vote): the
    reaction round must run after speeches (so reactors can answer the
    statements) and before `EXILE_RESOLVED` (so the vote is cast with the full
    exchange in memory).
    """
    stream = run_game(ROSTER, seed=42, decisions=_dialogue_werewolf_win_script())
    day1 = [e for e in stream.log.events if e.round == 1 and e.phase.value == "day"]

    speech_seqs = [e.seq for e in day1 if e.type == SPEECH]
    reaction_seqs = [e.seq for e in day1 if e.type in {ACCUSATION, DEFENSE}]
    exile_seqs = [e.seq for e in day1 if e.type == EXILE_RESOLVED]

    assert speech_seqs
    assert reaction_seqs
    assert exile_seqs
    assert max(speech_seqs) < min(reaction_seqs), "reactions must follow the statements"
    assert max(reaction_seqs) < min(exile_seqs), "reactions must precede the exile resolution"


def test_night_chat_events_precede_kill_events_within_a_round() -> None:
    """The two-phase night logs every `WEREWOLF_CHAT` before that round's kill.

    `_run_night` runs the chat sub-phase, drains+observes it, then runs the
    kill vote. So within a round the chat must appear before `KILL_RESOLVED`
    in the append-only log — proving the chat is available to the wolves
    before they vote, not merely logged alongside the kill.
    """
    stream = run_game(ROSTER, seed=42, decisions=_dialogue_werewolf_win_script())
    events = stream.log.events

    for round_ in (1, 2):
        round_events = [e for e in events if e.round == round_]
        chat_seqs = [e.seq for e in round_events if e.type == WEREWOLF_CHAT]
        kill_seqs = [e.seq for e in round_events if e.type == KILL_RESOLVED]
        assert chat_seqs, f"round {round_} had no chat events"
        assert kill_seqs, f"round {round_} had no kill_resolved event"
        assert max(chat_seqs) < min(kill_seqs), f"round {round_}: chat must precede the kill"


def test_discussion_resolved_payload_pins_speakers_and_bid_map() -> None:
    """The public `DISCUSSION_RESOLVED` carries `speakers` (in order) and the full `bids` map.

    Audience replay (T31) consumes this — the chosen order plus the numeric
    bids that justified it must round-trip together so a reader can see why
    each speaker won a slot.
    """
    stream = run_game(ROSTER, seed=42, decisions=_dialogue_werewolf_win_script())

    discussion = next(e for e in stream.log.events if e.type == DISCUSSION_RESOLVED)
    bids_map = discussion.payload["bids"]
    assert bids_map == {
        "Wolf1": 9,
        "Wolf2": 0,
        "Seer": 5,
        "Doc": 0,
        "Vil2": 0,
        "Vil3": 0,
    }


def test_kill_ballots_event_carries_the_ballot_map_private_to_the_pack() -> None:
    """End-to-end: `KILL_BALLOTS` is private to the living wolves and carries the ballots.

    The public `KILL_RESOLVED` event carries only the victim — the ballot
    map rides a separate private event so villagers never see who voted
    for whom in the pack.
    """
    stream = run_game(ROSTER, seed=42, decisions=_dialogue_werewolf_win_script())

    kill_event = next(e for e in stream.log.events if e.type == KILL_RESOLVED)
    assert "ballots" not in kill_event.payload  # invariant #2 closure end-to-end
    assert kill_event.payload["victim"] == "Vil1"
    assert kill_event.recipients == ()  # public

    ballots_event = next(e for e in stream.log.events if e.type == KILL_BALLOTS)
    assert ballots_event.recipients == ("Wolf1", "Wolf2")
    assert ballots_event.payload["ballots"] == {"Wolf1": "Vil1", "Wolf2": "Vil1"}


def test_exile_ballots_appear_in_exile_resolved_payload() -> None:
    """`EXILE_RESOLVED.payload["ballots"]` carries the voter -> target map publicly.

    Exile votes are public by design (every alive player votes openly during
    the day), so this stays in the public event payload.
    """
    stream = run_game(ROSTER, seed=42, decisions=_dialogue_werewolf_win_script())

    exile_event = next(e for e in stream.log.events if e.type == EXILE_RESOLVED)
    assert exile_event.payload["ballots"] == {
        "Wolf1": "Vil2",
        "Wolf2": "Vil2",
        "Seer": "Vil2",
        "Doc": "Vil2",
        "Vil2": "Wolf1",
        "Vil3": "Wolf1",
    }


def test_dialogue_transcript_round_trips_through_jsonl() -> None:
    """Invariant #5 holds across the new event vocabulary — the dialogue-rich game replays losslessly."""
    stream = run_game(ROSTER, seed=42, decisions=_dialogue_werewolf_win_script())
    lines = list(stream.to_jsonl_lines())
    restored = EventStream.from_jsonl_lines(lines)

    assert list(restored.to_jsonl_lines()) == lines


def test_dialogue_script_is_deterministic() -> None:
    """Two runs of the same dialogue-enriched script produce byte-identical streams.

    A regression that bound recipient ordering to dict insertion or that
    re-randomized chat order would diverge here. A fresh script per run.
    """
    assert_deterministic(lambda seed: run_game(ROSTER, seed, _dialogue_werewolf_win_script()), seed=42)


def test_plain_villager_observation_view_excludes_chat_bids_and_speeches_addressed_to_others() -> None:
    """Invariant #2 closure under the new vocabulary.

    A plain villager observes neither werewolf-chat nor any bid from another
    player nor any tool-rejected event for another caller. They still observe
    public speeches, the public discussion-resolved event, and their own bid.
    """
    stream = run_game(ROSTER, seed=42, decisions=_dialogue_werewolf_win_script())
    observed = observations_for(stream.log.events, "Vil3")

    assert observed  # routing is non-vacuous

    # No werewolf-chat or seer-inspect or doctor-protect events.
    assert all(e.type not in PRIVATE_EVENT_TYPES or "Vil3" in e.recipients for e in observed)

    # The villager sees its own bid but no other bidder's BID event.
    bids_seen = [e for e in observed if e.type == BID]
    assert len(bids_seen) == 1
    assert bids_seen[0].payload["bidder"] == "Vil3"

    # Speeches and the resolution event are public — the villager sees them all.
    assert any(e.type == SPEECH for e in observed)
    assert any(e.type == DISCUSSION_RESOLVED for e in observed)


def test_dead_werewolf_is_not_a_recipient_of_later_chat_events() -> None:
    """A werewolf killed earlier in the game is excluded from later chat recipients.

    The script kills Wolf2 on round 1 via day exile, then round-2 chat names a
    lone living werewolf as the only recipient. Without the dynamic-pack
    guard a dead wolf could spy on the surviving wolf's chat by replaying its
    memory of the broadcast.
    """
    script = ScriptedDecisions(
        nights=[
            NightActions(kill_votes={"Wolf1": "Vil1", "Wolf2": "Vil1"}),
            NightActions(kill_votes={"Wolf1": "Vil2"}),
        ],
        days=[
            # Day 1: village exiles Wolf2.
            DayActions(
                exile_votes={
                    "Wolf1": "Seer",
                    "Wolf2": "Seer",
                    "Seer": "Wolf2",
                    "Doc": "Wolf2",
                    "Vil2": "Wolf2",
                    "Vil3": "Wolf2",
                }
            ),
            # Day 2: with Wolf2 dead and Vil2 freshly killed, the village exiles Wolf1 — villager win.
            DayActions(
                exile_votes={
                    "Wolf1": "Seer",
                    "Seer": "Wolf1",
                    "Doc": "Wolf1",
                    "Vil3": "Wolf1",
                }
            ),
        ],
        night_chats=[
            (("Wolf1", "we strike Vil1"), ("Wolf2", "yes")),
            (("Wolf1", "alone now — Vil2"),),
        ],
    )
    stream = run_game(ROSTER, seed=42, decisions=script)

    round2_chat = [e for e in stream.log.events if e.type == WEREWOLF_CHAT and e.round == 2]
    assert len(round2_chat) == 1
    assert round2_chat[0].recipients == ("Wolf1",)  # Wolf2 is dead — pack is just Wolf1


def test_run_game_raises_when_a_script_never_terminates() -> None:
    """A non-terminating game hits the `max_rounds` safety stop and fails loud.

    `run_game` must not loop forever on a script that never reaches a terminal
    state; the `max_rounds` guard raises `RuntimeError` rather than hanging.
    """
    with pytest.raises(RuntimeError, match="round"):
        run_game(ROSTER, seed=42, decisions=_StallingDecisions(), max_rounds=3)
