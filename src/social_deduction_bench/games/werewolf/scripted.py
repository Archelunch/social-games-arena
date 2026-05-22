"""A `DecisionSource` backed by pre-written per-round actions.

`ScriptedDecisions` plays a fixed script: per-round night actions, day actions,
and optional werewolf chat, bids, and speeches. M2 has no agents, so tests
and the T23 smoke runs drive `run_game` with a script — it lives in `src/`
like the determinism harness because it is production wiring, not a test
fixture. A fixed script ignores the live `GameState` apart from reading the
living werewolf pack when staging `WEREWOLF_CHAT` drafts (so recipients
exclude any wolf killed earlier in the game).
"""

from dataclasses import dataclass, field

from social_deduction_bench.engine import Event, GameState
from social_deduction_bench.games.werewolf.day import DayActions
from social_deduction_bench.games.werewolf.events import (
    ACCUSATION,
    BID,
    DEFENSE,
    SPEECH,
    WEREWOLF_CHAT,
    EventDraft,
)
from social_deduction_bench.games.werewolf.night import NightActions
from social_deduction_bench.games.werewolf.roles import Role


@dataclass
class ScriptedDecisions:
    """A `DecisionSource` backed by a fixed list of per-round actions.

    Used by tests and the T23 smoke runs; lives in `src/` like the determinism
    harness. The per-round cursors are mutable state, so a fresh instance must
    be built per game run — replay determinism (invariant #4) compares the loop,
    not shared cursors.

    `night_chats`, `day_bids`, `day_speeches`, and `day_reactions` are optional
    staging fields. Each entry is the script for one round of the matching phase.
    When the cursor outruns a list, that phase emits no agent drafts — the legacy
    `nights=`, `days=`-only construction still runs. Each `day_reactions` entry is
    a tuple of `(reactor, kind, target, reason)` where `kind` is `"accuse"` or
    `"defend"`; a reactor absent from the round's tuple passes (stages nothing).
    """

    nights: list[NightActions]
    days: list[DayActions]
    night_chats: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    day_bids: list[dict[str, int]] = field(default_factory=list)
    day_speeches: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    day_reactions: list[tuple[tuple[str, str, str, str], ...]] = field(default_factory=list)
    _night_cursor: int = field(default=0, init=False)
    _day_cursor: int = field(default=0, init=False)
    _bids_cursor: int = field(default=0, init=False)
    _pending: list[EventDraft] = field(default_factory=list, init=False)

    def night_chat(self, state: GameState, /) -> None:
        """Stage the scripted werewolf chat for this night — the chat sub-phase.

        Each scripted `(speaker, message)` for the current night index becomes
        one `WEREWOLF_CHAT` draft whose `recipients` is the sorted set of
        *currently-living* werewolf names (a wolf killed earlier is excluded).
        `Event.__post_init__` re-sorts recipients, but sorting here keeps the
        staged draft byte-identical to the logged event. Uses the night index
        the next `night_actions` will consume, so chat and kill stay aligned.
        """
        if self._night_cursor < len(self.night_chats):
            living_pack = tuple(sorted(p.name for p in state.alive_players() if p.role == Role.WEREWOLF.value))
            for speaker, message in self.night_chats[self._night_cursor]:
                self._pending.append(
                    EventDraft(
                        type=WEREWOLF_CHAT,
                        payload={"speaker": speaker, "message": message},
                        recipients=living_pack,
                    )
                )

    def night_actions(self, state: GameState, /) -> NightActions:
        """Return the next scripted night actions; fail loud if the script is exhausted.

        Chat is staged separately in `night_chat` (the two-phase night), so
        this method only advances the night cursor and returns the actions.
        `state` is accepted for Protocol shape and intentionally unused.
        """
        if self._night_cursor >= len(self.nights):
            raise RuntimeError(f"scripted decisions: no night actions for round {self._night_cursor + 1}")
        actions = self.nights[self._night_cursor]
        self._night_cursor += 1
        return actions

    def day_actions(self, state: GameState, /) -> DayActions:
        """Return the next scripted day actions; fail loud if the script is exhausted.

        A fixed script ignores `state` — that is intentional. A cursor past the
        end means the game ran longer than the script anticipated, a caller bug.
        """
        if self._day_cursor >= len(self.days):
            raise RuntimeError(f"scripted decisions: no day actions for round {self._day_cursor + 1}")
        actions = self.days[self._day_cursor]
        self._day_cursor += 1
        return actions

    def bids(self, state: GameState, /) -> dict[str, int]:
        """Return the next scripted bid map; stage one private `BID` draft per bidder.

        Each `BID` draft is private to its bidder (`recipients=(bidder,)`).
        When the cursor outruns the list, returns `{}` and stages no drafts —
        legacy scripts without bid data run unchanged. `state` is accepted
        for Protocol shape and intentionally unused.
        """
        if self._bids_cursor >= len(self.day_bids):
            return {}
        bid_map = self.day_bids[self._bids_cursor]
        for bidder, amount in bid_map.items():
            self._pending.append(
                EventDraft(
                    type=BID,
                    payload={"bidder": bidder, "amount": amount},
                    recipients=(bidder,),
                )
            )
        self._bids_cursor += 1
        return dict(bid_map)

    def next_speech(self, state: GameState, speaker: str, /) -> str:
        """Return `speaker`'s scripted line for this day and stage its public `SPEECH` draft.

        Indexed by `state.round` (day N reads `day_speeches[N - 1]`), so the
        per-speaker driver calls stay aligned with the round without a cursor.
        Returns `""` and stages nothing when the round has no script or the
        speaker has no scripted line — the driver tolerates a silent speaker.
        """
        index = state.round - 1
        if index < 0 or index >= len(self.day_speeches):
            return ""
        message = dict(self.day_speeches[index]).get(speaker)
        if message is None:
            return ""
        self._pending.append(
            EventDraft(
                type=SPEECH,
                payload={"speaker": speaker, "message": message},
                recipients=(),
            )
        )
        return message

    def next_reaction(self, state: GameState, reactor: str, /) -> None:
        """Stage `reactor`'s scripted day reaction (accuse/defend), or nothing.

        Indexed by `state.round` (day N reads `day_reactions[N - 1]`), mirroring
        `next_speech`. A reactor with no scripted entry for the round passes and
        stages no draft — the driver tolerates a silent reactor. An `"accuse"`
        kind stages a public `ACCUSATION` (`{accuser, target, reason}`); a
        `"defend"` kind stages a public `DEFENSE` (`{defender, defended, reason}`).
        """
        index = state.round - 1
        if index < 0 or index >= len(self.day_reactions):
            return
        by_reactor = {entry[0]: entry[1:] for entry in self.day_reactions[index]}
        spec = by_reactor.get(reactor)
        if spec is None:
            return
        kind, target, reason = spec
        if kind == "accuse":
            self._pending.append(
                EventDraft(type=ACCUSATION, payload={"accuser": reactor, "target": target, "reason": reason})
            )
        elif kind == "defend":
            self._pending.append(
                EventDraft(type=DEFENSE, payload={"defender": reactor, "defended": target, "reason": reason})
            )
        else:
            raise ValueError(f"scripted day reaction kind must be 'accuse' or 'defend', got {kind!r}")

    def drain_drafts(self) -> tuple[EventDraft, ...]:
        """Return and clear the staged event drafts (chats / bids / speeches / reactions).

        Returns `()` when nothing is pending — repeated calls after a flush
        are idempotent.
        """
        drained = tuple(self._pending)
        self._pending.clear()
        return drained

    def observe(self, state: GameState, new_events: tuple[Event, ...], /) -> None:
        """No-op: a fixed script has no per-player memory to update."""
