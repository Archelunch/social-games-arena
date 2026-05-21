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
    BID,
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

    `night_chats`, `day_bids`, and `day_speeches` are optional staging fields.
    Each entry is the script for one round of the matching phase. When the
    cursor outruns a list, that phase emits no agent drafts — the legacy
    `nights=`, `days=`-only construction still runs.
    """

    nights: list[NightActions]
    days: list[DayActions]
    night_chats: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    day_bids: list[dict[str, int]] = field(default_factory=list)
    day_speeches: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    _night_cursor: int = field(default=0, init=False)
    _day_cursor: int = field(default=0, init=False)
    _bids_cursor: int = field(default=0, init=False)
    _speeches_cursor: int = field(default=0, init=False)
    _pending: list[EventDraft] = field(default_factory=list, init=False)

    def night_actions(self, state: GameState, /) -> NightActions:
        """Return the next scripted night actions; fail loud if the script is exhausted.

        Before returning the actions, stage any scripted werewolf chat for the
        same night index as `WEREWOLF_CHAT` drafts whose `recipients` is the
        sorted set of *currently-living* werewolf names — a wolf killed earlier
        in the game is excluded. `Event.__post_init__` re-sorts recipients, but
        sorting here keeps the staged draft byte-identical to the logged event.
        """
        if self._night_cursor >= len(self.nights):
            raise RuntimeError(f"scripted decisions: no night actions for round {self._night_cursor + 1}")
        actions = self.nights[self._night_cursor]
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

    def speeches(self, state: GameState, speakers: tuple[str, ...], /) -> tuple[tuple[str, str], ...]:
        """Return the next scripted speeches; stage one public `SPEECH` draft per speech.

        `speakers` is the order resolved by `resolve_discussion` — the script
        is trusted to align with it (a divergence means the script and the
        bid resolver disagree, a caller bug). When the cursor outruns the
        list, returns `()` and stages no drafts.
        """
        if self._speeches_cursor >= len(self.day_speeches):
            return ()
        scripted = self.day_speeches[self._speeches_cursor]
        for speaker, message in scripted:
            self._pending.append(
                EventDraft(
                    type=SPEECH,
                    payload={"speaker": speaker, "message": message},
                    recipients=(),
                )
            )
        self._speeches_cursor += 1
        return scripted

    def drain_drafts(self) -> tuple[EventDraft, ...]:
        """Return and clear the staged event drafts (chats / bids / speeches).

        Returns `()` when nothing is pending — repeated calls after a flush
        are idempotent.
        """
        drained = tuple(self._pending)
        self._pending.clear()
        return drained

    def observe(self, state: GameState, new_events: tuple[Event, ...], /) -> None:
        """No-op: a fixed script has no per-player memory to update."""
