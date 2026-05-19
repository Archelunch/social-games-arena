"""Werewolf event-type constants and the `EventDraft` resolution spec.

These constants name every event the Werewolf rules emit. `KILL_RESOLVED`,
`EXILE_RESOLVED`, and `GAME_OVER` are public — the night's death, the day's
exile announcement, and the final winning-faction declaration are common
knowledge. The three private
constants (`SEER_INSPECT`, `DOCTOR_PROTECT`, `WEREWOLF_CHAT`) are bound to
`config.PRIVATE_EVENT_TYPES`, the frozenset the engine's private-event guard
enforces; keeping the constants and that set in lockstep is what stops a private
channel from silently shipping unguarded (invariant #2).

`EventDraft` is the transient event spec a resolver returns; the game loop turns
each draft into a logged `Event` on the append-only stream.
"""

from dataclasses import dataclass, field

# Public: the night's death announcement, broadcast to every player.
KILL_RESOLVED = "kill_resolved"

# Public: the day's exile announcement, broadcast to every player.
EXILE_RESOLVED = "exile_resolved"

# Public: the final transcript event, naming the winning faction.
GAME_OVER = "game_over"

# The literal an exile vote uses to mean "no choice"; excluded from the tally.
ABSTAIN = "abstain"

# Private: each must be routed to specific recipients only. These three MUST
# equal `config.PRIVATE_EVENT_TYPES`, the set the engine's guard enforces.
SEER_INSPECT = "seer_inspect"
DOCTOR_PROTECT = "doctor_protect"
WEREWOLF_CHAT = "werewolf_chat"


@dataclass(frozen=True, slots=True)
class EventDraft:
    """A transient event spec a resolver returns for the game loop to log.

    Frozen so a draft handed from a resolver to the loop cannot be altered in
    transit, desyncing the logged `Event` from the resolved outcome. Empty
    `recipients` marks a public broadcast (invariant #2's routing marker).
    """

    type: str
    payload: dict[str, object] = field(default_factory=dict)
    recipients: tuple[str, ...] = ()
