"""Read-side observation routing.

Enforces benchmark invariant #2: agents never read hidden state. T04's
`events.py` carries the routing marker — an `Event`'s `recipients` tuple is
empty for a public broadcast and non-empty for a private event addressed to
exactly those players. This module is the filter that consumes that marker:
given the event stream and an observer's name, `observations_for` returns only
the events that player is permitted to see, so no consumer (agent memory, T19)
re-implements the visibility rule and risks leaking a seer result or pack-chat
message to a non-recipient.

Routed events keep their original engine-assigned `seq`, so a filtered observer
sees a non-contiguous sequence (e.g. 0, 1, 3). That gap reveals only that *some*
private action occurred — already common knowledge from the public rules, since
every night has a seer/doctor/werewolf action — never its content, type, or
recipients, so it is not a hidden-state leak. `seq` is preserved deliberately:
it is the stable global id that replay (T28) and the determinism harness (T08)
correlate against, and renumbering would fork that namespace. The residual side
channel — counting gaps — is accepted; a stronger fix (a public
`private_action_occurred` placeholder) is a game-layer decision, out of scope.
"""

from collections.abc import Iterable

from social_deduction_bench.engine.events import Event


def observations_for(events: Iterable[Event], player: str) -> tuple[Event, ...]:
    """Return every event `player` may observe, in original log order.

    That is every public broadcast plus the private events naming `player` as
    a recipient. A blank or non-`str` `player` is rejected with `ValueError`:
    it can never be a valid recipient, so returning the public-only view would
    silently mask a caller bug (invariant #2, fail-loud).
    """
    if not isinstance(player, str) or not player:
        raise ValueError(f"player name must be a non-empty string, got {player!r}")
    return tuple(event for event in events if event.is_public or player in event.recipients)


def public_events(events: Iterable[Event]) -> tuple[Event, ...]:
    """Return only the public broadcast events — the spectator / replay view.

    A replay or spectator viewer must never be shown a private event; this is
    `observations_for` with no observer, expressed as its own named function.
    """
    return tuple(event for event in events if event.is_public)
