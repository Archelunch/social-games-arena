"""Tests for the Werewolf-specific cognitive tools (T16).

`get_public_state` and `get_private_info` are the read-side LLM surface where
hidden state could leak if the renderer is sloppy. `state.players` carries
every player's `role`; a naive rendering would publish those to every caller,
breaking invariant #2 ("agents never read hidden state"). These tests pin
the *exact* shape of both renderers and add explicit leak guards:

  - `get_public_state` must not surface any role string from `Role`.
  - A villager's / doctor's `get_private_info` must not surface any other
    player's name (only their own role/faction).

Determinism (invariant #4) is also encoded: same input → byte-identical
output. The seer's inspection history is rendered from `memory.events`
filtered to `SEER_INSPECT` — `observations_for` (T05) already guarantees
only the seer's own inspections reach their memory, so this layer renders
them directly.
"""

import re

import pytest

from social_deduction_bench.agents import GameMemory
from social_deduction_bench.engine import Event, GameState, Phase
from social_deduction_bench.games.werewolf.cognitive import get_private_info, get_public_state
from social_deduction_bench.games.werewolf.events import SEER_INSPECT
from social_deduction_bench.games.werewolf.roles import Role

ROSTER = (
    ("Wolf1", "werewolf"),
    ("Wolf2", "werewolf"),
    ("Seer", "seer"),
    ("Doc", "doctor"),
    ("Vil1", "villager"),
    ("Vil2", "villager"),
    ("Vil3", "villager"),
)


def _day(round_: int = 2) -> GameState:
    """A day-phase state at `round_` (default 2 so we exercise a non-initial round)."""
    s = GameState.initial(ROSTER).with_phase(Phase.DAY)
    while s.round < round_:
        s = s.advanced_round()
    return s


def _inspect_event(round_: int, target: str, faction: str) -> Event:
    """Build the private SEER_INSPECT event the seer's memory would hold."""
    return Event(
        seq=0,
        round=round_,
        phase=Phase.NIGHT,
        type=SEER_INSPECT,
        payload={"target": target, "faction": faction},
        recipients=("Seer",),
    )


# --- get_public_state --------------------------------------------------------


def test_get_public_state_pins_literal_layout() -> None:
    """The rendered string is a known literal — pin it so a regression flips visibly.

    `phase=day` is `Phase.DAY.value` (the `StrEnum` string form); a future
    refactor that changes how `Phase` renders into the f-string would
    visibly diff this literal.
    """
    expected = "round=2 phase=day\nalive: Wolf1, Wolf2, Seer, Doc, Vil1, Vil2, Vil3\ndead: (none)"
    assert get_public_state(_day(), GameMemory(), "Vil1") == expected


def test_get_public_state_moves_killed_player_to_dead_list() -> None:
    """A killed player disappears from `alive:` and appears in `dead:`."""
    s = _day().with_player_killed("Wolf2")
    expected = "round=2 phase=day\nalive: Wolf1, Seer, Doc, Vil1, Vil2, Vil3\ndead: Wolf2"
    assert get_public_state(s, GameMemory(), "Vil1") == expected


def test_get_public_state_dead_list_follows_engine_player_order() -> None:
    """The dead list iterates `state.players`, not the kill order.

    Kills here are deliberately in reverse roster order ("Vil3" before
    "Wolf2"), but the rendered `dead:` line must still be in the original
    roster order (`Wolf2, Vil3`). A regression that iterated kill order
    instead would visibly flip these names.
    """
    s = _day().with_player_killed("Vil3").with_player_killed("Wolf2")
    expected = "round=2 phase=day\nalive: Wolf1, Seer, Doc, Vil1, Vil2\ndead: Wolf2, Vil3"
    assert get_public_state(s, GameMemory(), "Vil1") == expected


def test_get_public_state_all_dead_renders_alive_none() -> None:
    """All-dead board (terminal edge case) renders `alive: (none)`."""
    s = _day()
    for name in [n for n, _ in ROSTER]:
        s = s.with_player_killed(name)
    rendered = get_public_state(s, GameMemory(), "Vil1")
    assert "alive: (none)" in rendered
    assert "dead: Wolf1, Wolf2, Seer, Doc, Vil1, Vil2, Vil3" in rendered


def test_get_public_state_does_not_leak_role_strings() -> None:
    """**Invariant #2 leak guard for `get_public_state`.**

    `state.players` carries every player's role. A naive `str(state)` would
    publish all of them. Iterating over `Role` directly (not a hard-coded
    list) means a future fifth role auto-participates in the guard.
    """
    rendered = get_public_state(_day(), GameMemory(), "Vil1")
    for role in Role:
        assert role.value not in rendered, f"role {role.value!r} leaked into public-state output"


def test_get_public_state_does_not_mutate_state() -> None:
    """Pure read — `state` is byte-identical after the call."""
    s = _day()
    snapshot = s
    get_public_state(s, GameMemory(), "Vil1")
    assert s == snapshot


def test_get_public_state_is_a_deterministic_function_of_state() -> None:
    """Two states built from the same op sequence render byte-identical strings.

    Calling the renderer twice on the same instance only pins idempotency;
    this builds two independent states and compares, which is the real
    invariant-#4 check. The earlier version of this test also doubled as
    a "caller-independent" pin; that property still holds because
    `get_public_state` does not read `caller` beyond the existence check.
    """
    s1 = GameState.initial(ROSTER).with_phase(Phase.DAY).advanced_round().with_player_killed("Wolf2")
    s2 = GameState.initial(ROSTER).with_phase(Phase.DAY).advanced_round().with_player_killed("Wolf2")
    assert get_public_state(s1, GameMemory(), "Vil1") == get_public_state(s2, GameMemory(), "Vil1")


def test_get_public_state_unknown_caller_raises_key_error() -> None:
    """An unrecognized caller is a loop bug — fail loud per `state.player`."""
    with pytest.raises(KeyError):
        get_public_state(_day(), GameMemory(), "Nobody")


# --- get_private_info: per-role rendering ------------------------------------


def test_get_private_info_werewolf_lists_other_werewolves() -> None:
    """A werewolf's private view: own role/faction + the sorted partner list,
    caller excluded. Pack members see each other (WEREWOLF_DESIGN.md §5)."""
    expected = "you=Wolf1 role=werewolf faction=werewolves\npartners: Wolf2"
    assert get_private_info(_day(), GameMemory(), "Wolf1") == expected


def test_get_private_info_last_werewolf_alive_renders_partners_none() -> None:
    """When the caller is the only living werewolf, partners is `(none)`."""
    s = _day().with_player_killed("Wolf2")
    expected = "you=Wolf1 role=werewolf faction=werewolves\npartners: (none)"
    assert get_private_info(s, GameMemory(), "Wolf1") == expected


def test_get_private_info_seer_no_inspections_renders_none() -> None:
    """Empty memory → `inspections: (none)`. Seer always gets the section."""
    expected = "you=Seer role=seer faction=villagers\ninspections: (none)"
    assert get_private_info(_day(), GameMemory(), "Seer") == expected


def test_get_private_info_seer_renders_inspection_history() -> None:
    """Two SEER_INSPECT events render in memory insertion order — that is
    the engine's emit order, deterministic by seed (invariant #4)."""
    m = GameMemory()
    m.record_event(_inspect_event(round_=1, target="Wolf1", faction="werewolves"))
    m.record_event(_inspect_event(round_=2, target="Vil1", faction="villagers"))
    expected = "you=Seer role=seer faction=villagers\ninspections: R1 Wolf1=werewolves, R2 Vil1=villagers"
    assert get_private_info(_day(), m, "Seer") == expected


def test_get_private_info_seer_ignores_non_inspect_events_in_memory() -> None:
    """Only `SEER_INSPECT` events render in the inspections section.

    The seer's memory also holds public `KILL_RESOLVED` etc. — they are not
    inspections and must not appear in this section. Full-string equality
    (not `.endswith`) pins the *complete* render so a regression prepending
    a stray line or swapping the header would still flip the test."""
    m = GameMemory()
    m.record_event(Event(seq=0, round=1, phase=Phase.NIGHT, type="kill_resolved", payload={"victim": "Vil1"}))
    m.record_event(_inspect_event(round_=1, target="Wolf2", faction="werewolves"))
    expected = "you=Seer role=seer faction=villagers\ninspections: R1 Wolf2=werewolves"
    assert get_private_info(_day(), m, "Seer") == expected


def test_get_private_info_seer_renders_inspections_in_memory_insertion_order() -> None:
    """Inspections render in memory insertion order, even when writes are
    not chronological. A regression that sorted by round (`event.round`)
    would visibly flip this output. Mirrors the deliberately-reversed
    insertion trick `test_get_beliefs_multiple_rows_sorted_by_player_name`
    uses to detect a missing sort."""
    m = GameMemory()
    m.record_event(_inspect_event(round_=3, target="Vil1", faction="villagers"))
    m.record_event(_inspect_event(round_=1, target="Wolf1", faction="werewolves"))
    expected = "you=Seer role=seer faction=villagers\ninspections: R3 Vil1=villagers, R1 Wolf1=werewolves"
    assert get_private_info(_day(), m, "Seer") == expected


def test_get_private_info_seer_has_no_partners_line() -> None:
    """The seer's render carries no `partners:` line — that section is
    werewolf-only. A regression broadening the werewolf-branch condition
    (e.g. dropping the role check) would slip past the other tests."""
    rendered = get_private_info(_day(), GameMemory(), "Seer")
    assert "partners:" not in rendered


def test_get_private_info_villager_renders_only_self_line() -> None:
    """A plain villager sees only their own role/faction — no partners, no inspections."""
    assert get_private_info(_day(), GameMemory(), "Vil1") == "you=Vil1 role=villager faction=villagers"


def test_get_private_info_doctor_renders_only_self_line() -> None:
    """The doctor has no team-aware section — only their own role/faction."""
    assert get_private_info(_day(), GameMemory(), "Doc") == "you=Doc role=doctor faction=villagers"


# --- get_private_info: hidden-state leak guards ------------------------------


def test_get_private_info_villager_leaks_no_other_names_or_other_roles() -> None:
    """**Invariant #2 leak guard for `get_private_info` (villager).**

    A villager has no team — their private view must not name any other
    player and must not surface any other player's role string. Roles
    are matched on word boundaries (regex `\\b...\\b`) so the
    `villager`/`villagers` substring collision (the faction name) does
    not false-positive. Iterating `Role` directly auto-covers a future
    fifth role.
    """
    rendered = get_private_info(_day(), GameMemory(), "Vil1")
    for name, _ in ROSTER:
        if name == "Vil1":
            continue
        assert name not in rendered, f"villager's private view leaked name {name!r}"
    for role in Role:
        if role.value == Role.VILLAGER.value:
            continue  # caller's own role is allowed in the self-line
        assert not re.search(rf"\b{role.value}\b", rendered), f"villager's private view leaked role {role.value!r}"


def test_get_private_info_doctor_leaks_no_other_names_or_other_roles() -> None:
    """Same leak guard for the doctor — they also have no team-aware
    section, and the role check guards against the `Doc`/`doctor`
    substring collision. Word-boundary regex avoids the `villager` ⊂
    `villagers` faction-substring false positive."""
    rendered = get_private_info(_day(), GameMemory(), "Doc")
    for name, _ in ROSTER:
        if name == "Doc":
            continue
        assert name not in rendered, f"doctor's private view leaked name {name!r}"
    for role in Role:
        if role.value == Role.DOCTOR.value:
            continue  # caller's own role is allowed in the self-line
        assert not re.search(rf"\b{role.value}\b", rendered), f"doctor's private view leaked role {role.value!r}"


def test_get_private_info_werewolf_does_not_leak_non_werewolf_names() -> None:
    """A werewolf sees other werewolves only — never villagers, the seer, or the doctor."""
    rendered = get_private_info(_day(), GameMemory(), "Wolf1")
    for name in ("Seer", "Doc", "Vil1", "Vil2", "Vil3"):
        assert name not in rendered, f"werewolf's private view leaked non-pack name {name!r}"


# --- get_private_info: invariants -------------------------------------------


def test_get_private_info_dead_caller_still_reads() -> None:
    """A dead caller may still read their own private view — they are a
    spectator, and the cognitive layer never gates on alive."""
    s = _day().with_player_killed("Wolf1")
    rendered = get_private_info(s, GameMemory(), "Wolf1")
    assert rendered.startswith("you=Wolf1 role=werewolf faction=werewolves")


def test_get_private_info_dead_werewolf_still_sees_living_pack() -> None:
    """The spectator-read contract for `partners:` — a dead werewolf
    still sees their living pack member. The "living-only" filter
    excludes *dead pack members from the list*, not *the caller from
    seeing the list*: the caller's own alive flag is not consulted."""
    s = _day().with_player_killed("Wolf1")
    expected = "you=Wolf1 role=werewolf faction=werewolves\npartners: Wolf2"
    assert get_private_info(s, GameMemory(), "Wolf1") == expected


def test_get_private_info_does_not_mutate_state() -> None:
    """Pure read — `state` byte-identical after the call (also for the seer
    path that scans `memory.events`)."""
    s = _day()
    snapshot = s
    m = GameMemory()
    m.record_event(_inspect_event(round_=1, target="Wolf1", faction="werewolves"))
    get_private_info(s, m, "Seer")
    assert s == snapshot


def test_get_private_info_unknown_caller_raises_key_error() -> None:
    """Unrecognized caller → fail loud via `state.player`."""
    with pytest.raises(KeyError):
        get_private_info(_day(), GameMemory(), "Nobody")
