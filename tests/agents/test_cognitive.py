"""Tests for the game-agnostic cognitive tools (T16).

The six tools in `agents/cognitive.py` (`recall`, `remember`, `get_beliefs`,
`set_belief`, `get_plan`, `set_plan`) are a thin LLM-facing wrapper over
`GameMemory`. The contract these tests encode:

  - Every tool validates the caller via `state.player(caller)` — an unknown
    name fails loud with `KeyError`, never silently degrades to a no-op.
  - Writers (`remember` / `set_belief` / `set_plan`) propagate the
    `ValueError`s `GameMemory` raises on invalid input; the cognitive
    layer does not catch and swallow them (CLAUDE.md rule 11, fail loud).
  - Render output is deterministic — sorted by player name for beliefs,
    fixed literal sentinels for empty sections — so a recorded LLM
    transcript replays byte-identical (invariant #4 at the read layer).
  - The wrappers add no game-specific knowledge: a placeholder role
    suffices to exercise every tool.
"""

import pytest

from social_deduction_bench.agents import Belief, GameMemory
from social_deduction_bench.agents.cognitive import (
    get_beliefs,
    get_plan,
    recall,
    remember,
    set_belief,
    set_plan,
)
from social_deduction_bench.engine import Event, GameState, Phase

ROSTER = (("Alice", "alpha"), ("Bob", "beta"), ("Carol", "gamma"))


def _state() -> GameState:
    """The default 3-player state: round 1, NIGHT (roles are placeholders)."""
    return GameState.initial(ROSTER)


def _state_at(round_: int, phase: Phase) -> GameState:
    """Build a state at an arbitrary round by chained advancement (rounds start at 1)."""
    s = GameState.initial(ROSTER).with_phase(phase)
    while s.round < round_:
        s = s.advanced_round()
    return s


# --- recall -------------------------------------------------------------------


def test_recall_empty_memory_returns_empty_string() -> None:
    """A fresh memory renders to ``""`` — matches `GameMemory.recall()` directly."""
    assert recall(_state(), GameMemory(), "Alice") == ""


def test_recall_returns_memory_recall_output_unchanged() -> None:
    """The wrapper is a one-line delegation; output is byte-identical to `memory.recall()`."""
    m = GameMemory()
    m.record_event(Event(seq=0, round=1, phase=Phase.NIGHT, type="kill_resolved", payload={"victim": "Bob"}))
    m.remember("Bob death looks suspicious", round_=2)
    assert recall(_state(), m, "Alice") == m.recall()


def test_recall_forwards_last_n_rounds() -> None:
    """`last_n_rounds` is passed straight through — output matches the direct call."""
    m = GameMemory()
    m.record_event(Event(seq=0, round=1, phase=Phase.DAY, type="x", payload={}))
    m.record_event(Event(seq=1, round=3, phase=Phase.DAY, type="y", payload={}))
    assert recall(_state(), m, "Alice", last_n_rounds=1) == m.recall(last_n_rounds=1)


def test_recall_negative_last_n_rounds_propagates_value_error() -> None:
    """`GameMemory.recall` raises on negative windows; the wrapper does not catch."""
    with pytest.raises(ValueError, match="last_n_rounds"):
        recall(_state(), GameMemory(), "Alice", last_n_rounds=-1)


def test_recall_unknown_caller_raises_key_error() -> None:
    """An unrecognized caller is a loop bug — fail loud, do not return public-only."""
    with pytest.raises(KeyError):
        recall(_state(), GameMemory(), "Nobody")


def test_recall_does_not_mutate_state() -> None:
    """Cognitive tools are pure reads — `state` is byte-identical after the call."""
    s = _state()
    snapshot = s
    recall(s, GameMemory(), "Alice")
    assert s == snapshot


# --- remember -----------------------------------------------------------------


def test_remember_writes_note_tagged_with_state_round() -> None:
    """The cognitive layer binds the "current round" from `state.round`.

    `GameMemory` is clock-agnostic (T19 docstring); the wrapper is where
    the engine clock meets the note. Pinning this means the LLM does not
    need to thread the round into its tool argument. The end-to-end
    closure (`recall` containing the rendered `[R2] note:` line) catches
    a regression where the round-binding stays correct in storage but
    drifts in the LLM-facing render.
    """
    s = _state_at(round_=2, phase=Phase.DAY)
    m = GameMemory()
    remember(s, m, "Alice", "Bob acted oddly")
    assert m.notes[0].round == 2
    assert m.notes[0].text == "Bob acted oddly"
    assert "[R2] note: Bob acted oddly" in recall(s, m, "Alice")


def test_remember_returns_ok_with_round_marker() -> None:
    """Success ack names the round so the LLM gets confirmation in its Observation."""
    assert remember(_state_at(round_=3, phase=Phase.DAY), GameMemory(), "Alice", "x") == "ok: noted at R3"


def test_remember_blank_text_propagates_value_error() -> None:
    """`Note.__post_init__` rejects blanks; the cognitive layer does not catch."""
    with pytest.raises(ValueError, match="text"):
        remember(_state(), GameMemory(), "Alice", "   ")


def test_remember_unknown_caller_raises_key_error() -> None:
    """Caller validation runs *before* `memory.remember`, so an unknown caller
    cannot accidentally append a note to a real memory."""
    m = GameMemory()
    with pytest.raises(KeyError):
        remember(_state(), m, "Nobody", "x")
    assert m.notes == ()


# --- get_beliefs --------------------------------------------------------------


def test_get_beliefs_empty_returns_none_sentinel() -> None:
    """The empty case has its own literal sentinel — distinguishable from a missing line."""
    assert get_beliefs(_state(), GameMemory(), "Alice") == "beliefs: (none)"


def test_get_beliefs_one_row_renders_literally() -> None:
    """A single belief renders header + one indented row, exact literal pin."""
    m = GameMemory()
    m.set_belief("Bob", "werewolf", "high", "voted with the pack")
    expected = "beliefs:\n  Bob: werewolf (high) -- voted with the pack"
    assert get_beliefs(_state(), m, "Alice") == expected


def test_get_beliefs_multiple_rows_sorted_by_player_name() -> None:
    """Rows are sorted by subject name regardless of insertion order.

    Insertion order is reversed alphabetically so a missing sort would
    visibly flip the rendered lines (the same trick T19's recall tests use).
    """
    m = GameMemory()
    m.set_belief("Carol", "villager", "low", "no evidence yet")
    m.set_belief("Alice", "werewolf", "high", "pack chat clue")
    m.set_belief("Bob", "villager", "medium", "voted alone")
    expected = (
        "beliefs:\n"
        "  Alice: werewolf (high) -- pack chat clue\n"
        "  Bob: villager (medium) -- voted alone\n"
        "  Carol: villager (low) -- no evidence yet"
    )
    assert get_beliefs(_state(), m, "Alice") == expected


def test_get_beliefs_empty_evidence_rendered_as_placeholder() -> None:
    """Empty `evidence` is legal at the storage layer; the renderer names it
    explicitly as `(no evidence)` so the row stays parseable."""
    m = GameMemory()
    m.set_belief("Bob", "werewolf", "high", "")
    assert get_beliefs(_state(), m, "Alice") == "beliefs:\n  Bob: werewolf (high) -- (no evidence)"


def test_get_beliefs_is_a_deterministic_function_of_write_sequence() -> None:
    """Two memories built from the same op sequence render byte-identical strings.

    The earlier "twice on the same instance" form only pinned idempotency.
    Building two independent `GameMemory`s and asserting equality is the
    full invariant-#4 check (mirrors `test_identical_operation_sequence_yields_identical_state`
    in `test_memory.py`).
    """
    m1, m2 = GameMemory(), GameMemory()
    for m in (m1, m2):
        m.set_belief("Bob", "werewolf", "high", "x")
        m.set_belief("Alice", "villager", "low", "y")
    assert get_beliefs(_state(), m1, "Alice") == get_beliefs(_state(), m2, "Alice")


def test_get_beliefs_unknown_caller_raises_key_error() -> None:
    """Caller validation runs even though the body does not need the caller."""
    with pytest.raises(KeyError):
        get_beliefs(_state(), GameMemory(), "Nobody")


# --- set_belief ---------------------------------------------------------------


def test_set_belief_writes_row_to_memory() -> None:
    """The wrapper delegates to `memory.set_belief`; the row is observable
    through `memory.beliefs[player]`."""
    m = GameMemory()
    set_belief(_state(), m, "Alice", "Bob", "werewolf", "high", "voted with the pack")
    assert m.beliefs["Bob"] == Belief(player="Bob", guess="werewolf", confidence="high", evidence="voted with the pack")


def test_set_belief_returns_ok_with_subject_name() -> None:
    """Success ack names the subject so the LLM's Observation is grounded."""
    assert set_belief(_state(), GameMemory(), "Alice", "Bob", "werewolf", "high", "ev") == "ok: belief set for Bob"


@pytest.mark.parametrize("bad_confidence", ["uncertain", "LOW", "Medium", "extreme", "", "medum"])
def test_set_belief_bad_confidence_propagates_value_error(bad_confidence: str) -> None:
    """`Belief.__post_init__` rejects a confidence outside `{low, medium, high}`;
    the cognitive layer does not catch. The parametrized list mirrors
    `test_set_belief_rejects_other_confidence_values` in `test_memory.py`
    so a future relaxation of the storage contract surfaces at both layers."""
    with pytest.raises(ValueError, match="confidence"):
        set_belief(_state(), GameMemory(), "Alice", "Bob", "werewolf", bad_confidence, "ev")  # type: ignore[arg-type]


def test_set_belief_overwrites_previous_row() -> None:
    """Second call for the same subject replaces the row (round-trip
    confirmation through the cognitive layer)."""
    m = GameMemory()
    set_belief(_state(), m, "Alice", "Bob", "werewolf", "high", "first")
    set_belief(_state(), m, "Alice", "Bob", "villager", "low", "second")
    assert m.beliefs["Bob"].guess == "villager"
    assert m.beliefs["Bob"].evidence == "second"


def test_set_belief_unknown_caller_raises_key_error() -> None:
    """Caller validation gates the write — an unknown caller cannot mutate memory."""
    m = GameMemory()
    with pytest.raises(KeyError):
        set_belief(_state(), m, "Nobody", "Bob", "werewolf", "high", "ev")
    assert dict(m.beliefs) == {}


# --- get_plan -----------------------------------------------------------------


def test_get_plan_default_returns_none_sentinel() -> None:
    """Default memory has no plan; the renderer signals it explicitly."""
    assert get_plan(_state(), GameMemory(), "Alice") == "plan: (none)"


def test_get_plan_after_set_renders_text() -> None:
    """A non-empty plan is rendered as a single `plan: <text>` line."""
    m = GameMemory()
    m.set_plan("reveal seer result on R3")
    assert get_plan(_state(), m, "Alice") == "plan: reveal seer result on R3"


def test_get_plan_unknown_caller_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_plan(_state(), GameMemory(), "Nobody")


# --- set_plan -----------------------------------------------------------------


def test_set_plan_writes_text_to_memory() -> None:
    """The wrapper delegates to `memory.set_plan`; `memory.plan` matches the text."""
    m = GameMemory()
    set_plan(_state(), m, "Alice", "stay quiet until R3")
    assert m.plan == "stay quiet until R3"


def test_set_plan_returns_ok_marker() -> None:
    """Success ack is the literal `ok: plan set`."""
    assert set_plan(_state(), GameMemory(), "Alice", "any plan") == "ok: plan set"


def test_set_plan_blank_text_propagates_value_error() -> None:
    """`GameMemory.set_plan` rejects blank text; the cognitive layer does not catch."""
    with pytest.raises(ValueError, match="Plan"):
        set_plan(_state(), GameMemory(), "Alice", "   ")


def test_set_plan_unknown_caller_raises_key_error() -> None:
    """Caller validation gates the write — an unknown caller cannot mutate the plan."""
    m = GameMemory()
    with pytest.raises(KeyError):
        set_plan(_state(), m, "Nobody", "x")
    assert m.plan == ""
