"""Tests for the Werewolf game-action tools (T15).

The seven game-action tools (WEREWOLF_DESIGN.md §6) are how an agent *commits*
a move. T15's `tools.py` turns a raw, parametrized invocation into a validated,
parsed action or an informative rejection — it does not emit events or mutate
state. These tests encode that contract and the benchmark invariants:

- #3 — agents change state only via *validated* tool calls; an illegal move
  (dead/unknown/self target, wrong role, wrong phase, bad argument) is rejected
  with an informative reason, never silently accepted.
- #1/#3 — a tool call is a pure verdict: it never mutates `GameState`.
- #4 — a rejected call's reason is deterministic, so the recorded error
  observation is byte-identical on replay.

Session decisions encoded here: self-targeting on a night-ability tool is
forbidden; `submit_bid` clamps to `[0, MAX_BID]` (T18 added the upper bound).
"""

import pytest

from social_deduction_bench.engine import GameState, Phase, available_tools
from social_deduction_bench.games.werewolf.config import BID_BUDGET, MAX_BID
from social_deduction_bench.games.werewolf.events import ABSTAIN
from social_deduction_bench.games.werewolf.tools import (
    ACCUSE,
    DEFEND,
    DOCTOR_PROTECT,
    PASS_TURN,
    SEER_INSPECT,
    SPEAK,
    SUBMIT_BID,
    SUBMIT_EXILE_VOTE,
    SUBMIT_KILL_VOTE,
    WEREWOLF_CHAT,
    WEREWOLF_TOOL_REQUIREMENTS,
    Reaction,
    ToolResult,
    accuse,
    defend,
    doctor_protect,
    pass_turn,
    seer_inspect,
    speak,
    submit_bid,
    submit_exile_vote,
    submit_kill_vote,
    werewolf_chat,
)

ROSTER = (
    ("Wolf1", "werewolf"),
    ("Wolf2", "werewolf"),
    ("Seer", "seer"),
    ("Doc", "doctor"),
    ("Vil1", "villager"),
    ("Vil2", "villager"),
    ("Vil3", "villager"),
)

ALL_TOOL_NAMES = frozenset(
    {
        WEREWOLF_CHAT,
        SUBMIT_KILL_VOTE,
        SEER_INSPECT,
        DOCTOR_PROTECT,
        SUBMIT_BID,
        SPEAK,
        SUBMIT_EXILE_VOTE,
        ACCUSE,
        DEFEND,
        PASS_TURN,
    }
)


def _night() -> GameState:
    """A fresh night-phase game (initial state — round 1, NIGHT)."""
    return GameState.initial(ROSTER)


def _day() -> GameState:
    """A fresh day-phase game, seeded with the full speaking budget so `submit_bid`
    happy-path tests bid against a real budget (the engine default is 0)."""
    return GameState.initial(ROSTER, bid_budget=BID_BUDGET).with_phase(Phase.DAY)


# --- registry / catalog -----------------------------------------------------


def test_registry_covers_exactly_all_tools() -> None:
    """The requirement registry has a gate for every tool and no extras.

    Invariant #3: a tool with no declared `ToolRequirement` would be ungated.
    The registry must name exactly the full tool set — no omission, no stray key.
    """
    assert set(WEREWOLF_TOOL_REQUIREMENTS) == ALL_TOOL_NAMES


def test_registry_is_immutable() -> None:
    """The registry cannot be mutated, so a tool's gates cannot drift mid-game."""
    with pytest.raises(TypeError):
        WEREWOLF_TOOL_REQUIREMENTS[SPEAK] = WEREWOLF_TOOL_REQUIREMENTS[SUBMIT_BID]  # type: ignore[index]


def test_night_tools_gate_the_night_phase_and_their_role() -> None:
    """The four night tools are gated to NIGHT and their acting role (§5/§6.1)."""
    expected = {
        WEREWOLF_CHAT: ("werewolf", False),
        SUBMIT_KILL_VOTE: ("werewolf", True),
        SEER_INSPECT: ("seer", True),
        DOCTOR_PROTECT: ("doctor", True),
    }
    for tool, (role, requires_target) in expected.items():
        req = WEREWOLF_TOOL_REQUIREMENTS[tool]
        assert req.phase is Phase.NIGHT
        assert req.role == role
        assert req.requires_target is requires_target


def test_day_tools_gate_the_day_phase_and_no_role() -> None:
    """The targetless day tools are gated to DAY with no role restriction (§5/§6.1).

    `submit_exile_vote` and `pass_turn` declare `requires_target=False` so the
    `ABSTAIN` literal / no-target reaction is not rejected as an unknown player —
    their functions do the target branch.
    """
    for tool in (SUBMIT_BID, SPEAK, SUBMIT_EXILE_VOTE, PASS_TURN):
        req = WEREWOLF_TOOL_REQUIREMENTS[tool]
        assert req.phase is Phase.DAY
        assert req.role is None
        assert req.requires_target is False


def test_reaction_target_tools_gate_the_day_phase_with_a_target() -> None:
    """`accuse` and `defend` are gated to DAY, any role, and require a target (§6).

    Both name a player (whom you accuse / defend), so the engine gate must demand
    a living-player target; neither carries a role restriction (every living
    player reacts).
    """
    for tool in (ACCUSE, DEFEND):
        req = WEREWOLF_TOOL_REQUIREMENTS[tool]
        assert req.phase is Phase.DAY
        assert req.role is None
        assert req.requires_target is True


# --- happy paths ------------------------------------------------------------


def test_submit_kill_vote_living_target_accepted() -> None:
    """A werewolf voting a living non-self player at night is accepted."""
    result = submit_kill_vote(_night(), "Wolf1", "Vil1")
    assert result == ToolResult(valid=True, value="Vil1")


def test_seer_inspect_living_target_accepted() -> None:
    """The seer inspecting a living non-self player at night is accepted."""
    result = seer_inspect(_night(), "Seer", "Wolf1")
    assert result == ToolResult(valid=True, value="Wolf1")


def test_doctor_protect_living_target_accepted() -> None:
    """The doctor protecting a living non-self player at night is accepted."""
    result = doctor_protect(_night(), "Doc", "Vil2")
    assert result == ToolResult(valid=True, value="Vil2")


def test_werewolf_chat_nonempty_message_accepted() -> None:
    """A werewolf chatting a non-empty message at night is accepted.

    The message is not a player name; acceptance proves no player-target lookup
    runs for a message-only tool.
    """
    result = werewolf_chat(_night(), "Wolf1", "let us target Vil1 tonight")
    assert result == ToolResult(valid=True, value="let us target Vil1 tonight")


def test_submit_bid_zero_accepted() -> None:
    """A bid of 0 (the floor — no desire to speak) is accepted."""
    assert submit_bid(_day(), "Vil1", 0) == ToolResult(valid=True, value=0)


def test_submit_bid_positive_accepted() -> None:
    """A positive bid below `MAX_BID` is accepted."""
    assert submit_bid(_day(), "Vil1", 7) == ToolResult(valid=True, value=7)


def test_submit_bid_at_max_bid_boundary_accepted() -> None:
    """The exact `MAX_BID` value is the upper inclusive bound — accepted.

    The cap is inclusive: an agent that bids the maximum must succeed, or the
    declared `[0, MAX_BID]` range would be off by one and exclude its top.
    """
    assert submit_bid(_day(), "Vil1", MAX_BID) == ToolResult(valid=True, value=MAX_BID)


def test_submit_bid_exceeding_remaining_budget_rejected() -> None:
    """A bid above the caller's remaining speaking budget is rejected.

    This is the bid economy's enforcement point: once a player has spent down
    their pool, they cannot keep bidding high. A depleted state (budget 10) must
    reject a bid of 11 even though 11 is well under `MAX_BID`.
    """
    depleted = GameState.initial(ROSTER, bid_budget=10).with_phase(Phase.DAY)
    result = submit_bid(depleted, "Vil1", 11)
    assert result.valid is False
    assert "budget" in result.reason
    assert "10" in result.reason
    assert "11" in result.reason


def test_submit_bid_at_exactly_remaining_budget_accepted() -> None:
    """Spending the entire remaining budget is allowed (inclusive bound)."""
    depleted = GameState.initial(ROSTER, bid_budget=10).with_phase(Phase.DAY)
    assert submit_bid(depleted, "Vil1", 10) == ToolResult(valid=True, value=10)


def test_submit_bid_max_bid_cap_applies_even_when_budget_is_larger() -> None:
    """`MAX_BID` is an independent per-bid ceiling, not subsumed by the budget.

    With a budget above `MAX_BID`, a bid over `MAX_BID` must still be rejected —
    the two caps are distinct (`min(MAX_BID, remaining)`), so this guards that the
    per-bid ceiling keeps biting when the constants diverge.
    """
    rich = GameState.initial(ROSTER, bid_budget=MAX_BID + 50).with_phase(Phase.DAY)
    result = submit_bid(rich, "Vil1", MAX_BID + 1)
    assert result.valid is False
    assert str(MAX_BID) in result.reason


def test_speak_nonempty_message_accepted() -> None:
    """A living player speaking a non-empty message in the day is accepted."""
    result = speak(_day(), "Vil1", "I suspect Wolf2")
    assert result == ToolResult(valid=True, value="I suspect Wolf2")


def test_submit_exile_vote_player_target_accepted() -> None:
    """An exile vote naming a living player is accepted with that player."""
    assert submit_exile_vote(_day(), "Vil1", "Wolf2") == ToolResult(valid=True, value="Wolf2")


def test_submit_exile_vote_abstain_accepted() -> None:
    """`abstain` is a legal exile-vote value — never rejected as an unknown player.

    `validate_tool_call` would reject `"abstain"` as an unknown target; the tool
    must special-case the `ABSTAIN` literal and accept it.
    """
    assert submit_exile_vote(_day(), "Vil1", ABSTAIN) == ToolResult(valid=True, value=ABSTAIN)


# --- rejection: engine gates delegated to validate_tool_call ----------------


def test_submit_kill_vote_by_non_werewolf_rejected() -> None:
    """A non-werewolf calling `submit_kill_vote` is rejected on the role gate."""
    result = submit_kill_vote(_night(), "Seer", "Vil1")
    assert result.valid is False
    assert "role" in result.reason


def test_seer_inspect_in_the_day_phase_rejected() -> None:
    """`seer_inspect` outside the night phase is rejected on the phase gate."""
    result = seer_inspect(_day(), "Seer", "Wolf1")
    assert result.valid is False
    assert "phase" in result.reason


def test_submit_kill_vote_dead_target_rejected() -> None:
    """A kill vote on an already-dead player is rejected (invariant #3).

    The reason must name the *target* — a dead-target reason that read like a
    dead-caller reason would mislead the agent's self-correction.
    """
    state = _night().with_player_killed("Vil1")
    result = submit_kill_vote(state, "Wolf1", "Vil1")
    assert result.valid is False
    assert "target" in result.reason


def test_submit_kill_vote_unknown_target_rejected() -> None:
    """A kill vote on a name that is not a player is rejected."""
    result = submit_kill_vote(_night(), "Wolf1", "Nobody")
    assert result.valid is False
    assert "unknown target" in result.reason


def test_submit_kill_vote_packmate_target_rejected() -> None:
    """A werewolf voting to kill a fellow werewolf is rejected (no friendly fire).

    Standard Werewolf rules: wolves know each other and cannot target their
    pack. Without this guard a model that misreads its own role can wipe
    out its team via two valid kill votes; the design doc resolved
    self-target as forbidden (T15), and this continues the same pattern
    for the packmate case. The reason must name the target as a fellow
    werewolf so the agent can self-correct on the next iteration.
    """
    result = submit_kill_vote(_night(), "Wolf1", "Wolf2")
    assert result.valid is False
    assert "fellow werewolf" in result.reason
    assert "Wolf2" in result.reason


def test_submit_kill_vote_villager_target_still_accepted() -> None:
    """Regression: the packmate guard must not block a legitimate villager target.

    A wolf voting a living villager is the canonical kill — that path
    must remain accepted after the new guard lands.
    """
    result = submit_kill_vote(_night(), "Wolf1", "Vil1")
    assert result == ToolResult(valid=True, value="Vil1")


def test_submit_kill_vote_seer_target_still_accepted() -> None:
    """Regression: the seer is a villager-faction non-wolf — wolves CAN target them.

    The guard checks `role`, not faction; the seer's role is `seer` not
    `werewolf`, so the kill vote stands.
    """
    result = submit_kill_vote(_night(), "Wolf1", "Seer")
    assert result == ToolResult(valid=True, value="Seer")


def test_tool_call_by_a_dead_caller_rejected() -> None:
    """A dead player cannot act — the call is rejected on the caller gate.

    The reason must name the *caller*, distinguishing it from a dead-target
    rejection so a gate-order regression cannot slip past this test.
    """
    state = _night().with_player_killed("Wolf1")
    result = submit_kill_vote(state, "Wolf1", "Vil1")
    assert result.valid is False
    assert "caller" in result.reason


def test_submit_exile_vote_unknown_player_target_rejected() -> None:
    """A non-`abstain` exile vote on an unknown name is rejected.

    Proves the player-target gates still run for `submit_exile_vote` even though
    its registry requirement declares `requires_target=False`.
    """
    result = submit_exile_vote(_day(), "Vil1", "Nobody")
    assert result.valid is False
    assert "unknown target" in result.reason


def test_werewolf_chat_by_non_werewolf_rejected() -> None:
    """A non-werewolf calling `werewolf_chat` is rejected on the role gate.

    The role gate must fire for a message tool too — a target-less tool still
    carries a role requirement.
    """
    result = werewolf_chat(_night(), "Seer", "let me into the pack chat")
    assert result.valid is False
    assert "role" in result.reason


def test_seer_inspect_by_non_seer_rejected() -> None:
    """A non-seer calling `seer_inspect` is rejected on the role gate."""
    result = seer_inspect(_night(), "Doc", "Wolf1")
    assert result.valid is False
    assert "role" in result.reason


def test_doctor_protect_by_non_doctor_rejected() -> None:
    """A non-doctor calling `doctor_protect` is rejected on the role gate."""
    result = doctor_protect(_night(), "Seer", "Vil1")
    assert result.valid is False
    assert "role" in result.reason


def test_submit_bid_in_the_night_phase_rejected() -> None:
    """A day-only tool called at night is rejected on the phase gate."""
    result = submit_bid(_night(), "Vil1", 3)
    assert result.valid is False
    assert "phase" in result.reason


def test_submit_exile_vote_abstain_by_a_dead_caller_rejected() -> None:
    """An abstain vote still runs the caller gate — a dead player cannot abstain.

    The abstain path skips only the player-*target* gate; it must still gate the
    caller, or a dead player could slip an abstain ballot into the tally.
    """
    state = _day().with_player_killed("Vil1")
    result = submit_exile_vote(state, "Vil1", ABSTAIN)
    assert result.valid is False
    assert "caller" in result.reason


def test_submit_exile_vote_abstain_in_the_night_phase_rejected() -> None:
    """An abstain vote outside the day phase is rejected on the phase gate.

    The abstain path must still run the phase gate — abstaining is a day action.
    """
    result = submit_exile_vote(_night(), "Vil1", ABSTAIN)
    assert result.valid is False
    assert "phase" in result.reason


# --- rejection: tool-specific argument gates --------------------------------


def test_submit_kill_vote_self_target_rejected() -> None:
    """A werewolf cannot kill-vote itself — self-targeting is forbidden."""
    result = submit_kill_vote(_night(), "Wolf1", "Wolf1")
    assert result.valid is False
    assert "cannot target the caller" in result.reason


def test_seer_inspect_self_target_rejected() -> None:
    """The seer cannot inspect itself — self-targeting is forbidden."""
    result = seer_inspect(_night(), "Seer", "Seer")
    assert result.valid is False
    assert "cannot target the caller" in result.reason


def test_doctor_protect_self_target_rejected() -> None:
    """The doctor cannot protect itself — self-targeting is forbidden."""
    result = doctor_protect(_night(), "Doc", "Doc")
    assert result.valid is False
    assert "cannot target the caller" in result.reason


def test_submit_bid_negative_amount_rejected() -> None:
    """A negative bid is rejected — the lower bound of the `[0, MAX_BID]` range."""
    result = submit_bid(_day(), "Vil1", -1)
    assert result.valid is False
    assert "non-negative" in result.reason


def test_submit_bid_above_max_bid_rejected() -> None:
    """A bid above `MAX_BID` is rejected — the upper bound of the range.

    Bounded to keep bid amounts comparable across rated games and to deny a
    misbehaving agent an unbounded-amount griefing surface (prompt-token bloat,
    integer-overflow surface). The reason must name both the cap and the
    offending amount so the agent's self-correction has the constraint to read.
    """
    result = submit_bid(_day(), "Vil1", MAX_BID + 1)
    assert result.valid is False
    assert str(MAX_BID) in result.reason
    assert str(MAX_BID + 1) in result.reason


def test_werewolf_chat_empty_message_rejected() -> None:
    """An empty werewolf-chat message is a malformed action — rejected."""
    result = werewolf_chat(_night(), "Wolf1", "")
    assert result.valid is False
    assert "non-empty" in result.reason


def test_speak_empty_message_rejected() -> None:
    """An empty public statement is a malformed action — rejected."""
    result = speak(_day(), "Vil1", "")
    assert result.valid is False
    assert "non-empty" in result.reason


def test_speak_whitespace_only_message_rejected() -> None:
    """A whitespace-only statement carries no content — rejected."""
    result = speak(_day(), "Vil1", "   \n\t ")
    assert result.valid is False
    assert "non-empty" in result.reason


# --- purity / determinism ---------------------------------------------------


def test_a_valid_tool_call_does_not_mutate_state() -> None:
    """An accepted tool call leaves `GameState` byte-identical (invariant #1)."""
    state = _night()
    submit_kill_vote(state, "Wolf1", "Vil1")
    assert state == GameState.initial(ROSTER)


def test_a_rejected_tool_call_does_not_mutate_state() -> None:
    """A rejected tool call leaves the position byte-identical (invariant #3)."""
    state = _night()
    submit_kill_vote(state, "Wolf1", "Wolf1")
    assert state == GameState.initial(ROSTER)


def test_a_rejected_calls_reason_is_deterministic() -> None:
    """The same rejected call yields an identical reason every time (invariant #4)."""
    state = _night()
    first = submit_kill_vote(state, "Seer", "Vil1")
    second = submit_kill_vote(state, "Seer", "Vil1")
    assert first.reason == second.reason
    assert first == second


# --- ToolResult shape -------------------------------------------------------


def test_tool_result_is_frozen() -> None:
    """`ToolResult` is immutable — a recorded verdict cannot be altered."""
    result = submit_bid(_day(), "Vil1", 1)
    with pytest.raises(AttributeError):
        result.valid = False  # type: ignore[misc]


def test_a_rejected_result_carries_no_value() -> None:
    """On rejection `value` is None — there is no parsed action to carry."""
    result = submit_bid(_day(), "Vil1", -1)
    assert result.valid is False
    assert result.value is None


def test_a_valid_result_carries_no_reason() -> None:
    """On success `reason` is empty — a reason exists only for a rejection."""
    result = submit_bid(_day(), "Vil1", 1)
    assert result.valid is True
    assert result.reason == ""


# --- T17: per-role/phase gating from the Werewolf registry ------------------
#
# `available_tools(state, caller, WEREWOLF_TOOL_REQUIREMENTS)` is the dual of
# `validate_tool_call`: the agent's "menu" of legal moves right now. These
# tests cross-check that the Werewolf registry produces the per-role menus
# WEREWOLF_DESIGN.md §5 prescribes (decision points 1-6).


def test_available_tools_werewolf_at_night() -> None:
    """A werewolf's night menu is exactly the pack chat + kill vote (§5 row 1).

    Pack chat lets them coordinate; the kill vote ends the decision point. Any
    other tool here would expose a move the werewolves should not have.
    """
    menu = available_tools(_night(), "Wolf1", WEREWOLF_TOOL_REQUIREMENTS)
    assert menu == (SUBMIT_KILL_VOTE, WEREWOLF_CHAT)


def test_available_tools_seer_at_night() -> None:
    """The seer's night menu is exactly `seer_inspect` (§5 row 2).

    The seer has one night ability and one shot; the menu must mirror that.
    """
    assert available_tools(_night(), "Seer", WEREWOLF_TOOL_REQUIREMENTS) == (SEER_INSPECT,)


def test_available_tools_doctor_at_night() -> None:
    """The doctor's night menu is exactly `doctor_protect` (§5 row 3)."""
    assert available_tools(_night(), "Doc", WEREWOLF_TOOL_REQUIREMENTS) == (DOCTOR_PROTECT,)


def test_available_tools_plain_villager_at_night_is_empty() -> None:
    """A plain villager has no night ability — the menu is empty.

    Invariant #2 — a villager has no night affordance; surfacing any tool
    would either let them act out of turn or hint that other roles exist.
    """
    assert available_tools(_night(), "Vil1", WEREWOLF_TOOL_REQUIREMENTS) == ()


@pytest.mark.parametrize("caller", ["Wolf1", "Seer", "Doc", "Vil1"])
def test_available_tools_any_role_at_day(caller: str) -> None:
    """Every alive role gets the same day menu — react, bid, speak, exile vote.

    §5 day rows are role-agnostic by design: every alive player bids, may speak
    (refinement to bid-winners is T18), reacts (accuse/defend/pass), and casts an
    exile vote. A role-specific day menu would change the social game. The menu
    is alphabetically sorted by `available_tools`.
    """
    menu = available_tools(_day(), caller, WEREWOLF_TOOL_REQUIREMENTS)
    assert menu == (ACCUSE, DEFEND, PASS_TURN, SPEAK, SUBMIT_BID, SUBMIT_EXILE_VOTE)


def test_available_tools_dead_werewolf_is_empty() -> None:
    """A dead werewolf has no menu, even though their role and phase match.

    Invariant #1 — dead players never act. Defense-in-depth with the call-time
    gate: an agent integration layer reading the menu must see "nothing", not
    a list of tools the engine will instantly reject.
    """
    state = _night().with_player_killed("Wolf1")
    assert available_tools(state, "Wolf1", WEREWOLF_TOOL_REQUIREMENTS) == ()


def test_available_tools_changes_when_phase_flips() -> None:
    """Same player at NIGHT and DAY sees different menus.

    Phase is the dial that flips the board. A menu that did not change with
    the phase would mean the gating is broken — either statically returning
    everything or ignoring `state.phase`.
    """
    seer_night = available_tools(_night(), "Seer", WEREWOLF_TOOL_REQUIREMENTS)
    seer_day = available_tools(_day(), "Seer", WEREWOLF_TOOL_REQUIREMENTS)
    assert seer_night != seer_day
    assert seer_night == (SEER_INSPECT,)
    assert seer_day == (ACCUSE, DEFEND, PASS_TURN, SPEAK, SUBMIT_BID, SUBMIT_EXILE_VOTE)


def test_registry_coverage_parity() -> None:
    """The union of all (role, phase) menus equals the registry's key set.

    Nothing in the catalog is unreachable — if a tool is declared, *some*
    living caller in *some* phase can call it. An orphaned tool would be a
    benchmark bug (an agent could never invoke it, skewing metrics).
    """
    callers = ("Wolf1", "Seer", "Doc", "Vil1")
    seen: set[str] = set()
    for state in (_night(), _day()):
        for caller in callers:
            seen.update(available_tools(state, caller, WEREWOLF_TOOL_REQUIREMENTS))
    assert seen == ALL_TOOL_NAMES


# --- reaction round: accuse / defend / pass_turn ----------------------------


def test_accuse_living_target_with_reason_accepted() -> None:
    """Accusing a living non-self player with a reason yields a `Reaction` value.

    The committed value carries both target and reason as a typed `Reaction`, so
    the adapter can build the public `ACCUSATION` event without re-parsing the
    LLM kwargs.
    """
    result = accuse(_day(), "Vil1", "Wolf2", "deflected every question")
    assert result == ToolResult(valid=True, value=Reaction(target="Wolf2", reason="deflected every question"))


def test_accuse_self_target_rejected() -> None:
    """A player cannot accuse itself — self-accusation is a no-op the referee rejects.

    Mirrors the `submit_kill_vote` self-target rule: a tool that names a victim
    must name someone other than the caller.
    """
    result = accuse(_day(), "Vil1", "Vil1", "framing myself?")
    assert result.valid is False
    assert "caller" in result.reason


def test_accuse_blank_reason_rejected() -> None:
    """An accusation with no reason is rejected — the reason is the public signal.

    A bare accusation carries no deduction content; requiring a non-empty reason
    keeps the `ACCUSATION` event meaningful (and the metric that reads it honest).
    """
    result = accuse(_day(), "Vil1", "Wolf2", "   ")
    assert result.valid is False
    assert "reason" in result.reason


def test_accuse_dead_target_rejected() -> None:
    """Accusing a dead player fails loud — the engine gate rejects the target."""
    state = _day().with_player_killed("Wolf2")
    result = accuse(state, "Vil1", "Wolf2", "still suspicious")
    assert result.valid is False


def test_accuse_in_night_phase_rejected() -> None:
    """`accuse` is a day-only action; calling it at night is rejected by the gate."""
    result = accuse(_night(), "Vil1", "Wolf2", "too early")
    assert result.valid is False
    assert "phase" in result.reason


def test_defend_living_target_with_reason_accepted() -> None:
    """Defending another living player with a reason yields a `Reaction` value."""
    result = defend(_day(), "Vil1", "Seer", "claimed seer and it checks out")
    assert result == ToolResult(valid=True, value=Reaction(target="Seer", reason="claimed seer and it checks out"))


def test_defend_self_target_accepted() -> None:
    """Self-defense is allowed — an accused player may defend itself.

    Unlike `accuse`, `defend` permits a self-target: the whole point of the
    reaction round is that an accused player can rebut, and the most direct rebut
    is defending yourself.
    """
    result = defend(_day(), "Vil1", "Vil1", "I was protecting, not killing")
    assert result == ToolResult(valid=True, value=Reaction(target="Vil1", reason="I was protecting, not killing"))


def test_defend_blank_reason_rejected() -> None:
    """A defense with no reason is rejected — the reason is the public signal."""
    result = defend(_day(), "Vil1", "Seer", "")
    assert result.valid is False
    assert "reason" in result.reason


def test_defend_dead_target_rejected() -> None:
    """Defending a dead player fails loud — the engine gate rejects the target."""
    state = _day().with_player_killed("Seer")
    result = defend(state, "Vil1", "Seer", "they were cleared")
    assert result.valid is False


def test_pass_turn_in_day_accepted() -> None:
    """Passing the reaction is valid in the day phase and carries no value.

    A reaction loop must be able to commit "nothing to add"; `pass_turn` is that
    terminal, and it stages no public event.
    """
    assert pass_turn(_day(), "Vil1") == ToolResult(valid=True, value=None)


def test_pass_turn_in_night_phase_rejected() -> None:
    """`pass_turn` is a day-only reaction; at night the gate rejects it."""
    result = pass_turn(_night(), "Vil1")
    assert result.valid is False
    assert "phase" in result.reason
