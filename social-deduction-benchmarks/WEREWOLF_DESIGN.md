# Werewolf Benchmark — Design Document

Design spec for a multi-agent LLM benchmark on classic **Werewolf**, built with
DSPy + ReAct. Memory, planning, and observations are cognitive tools; every game
action is a tool the agent calls. Different LLMs then compete on a rating board.

---

## 1. Scope & game ladder

This benchmark is the second of a three-game ladder. Werewolf is implemented
first as the **core** game because it is the most-studied (published baselines
to validate the harness against) and because multi-round play is where memory
and planning actually matter.

| Phase | Game | Role |
|---|---|---|
| 1 — Prototype | One Night Ultimate Werewolf | Validate the harness on the simplest loop (later) |
| **2 — Core** | **Werewolf (classic)** | **This document** |
| 3 — Signature | Secret Hitler | Novel contribution; no existing benchmark (later) |

The engine and agent abstraction are designed to be game-agnostic so phases 1
and 3 plug in without a rewrite.

---

## 2. Game configuration

**Benchmark default: 7 players** — long enough that memory and planning matter,
short enough to stay cheap.

| Role | Count | Faction | Night ability |
|---|---|---|---|
| Werewolf | 2 | Werewolves | Joint kill; private chat channel |
| Seer | 1 | Villagers | Inspect one player's faction |
| Doctor | 1 | Villagers | Protect one player from the kill |
| Villager | 3 | Villagers | None |

**Win conditions**
- **Villagers win** when both werewolves are dead.
- **Werewolves win** when `#werewolves_alive >= #non_werewolves_alive`.

**Variants** (post-MVP): 5-player (1 werewolf — fast smoke test) and 9-player
(3 werewolves + extra villager — harder).

---

## 3. Architecture: engine as referee

The **engine** is the single source of truth. Two hard rules:

1. **Agents never read hidden state.** They only receive *observations* the
   engine chooses to emit (public events to everyone; private events only to
   their named recipients — which may be a group, e.g. the werewolf pack).
2. **Agents only change state via validated tool calls.** Illegal moves (dead
   target, wrong phase, wrong role) are rejected with an error observation the
   agent can retry inside its ReAct loop.

**Private-event guard.** An event is private exactly when it carries a non-empty
`recipients` set; "public" is the empty set. The engine core is game-agnostic —
it sees an event's `type` only as an opaque string — so it cannot, on its own,
tell an intentional broadcast from a private event emitted with no recipients.
Therefore each **game definition declares its private event types** (Werewolf:
`seer_inspect`, `werewolf_chat`, `doctor_protect`), and the engine rejects any
declared-private type emitted with empty `recipients`. The hidden-state
guarantee is thus enforced once, in shared engine machinery, and reused by every
game (ONUW, Secret Hitler) rather than re-implemented per game.

The engine is seeded → deterministic → fully replayable. Every game is logged as
an append-only event stream for debugging and replay.

```
+-----------+        observations         +-------------------+
|  Engine   | --------------------------> |  Agent (DSPy      |
| (referee, |                             |  ReAct + memory)  |
|  seeded)  | <-------------------------- |                   |
+-----------+      validated tool calls   +-------------------+
```

---

## 4. Game loop

```
round = 0
while not terminal:
    round += 1
    # --- NIGHT ---
    # Two sub-phases so wolves coordinate before they vote:
    werewolves   -> chat sub-phase: each wolf sends one `werewolf_chat`
                 -> drain + observe the chat into every wolf's memory
                 -> vote sub-phase: each wolf commits `submit_kill_vote`
    seer         -> inspect 1 player  (private result)
    doctor       -> protect 1 player
    resolve kill (suppressed if protected)
    check terminal
    # --- DAY ---
    announce death(s)
    discussion: K speaking slots, order decided by bidding
                each speaker speaks in turn; the engine drains + observes
                each speech before the next speaker, so later speakers see
                and can react to earlier ones
    reaction round: every alive player reacts once, in a seeded order
                    -> accuse / defend / pass; the engine drains + observes
                       each reaction before the next, so a later reactor
                       sees earlier accusations and can answer them
    voting: every alive player casts an exile vote
    resolve exile (plurality; tie -> no exile)   # see §12
    check terminal
```

**Two-phase night.** Werewolf chat and the kill vote are separate sub-phases.
All wolves chat first; the engine drains those `WEREWOLF_CHAT` events and routes
them into every living wolf's memory; only then does each wolf commit its kill
vote. Without this split a wolf cannot see its packmate's message before voting,
so "coordination" would be an illusion. When only one werewolf remains alive the
chat sub-phase is skipped entirely — a lone wolf has no packmate to coordinate
with, so the chat would be a wasted decision (and LM call).

**Speaking order — bidding with a budget.** Each discussion round, every alive
agent submits a bid (how much it wants to speak); the top `K` bidders win
speaking slots, in bid order, and **pay their bid** out of a per-game
`BID_BUDGET` pool (first-price, winners-pay; losers pay nothing). The pool
depletes across rounds, so a bid is a real strategic signal — spending to speak
now costs voice later — instead of the degenerate always-max it was without a
cost. A player at 0 budget can still bid 0 and win a slot via the seeded
tie-break. This models *when* to speak, not just what — adopted from Werewolf
Arena (Google, 2024). Speeches are delivered one at a time and observed between
deliveries, so a later speaker reads the earlier speeches and can rebut them
within the same day. Every bid is public at resolution (`DISCUSSION_RESOLVED`
broadcasts the full bid map), so each player's remaining budget is common
knowledge and is surfaced in every agent's day brief (day only — at night no one
bids).

**Reaction round.** Bidding rations the *primary* statements (only K speak), so
without more the rest of the table is mute — accusations land into silence and no
one can defend the accused. After the statements, every living player therefore
reacts exactly once: a structured `accuse(target, reason)`, `defend(target,
reason)` (self-defense allowed), or `pass_turn`. Accusations and defenses are
public (`ACCUSATION` / `DEFENSE`, broadcast), so the whole table — and the
post-hoc suspicion-accuracy metric — can read who accuses or defends whom and
why; a wolf defending a wolf is a tell. The round is **sequential in a seeded
order** (a `rng.shuffle` of the living players, drawn from the same engine seed
as the discussion tie-break, *after* it), with drain + observe between reactors,
so a player reacting later sees the earlier accusations and can answer the same
day. A seeded order — not roster order — keeps a fixed seat from gaining a
systematic last-mover information edge that would confound cross-play ratings.
A reaction whose loop fails to commit (a truncated / unparseable response, or no
commit within `max_iters`) degrades to a silent **pass** — the round is optional
signal, so one bad reaction never aborts the game. (A *guaranteed* rebuttal turn
for anyone accused late in the order is deliberately deferred — see §12.)

---

## 5. Decision points

A ReAct loop runs at each point below. The agent may call cognitive tools
freely, then **terminates** the loop with exactly one game-action tool.

| # | Phase | Acting role | Terminal game action |
|---|---|---|---|
| 1 | Night (chat) | Werewolves | `werewolf_chat` (one message; the chat sub-phase) |
| 2 | Night (vote) | Werewolves | `submit_kill_vote` |
| 3 | Night | Seer | `seer_inspect` |
| 4 | Night | Doctor | `doctor_protect` |
| 5 | Day | All alive | `submit_bid` (0..remaining budget — desire to speak) |
| 6 | Day | Bid winners | `speak` (public statement) |
| 7 | Day (reaction) | All alive | `accuse` / `defend` / `pass_turn` (one short reaction) |
| 8 | Day | All alive | `submit_exile_vote` |

There is **no `finish` step**: a valid game-action commit ends the decision
point. (The agent loop builds its own ReAct predictor without DSPy's
auto-injected `finish` tool, so a model cannot plan a wasted second round-trip.)

**Brief grounding.** Each decision point's first message (the ReAct "brief")
is grounded with what the agent is entitled to know: an identity line restating
"you are <name>, <role>" *every* turn (the day briefs otherwise dropped the role,
so a wolf mid-day claimed to be a villager), the living roster, and — for a
werewolf — its living allies *excluding its own name* (listing the caller beside
its packmate made a wolf conflate the two and defend the wrong player); the
public rules (each faction's
win condition; which channels are public — `speak` and votes are seen by all,
`werewolf_chat` is pack-private; and how the night resolves — the doctor's guard
saves only if it matches the wolves' target, so guarding an un-targeted player
has no visible effect, which stops the "impossible protection" false-tell that
once mis-exiled a doctor); every living player's remaining speaking budget
(public, derivable from the broadcast bids); and a rendered, *legible* summary of
its own memory — its plan, suspicions, and recent events translated to plain
language (a night-kill reads differently from a day-exile) rather than raw event
JSON. The win-condition / channel lines are **rules, not strategy**: the brief
states facts and never coaches deception. This grounding removes the iterations
agents otherwise wasted calling read-only tools just to learn who they are or
what happened. Consequently the cognitive toolbelt is the **structured-write**
tools (`set_belief`, `set_plan`) plus `recall` for full-history dives; the
free-text `remember` is gone (a one-line rationale is now attached to the
committing action via an optional `note` argument — record-and-act in one call,
not a separate iteration); and the read-only tools (`get_private_info`,
`get_beliefs`, `get_plan`, `get_public_state`) are not offered because their
content is already in the brief.

**Cognitive tools are named in the brief.** The kept cognitive tools are listed
in every brief with the explicit rule that they do **not** end the turn (only the
game action does) — without this a small model could not tell they existed or
whether calling one terminated the loop, and burned whole iterations on the
ambiguity. The exile-vote brief additionally prompts the player to record a
suspicion via `set_belief` before voting: the vote is the one point the player
has heard the full statements + reactions, and populating the belief table is
what feeds the suspicion-accuracy metric (process prompting, never the answer).

---

## 6. Tool set

The core idea: **cognitive tools are how ReAct *thinks*; game-action tools are
how it *commits*.** One game action ends each decision point.

### 6.1 Game-action tools (mutate state, engine-validated, terminal)

| Tool | Args | Available to |
|---|---|---|
| `werewolf_chat(message)` | str | Werewolves (night) |
| `submit_kill_vote(target)` | player | Werewolves (night) |
| `seer_inspect(target)` | player | Seer (night) |
| `doctor_protect(target)` | player | Doctor (night) |
| `submit_bid(amount)` | int 0..min(`MAX_BID`, remaining budget) | All alive (day) |
| `speak(message)` | str | Bid winners (day) |
| `accuse(target, reason)` | player (not self) + str | All alive (day reaction) |
| `defend(target, reason)` | player (self allowed) + str | All alive (day reaction) |
| `pass_turn()` | — | All alive (day reaction) |
| `submit_exile_vote(target)` | player \| `"abstain"` | All alive (day) |

Every game-action tool also accepts an optional `note: str` — a short private
rationale recorded to the caller's memory on a *valid* commit (nothing is
recorded on a rejected call or a blank note). This folds "record my reasoning +
act" into one call, replacing the old standalone `remember` tool. `submit_bid` is
capped by both the per-bid ceiling `MAX_BID` and the player's remaining per-game
`BID_BUDGET`; winners pay their bid, so the budget depletes (§4, §6.1 economy).

### 6.2 Cognitive tools (read-only / private scratchpad, intermediate)

| Tool | Purpose | Exposed to loops? |
|---|---|---|
| `recall(query, last_n_rounds)` | Retrieve from this agent's memory (see §7) | Yes — deep history beyond the brief window |
| `set_belief(player, guess, confidence, evidence)` | Update one row of the table | Yes |
| `set_plan(text)` | Overwrite the strategic plan | Yes |
| `get_public_state()` | Alive players, round, death log, full vote history, transcript | No — rendered into the brief |
| `get_private_info()` | Your role, faction, werewolf partners, all your seer results | No — rendered into the brief |
| `get_beliefs()` | Read the structured suspicion table | No — rendered into the brief |
| `get_plan()` | Read the persistent strategic plan | No — rendered into the brief |

The read-only tools still exist (they back the brief's grounding render) but are
not offered to the ReAct loop: their content is in the first message, so
offering them only bloats the per-iteration prompt and invites wrong-arg calls.

---

## 7. Memory — no vector store

A single 7-player game's full transcript is only a few thousand tokens, so there
is no "doesn't fit in context" problem to solve. An LLM-driven memory layer
(e.g. mem0) would only add nondeterminism and a confound: when an agent plays
badly, was it the model or did retrieval miss?

**Decision: no vector store.** Memory is a plain in-process object per agent —
the engine-pushed event log, the agent's own notes, and the belief table.

```python
from dataclasses import dataclass, field

@dataclass
class GameMemory:
    """Per-agent memory. No DB, no embeddings, no extra LLM calls."""
    events: list[dict]  = field(default_factory=list)  # engine-pushed public events
    notes:  list[dict]  = field(default_factory=list)  # agent's remember() entries
    beliefs: dict       = field(default_factory=dict)  # player -> {guess, confidence, evidence}

    def remember(self, note: str, round_: int):
        self.notes.append({"round": round_, "text": note})

    def recall(self, query: str | None = None, last_n_rounds: int | None = None) -> str:
        items = self.events + self.notes
        if last_n_rounds is not None:
            cutoff = max((i["round"] for i in items), default=0) - last_n_rounds
            items = [i for i in items if i["round"] > cutoff]
        if query:  # cheap keyword filter — Tier 1
            terms = query.lower().split()
            items = [i for i in items
                     if any(t in str(i).lower() for t in terms)] or items
        items.sort(key=lambda i: i["round"])
        return "\n".join(f"[R{i['round']}] {i.get('text') or i}" for i in items)

    def set_belief(self, player, guess, confidence, evidence):
        self.beliefs[player] = {"guess": guess, "confidence": confidence,
                                "evidence": evidence}
```

### Retrieval tiers

`GameMemory` is an **interface**; each tier is a swappable implementation so the
memory mechanism can be measured as a benchmark variable.

| Tier | `recall` behavior | When |
|---|---|---|
| **0 — start here** | Return full event log + notes, optionally `last_n_rounds` filtered | 7-player games — fits in context |
| **1** | Keyword / substring filter (the `query` branch above) | If transcripts get noisy |
| **2** | Local embeddings + numpy cosine | Only at 12+ players or memory persisted *across* games |

For Tier 2: `sentence-transformers` (`all-MiniLM-L6-v2`, ~80 MB, local,
deterministic) + a numpy array + cosine — no vector DB, no server. Alternatively
call the existing local vLLM embedding endpoint with the same numpy-cosine logic.

---

## 8. Belief state

Memory holds raw events; **beliefs** are separate and *structured* — one row per
player: `{faction_guess, confidence, evidence}`, updated via `set_belief`. LLMs
track beliefs poorly in free-form memory, so forcing a structured table is itself
a measurable design lever (the structured table can be ablated on/off).

---

## 9. Planning

Each agent keeps one persistent `plan` string, read/written via `get_plan` /
`set_plan` — e.g. the Seer's "reveal my result on round 2 if I am accused." The
plan survives across decision points within a game; it is part of memory.

---

## 10. Rating & metrics

Werewolf is a **team game with asymmetric roles**, so use **TrueSkill**, not
plain Elo — it updates individual ratings from team outcomes and handles role
asymmetry. Per-game outcome = win/loss per player.

**Tracked metrics**
- TrueSkill rating per LLM (primary leaderboard).
- Per-role win rate (werewolf / seer / doctor / villager).
- **Deceiver vs. detector split** — werewolf win rate vs. villager exile accuracy.
  LLMs deceiving better than they detect was the recurring finding across the
  upstream survey; surface it as a first-class metric.
- Game length, illegal-move rate, tokens per game (cost).

**Competition format.** Cross-play: each game seats a mix of LLMs; many seeded
games; ratings aggregate across the population.

---

## 11. Engineering notes

- **Stack:** Python 3.13, Poetry, Ruff (line-length 120). DSPy for the agent;
  `litellm` under DSPy for multi-provider LLM access.
- **Determinism:** engine seeded; role assignment, kill resolution, and tie-breaks
  all derive from the seed. Same seed + same models → same game.
- **Transcript:** every game saved as an append-only event stream (JSONL) →
  replay, debugging, and post-hoc metric computation.
- **Illegal moves:** rejected with an error observation, retried within the ReAct
  loop, counted toward the illegal-move-rate metric.
- **Game-agnostic core:** engine, agent, memory, and rating are reusable; only the
  rules, roles, phases, and game-action tools are Werewolf-specific.

---

## 12. Open questions / later

- Discussion slot count `K` — **PARTIALLY RESOLVED (T18, 2026-05-20):
  `K_DISCUSSION_SLOTS = 3`** (Werewolf Arena baseline for 7-player games).
  Number of discussion rounds per day remains open — tune for signal vs.
  token cost once T23 produces real-LLM smoke games.
- Tie-break on exile votes — **RESOLVED (T12, 2026-05-19): no-exile on a tie.**
  No revote, no seed tie-break, so day resolution is RNG-free and deterministic.
  Relatedly, the §4 loop's "exile" rule is implemented as **plurality** (most
  votes wins), not an absolute >50% majority: with 7 players a strict-majority
  rule would stall most days. "Majority" in earlier drafts of this doc meant
  plurality; §4 now says so.
- Self-targeting on night-ability tools — **RESOLVED (T15, 2026-05-19): forbidden.**
  `submit_kill_vote`, `seer_inspect`, and `doctor_protect` reject a call whose
  target is the caller. The doc was silent; no doctor self-protect, no seer
  self-inspect, no werewolf self-kill-vote.
- Packmate-targeting on `submit_kill_vote` — **RESOLVED (2026-05-21): forbidden.**
  A werewolf voting to kill a fellow werewolf is rejected with a reason naming
  the target's pack membership. Standard Werewolf rules: wolves know each
  other (see `get_private_info`) and cannot kill their pack. Without this
  guard, a model that misreads its own role can wipe out its team via two
  valid kill votes; the resolution extends the T15 "self-target forbidden"
  pattern to the packmate case.
- `submit_bid` amount range — **RESOLVED (T15+T18, 2026-05-20): `[0, 100]`.**
  T15 set the lower bound to 0 (reject negative); T18 set the upper bound to
  `MAX_BID = 100` (reject above). Bounded so an agent cannot grief a rated
  game with an unbounded bid (RNG cost, prompt-token bloat, integer-overflow
  surface). Top-K speaker selection lands in `resolve_discussion`
  (`K_DISCUSSION_SLOTS = 3`, seeded tie-break via `rng.shuffle` over a sorted
  per-amount tie group, so the §4 "majority by bidding" ordering is
  deterministic and replayable, invariant #4).
- Bid economy — **RESOLVED (2026-05-22): per-game depleting budget.** The
  zero-cost bid was degenerate (rational play is always-max → speaker order
  collapsed to the seeded tie-break). Each player now has a per-game
  `BID_BUDGET = 100` pool; the top-K bidders win and **pay their bid** out of it
  (first-price, winners-pay), so the pool depletes across rounds and a bid
  carries real signal. `submit_bid` is capped by `min(MAX_BID, remaining
  budget)`. The budget is engine state (`PlayerState.bid_budget`, deducted in
  the seeded `resolve_discussion` ⇒ replayable); remaining budgets are public
  (bids are broadcast) and shown in every brief so agents can read who is eager
  vs. quiet. Budget size / pay-rule are `config.py` knobs.
- Day reaction round — **RESOLVED (2026-05-22): everyone reacts, structured,
  seeded order.** Bidding rations the K statements, so the rest of the table was
  mute and accusations went unanswered. After the statements every living player
  now reacts once with a structured `accuse` / `defend` / `pass_turn`; accusations
  and defenses are public events (`ACCUSATION` / `DEFENSE`) so the table reads who
  accuses/defends whom (and the suspicion metric can score it). The round is
  sequential in a **seeded order** — `rng.shuffle(sorted(alive_names))` drawn from
  the engine seed *after* the discussion tie-break — so it is replayable
  (invariant #4) and avoids a fixed seat-position advantage that would confound
  cross-play ratings; drain + observe between reactors lets a later reactor answer
  an earlier accusation the same day. A reaction that fails to commit degrades to
  a silent pass (optional signal must not crash the game). **Open:** genuinely
  *cheap* reactions — the short per-call token cap tried first truncated ReAct's
  mandatory `next_thought` mid-stream and broke parsing, so reactions currently
  use the full per-decision budget; a single-shot (non-ReAct) reaction predictor
  would be the way to make them cheap. Also open: a *guaranteed* rebuttal turn for
  a player accused late in the seeded order (currently they answer next day).
- Whether werewolves see each other's identity at game start (default: yes).
- Cross-game memory persistence (would push Tier 2 retrieval).
