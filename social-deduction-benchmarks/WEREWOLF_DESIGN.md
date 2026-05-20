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
    werewolves   -> private chat sub-loop -> joint kill vote
    seer         -> inspect 1 player  (private result)
    doctor       -> protect 1 player
    resolve kill (suppressed if protected)
    check terminal
    # --- DAY ---
    announce death(s)
    discussion: K speaking slots, order decided by bidding
    voting: every alive player casts an exile vote
    resolve exile (plurality; tie -> no exile)   # see §12
    check terminal
```

**Speaking order — bidding.** Each discussion round, every alive agent submits a
bid (how much it wants to speak); the top `K` bidders get speaking slots, in bid
order. This models *when* to speak, not just what — adopted from Werewolf Arena
(Google, 2024), which also gives a published reference to validate against.

---

## 5. Decision points

A ReAct loop runs at each point below. The agent may call cognitive tools
freely, then **terminates** the loop with exactly one game-action tool.

| # | Phase | Acting role | Terminal game action |
|---|---|---|---|
| 1 | Night | Werewolves | `werewolf_chat` xN, then `submit_kill_vote` |
| 2 | Night | Seer | `seer_inspect` |
| 3 | Night | Doctor | `doctor_protect` |
| 4 | Day | All alive | `submit_bid` (0..N — desire to speak) |
| 5 | Day | Bid winners | `speak` (public statement) |
| 6 | Day | All alive | `submit_exile_vote` |

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
| `submit_bid(amount)` | int 0..N | All alive (day) |
| `speak(message)` | str | Bid winners (day) |
| `submit_exile_vote(target)` | player \| `"abstain"` | All alive (day) |

### 6.2 Cognitive tools (read-only / private scratchpad, intermediate)

| Tool | Purpose |
|---|---|
| `get_public_state()` | Alive players, round, death log, full vote history, transcript |
| `get_private_info()` | Your role, faction, werewolf partners, all your seer results |
| `recall(query, last_n_rounds)` | Retrieve from this agent's memory (see §7) |
| `remember(note)` | Write a free-text note to memory |
| `get_beliefs()` | Read the structured suspicion table |
| `set_belief(player, guess, confidence, evidence)` | Update one row of the table |
| `get_plan()` | Read the persistent strategic plan |
| `set_plan(text)` | Overwrite the strategic plan |

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
            items = [i for i in items if i["round"] >= cutoff]
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
- `submit_bid` amount range — **RESOLVED (T15+T18, 2026-05-20): `[0, 100]`.**
  T15 set the lower bound to 0 (reject negative); T18 set the upper bound to
  `MAX_BID = 100` (reject above). Bounded so an agent cannot grief a rated
  game with an unbounded bid (RNG cost, prompt-token bloat, integer-overflow
  surface). Top-K speaker selection lands in `resolve_discussion`
  (`K_DISCUSSION_SLOTS = 3`, seeded tie-break via `rng.shuffle` over a sorted
  per-amount tie group, so the §4 "majority by bidding" ordering is
  deterministic and replayable, invariant #4).
- Whether werewolves see each other's identity at game start (default: yes).
- Cross-game memory persistence (would push Tier 2 retrieval).
