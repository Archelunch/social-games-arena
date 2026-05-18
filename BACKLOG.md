# Backlog — Social Deduction Benchmark

Durable task list. Derived from `social-deduction-benchmarks/WEREWOLF_DESIGN.md`.
Status: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked.
Pick the next task whose dependencies are all `[x]`. Update this file when a task changes state.

---

## M1 — Project scaffold & engine core

The game-agnostic, pure, seeded state machine. No game rules here.

- [x] **T01** — Scaffold `pyproject.toml` (Poetry, py3.13; deps: `dspy`, `litellm`, `trueskill`; dev: `pytest`, `ruff`, `pyrefly`), `src/social_deduction_bench/` package layout, `.gitignore` (`.reviews/`, `.dspy_cache/`, caches), `[tool.ruff]` + `[tool.pyrefly]` config. _Depends: —_
- [x] **T02** — Seeded RNG utility: a `Random` threaded explicitly through the engine; no global random state. _Depends: T01_
- [x] **T03** — Game state model: players, roles, alive/dead, round, current phase. Frozen/structured types. _Depends: T01_
- [x] **T04** — Event stream: append-only event log + JSONL serialization (write + read back). _Depends: T03_
- [x] **T05** — Observation routing: public events to all, private events to one agent; agents never see hidden state. _Depends: T04_
- [x] **T06** — Phase state machine: night/day transitions, terminal detection hook. _Depends: T03_
- [x] **T07** — Tool-call validation: reject illegal moves (dead target, wrong phase/role) with an informative error observation. _Depends: T05, T06_
- [ ] **T08** — Determinism harness: test utility asserting same seed → identical event stream. _Depends: T04_

## M2 — Werewolf game rules

The Werewolf-specific rules plugged into the M1 engine.

- [ ] **T09** — Role definitions (Werewolf, Seer, Doctor, Villager) + 7-player default config. The game definition also **declares its private event types** (`seer_inspect`, `werewolf_chat`, `doctor_protect`) for the engine's private-event guard — see Notes. _Depends: T03_
- [ ] **T10** — Seeded role assignment. _Depends: T02, T09_
- [ ] **T11** — Night resolution: werewolf joint kill vote, seer inspect, doctor protect, protection suppresses the kill. Private night events carry non-empty `recipients`; the engine **rejects a declared-private event type emitted with empty `recipients`** — see Notes. _Depends: T06, T07, T10_
- [ ] **T12** — Day resolution: exile vote, majority rule, tie → no exile. _Depends: T06, T07_
- [ ] **T13** — Win-condition checks: villagers win when both werewolves dead; werewolves win at parity. Checked after night AND after exile. _Depends: T11, T12_
- [ ] **T14** — Full game-loop integration test: a scripted 7-player game runs to a terminal state deterministically. _Depends: T08, T13_

## M3 — Tool set

The tools the ReAct agent calls. See WEREWOLF_DESIGN.md §6.

- [ ] **T15** — Game-action tools: `werewolf_chat`, `submit_kill_vote`, `seer_inspect`, `doctor_protect`, `submit_bid`, `speak`, `submit_exile_vote`. _Depends: T11, T12_
- [ ] **T16** — Cognitive tools: `get_public_state`, `get_private_info`, `recall`, `remember`, `get_beliefs`, `set_belief`, `get_plan`, `set_plan`. _Depends: T18, T19_
- [ ] **T17** — Tool/role/phase gating: each tool exposed only to the allowed role in the allowed phase. _Depends: T15_
- [ ] **T18** — Bidding-based speech ordering: collect bids, top-K speak in bid order. _Depends: T15_

## M4 — DSPy ReAct agent & memory

- [ ] **T19** — `GameMemory` (Tier 0): events, notes, beliefs; `remember`/`recall`/`set_belief`. _Depends: T01_
- [ ] **T20** — Belief table + persistent plan string wired into `GameMemory`. _Depends: T19_
- [ ] **T21** — DSPy ReAct agent: one decision-point loop, cognitive tools as intermediate steps, one game-action tool terminates the loop. _Depends: T16, T19_
- [ ] **T22** — `litellm` multi-model config: seat different LLMs as different players. _Depends: T21_
- [ ] **T23** — Smoke game with real LLM agents (auto-skips when no API key). _Depends: T14, T22_

## M5 — Rating, metrics & tournament

- [ ] **T24** — Metric extraction from the event stream: win/loss, per-role win rate, illegal-move rate, game length, tokens. _Depends: T04, T14_
- [ ] **T25** — Deceiver-vs-detector split metric (werewolf win rate vs. villager exile accuracy). _Depends: T24_
- [ ] **T26** — TrueSkill rating: update individual ratings from team outcomes. _Depends: T24_
- [ ] **T27** — Tournament runner: seeded cross-play over many games, ratings aggregated. _Depends: T23, T26_
- [ ] **T28** — Leaderboard output + replay tool (event stream → deterministic re-run). _Depends: T08, T27_

---

## Notes

- M1 is fully game-agnostic; M2 is the first game plugged in. ONUW and Secret Hitler reuse M1/M3-M5.
- Open design questions (discussion slot count `K`, exile tie-break, cross-game memory) are parked in WEREWOLF_DESIGN.md §12 — resolve before the task that needs them, not earlier.
- **Private-event guard (cross-game).** An `Event` is private exactly when its `recipients` set is non-empty; "public" is the empty set. The game-agnostic engine cannot tell a broadcast from a private event that forgot its recipients, since it sees `type` only as an opaque string. So each game **declares its private event types** (T09) and the engine **rejects a declared-private type emitted with empty `recipients`** (enforced in T11). One shared M1 check, reused by Werewolf / ONUW / Secret Hitler — not re-coded per game. Rationale: WEREWOLF_DESIGN.md §3. Origin: integrity-review High on the T04 multi-recipient `recipients` change (2026-05-18).
