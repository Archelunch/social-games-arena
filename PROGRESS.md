# Progress Log — Social Deduction Benchmark

Append-only. Newest entry on top. Read this first when starting a session.

**Current state:** T17 done — tool/role/phase exposure-query primitive landed in the engine. **M3 in progress**; all checks green (252/252).
**Next task:** T18 — Bidding-based speech ordering: collect bids, top-K speak in bid order. _Depends: T15 — done._ T16 (cognitive tools) remains blocked by T18 + T19.

**Tracked design decision (T09 + T11):** the private-event guard — each game
declares its private event types, the engine rejects a declared-private type
emitted with empty `recipients` — is recorded as acceptance criteria on T09
(declaration) and T11 (enforcement). T09 declares `PRIVATE_EVENT_TYPES`;
enforcement (engine function + loop wiring + negative test) lands in T11.
Cross-game rationale in `BACKLOG.md` Notes and `WEREWOLF_DESIGN.md` §3.

**M2 plan:** the whole milestone (T09–T14) is planned in
`~/.claude/plans/i-need-you-to-curried-conway.md`. Decided: flat modules in
`games/werewolf/` (no `GameDefinition` bundle); resolution functions pure; the
private-event guard is one shared `engine` function; T14 ships a production
`run_game` driver + `DecisionSource` Protocol.

---

## 2026-05-20 — T17: tool/role/phase exposure query

- Added `available_tools(state, caller, registry)` to `engine/validation.py`
  (exported from `engine/__init__.py`) — the dual of `validate_tool_call`:
  the latter rejects an illegal call, the former returns the *menu* of legal
  calls the caller may make right now. Pure read over `GameState`; filters a
  `Mapping[str, ToolRequirement]` to entries where `requirement.phase is None
  or == state.phase` *and* `requirement.role is None or == caller.role`; dead
  caller → empty tuple; unknown caller raises `KeyError` (matches
  `state.player`). Output is `tuple(sorted(names))` so it is byte-identical
  across runs (invariant #4).
- **Decision (plan, user-confirmed):** scope is the **exposure query only**.
  T07 had parked "error-observation emission" to "the game loop / T17 tool
  wiring" — re-scoped to T19/T21 (the agent-invocation layer that owns the
  retry loop and event context). `ToolResult.reason` still ends at memory;
  becoming a private `Event` belongs with the agent integration.
- **Decision:** primitive lives in `engine/validation.py`, next to
  `ToolRequirement` and `validate_tool_call` — same conceptual unit. A new
  `engine/gating.py` for one short function would have been the speculative
  indirection CLAUDE.md rule 2 rejects.
- **Decision:** no Werewolf-side wrapper. Callers invoke
  `available_tools(state, caller, WEREWOLF_TOOL_REQUIREMENTS)` directly,
  matching the T15 plan-deviation precedent (the werewolf sub-package
  convention is direct module imports, no re-export surface).
- **Decision:** dead caller → `()` (semantic emptiness) but unknown caller
  → `KeyError` (structural fail-loud). Mirrors `observations_for`'s
  blank-vs-unknown split — invariants get fail-loud, queries get empty.
- Tests: `tests/engine/test_gating.py` (new, 12) — happy path (role+phase
  match), per-axis filtering (wrong phase, wrong role), `None` = "no
  constraint" on each axis and on both, dead-caller empty, unknown-caller
  KeyError, empty-registry empty, state non-mutation (whole-object equality),
  output is a sorted `tuple` with deliberately reversed insertion order,
  determinism across two independent state builds. Werewolf cross-check in
  `tests/games/werewolf/test_tools.py` (11 new cases incl. a parametrized
  4-role day-menu test): per-role night menus (§5 rows 1–3), villager-empty
  night, every role at DAY → `(SPEAK, SUBMIT_BID, SUBMIT_EXILE_VOTE)` (§5
  rows 4–6), dead werewolf → empty, phase-flip changes the menu, registry
  coverage parity (the union of all (role, phase) menus equals the registry's
  key set — nothing in the catalog is unreachable). Suite 252/252.
- `/sdb-review`: python + test + integrity reviewers **all PASS** (0
  critical, 0 high, 0 medium). Two scoped-out follow-ups recorded in
  `consolidated.md`: T18 must filter `speak` out of the day menu for
  non-bid-winners once the speech sub-phase exists; T19/T21 should add an
  integration test pinning that any tool surfaced by `available_tools` is one
  `validate_tool_call` accepts. Reports in
  `.reviews/20260520-0812-7e4bc81-T17/`.
- Verified: `pytest` 252/252, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T18 — bidding-based speech ordering.

---

## 2026-05-19 — T15: Werewolf game-action tools (M3 begins)

- Added `games/werewolf/tools.py` — the seven game-action tools (WEREWOLF_DESIGN.md
  §6): `werewolf_chat`, `submit_kill_vote`, `seer_inspect`, `doctor_protect`,
  `submit_bid`, `speak`, `submit_exile_vote`. Each is a pure verdict function
  `(state, caller, arg) -> ToolResult` — it reads `GameState`, never mutates it,
  emits no events. `ToolResult` (frozen) carries `valid`/`reason`/`value`
  (the parsed argument on success, `None` on rejection).
- Role/phase/target gates are delegated to the engine's `validate_tool_call`
  via `WEREWOLF_TOOL_REQUIREMENTS`, a `MappingProxyType` registry of one
  `ToolRequirement` per tool (the catalog T17 gating will filter). Tool-specific
  argument rules are checked in the per-tool functions after the generic gate.
- **Scope decision (plan, D-scope):** T15 is purely additive — verdict + parsed
  value only. No event emission (would collide with the resolver-emitted
  `SEER_INSPECT`/`DOCTOR_PROTECT` drafts and need the loop's `EventLog`), no
  accumulation into `NightActions`/`DayActions` (the M4 `DecisionSource` adapter
  owns that). No engine/loop/resolver changes.
- **Decision (user):** self-targeting on a night-ability tool is **forbidden** —
  `submit_kill_vote`/`seer_inspect`/`doctor_protect` reject `caller == target`.
  **Decision (user):** `submit_bid` gates only `amount >= 0`; the upper bound is
  deferred to T18 (bidding). The doc was silent on both — `WEREWOLF_DESIGN.md`
  §12 updated with two flagged RESOLVED notes.
- **Decision:** `submit_exile_vote` is registered `requires_target=False` because
  its target may be the `ABSTAIN` literal (`validate_tool_call` would reject
  `"abstain"` as an unknown player). The function branches: an abstain vote runs
  caller/phase gates only; a non-abstain vote runs the full player-target gate
  via a requirement derived from the registry entry with `dataclasses.replace`.
- **Decision (plan deviation):** did *not* add re-exports to `games/werewolf/__init__.py`.
  The plan suggested it ("matches `engine/__init__.py`"), but the werewolf
  sub-package convention is direct module imports (`loop.py`, every test imports
  from `night`/`day`/`events` directly) — a lone re-export surface would break
  conformance (CLAUDE.md rule 10). T17 imports `from ...werewolf.tools import`.
- Tests `tests/games/werewolf/test_tools.py` (38): registry covers exactly the
  seven tools + is immutable + per-tool phase/role gates pinned; one happy path
  per tool incl. `abstain` accepted as a legal value; engine-gate rejections
  (wrong role for all four night tools, wrong phase, dead/unknown target, dead
  caller, day-tool-at-night, abstain-path caller/phase gates); tool-specific
  rejections (self-target ×3, negative bid, empty/whitespace message); purity
  (state byte-identical after a valid and an invalid call), determinism (same
  rejected call → identical reason), `ToolResult` frozen/shape. Suite 229/229.
- `/sdb-review`: python + integrity reviewers PASS; test reviewer NEEDS FIXES
  (1 High). Addressed before commit — High: dead-caller/dead-target tests both
  asserted `"dead"` (indistinguishable) → now assert `"caller"`/`"target"`;
  Mediums: message tests assert `"non-empty"`, added the four night-tool
  role-gate tests + the two abstain-path engine-gate tests; `submit_exile_vote`
  derives its requirement from the registry (one source of truth); `submit_bid`
  reason uses the `SUBMIT_BID` constant. Reports in
  `.reviews/20260519-1444-12c7d40-T15/`.
- Verified: `pytest` 229/229, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T17 — tool/role/phase gating.

---

## 2026-05-19 — T14: full game-loop integration (M2 complete)

- Added `games/werewolf/loop.py` — the production game-loop driver:
  - `DecisionSource` (`Protocol`) — the seam supplying decided night/day
    actions. Scripted in M2; an M4 DSPy agent adapter implements the same
    Protocol with no driver change. Positional-only `state` param (matches the
    `TerminalCheck` convention).
  - `run_game(roster, seed, decisions, game_id, max_rounds)` — the single owner
    of the `EventLog` and one seed-derived `GameRNG`. Runs the
    WEREWOLF_DESIGN §4 loop: night → check terminal → day → check terminal →
    next round. Logs every `EventDraft` through `assert_recipients_present`
    *before* append (a leaky private event never enters the stream). Appends a
    final public `GAME_OVER` event. `max_rounds` is a fail-loud `RuntimeError`
    safety stop against a non-terminating script.
- Added `games/werewolf/scripted.py` — `ScriptedDecisions`, a `DecisionSource`
  backed by pre-written per-round action lists with cursor-advance semantics
  and fail-loud exhaustion. Lives in `src/` (reused by T23), same rationale as
  the determinism harness. New public event constant `GAME_OVER`.
- **Decision (D3, plan):** the loop ships as production code, not test-only
  orchestration — T23/T27 reuse `run_game`. It is a thin driver: one hard-coded
  night→day→check loop parameterized only by `DecisionSource`. No phase
  registry, no `GameDefinition` ABC.
- Tests `tests/games/werewolf/test_game_loop.py` (8): a scripted 7-player
  werewolf-win game reaches a terminal `GAME_OVER` (public); a scripted
  villager-win game terminates *after a day exile* (the complementary
  post-exile terminal path); `assert_deterministic` via the T08 harness; a
  tied kill vote diverges by seed (the *victim* itself diverges); no private
  event is logged with empty recipients; the transcript round-trips through
  JSONL; a plain villager's `observations_for` view leaks no private event;
  a non-terminating script hits the `max_rounds` `RuntimeError`. Suite 191/191.
- `/sdb-review`: python + test + integrity reviewers all PASS (0 critical,
  0 high). Addressed the actionable test Mediums before commit: added the
  scripted villager-win game (exercises the post-exile break), strengthened the
  seed-divergence test to assert the kill victim diverges, asserted `GAME_OVER`
  is a public broadcast, switched the observer sample to a clearer villager.
  Reports in `.reviews/20260519-1148-7683cec/`.
- Verified: `pytest` 191/191, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**M2 (Werewolf game rules) is complete** — T09–T14 all done. The engine now
runs a full, seeded, deterministic, replayable 7-player Werewolf game end to
end with scripted decisions. M3 (the tool set agents call) is next.

**Next:** T15 — game-action tools.

---

## 2026-05-19 — T13: win-condition checks

- Added `games/werewolf/win.py` — `Winner` (`StrEnum`, WEREWOLVES/VILLAGERS),
  `winner(state) -> Winner | None`, and `is_game_over(state) -> bool`.
  `winner` is a pure read over `GameState`: villagers win at zero werewolves
  alive; werewolves win at parity (`#werewolves_alive >= #non_werewolves_alive`).
- **Decision (correctness invariant):** the villager check runs *before* the
  werewolf parity check. At the moment the last werewolf dies, `0 >= 0` parity
  is also true — checking parity first would mis-credit the werewolves a win at
  the instant they are wiped out. A dedicated test pins this ordering.
- `is_game_over` is the game's `TerminalCheck`: a plain function matching the
  engine's positional-only single-arg Protocol (T06), so the loop (T14) calls
  `is_terminal(state, is_game_over)`. The engine owns no win condition — it
  only relays this verdict (invariant #1).
- Tests `tests/games/werewolf/test_win.py` (11): villager win, werewolf parity
  win, werewolf outnumber win, game continues at 2v5 and 2v3, the check-order
  test (empty board → VILLAGERS not parity), `is_game_over` true/false, the
  engine `is_terminal` seam, `winner` purity, `Winner` string literals pinned.
  Suite 183/183.
- `/sdb-review`: python + test + integrity reviewers all PASS (0 critical,
  0 high). Only optional Low/Medium nits (a readability local; an unreachable
  bad-role test) — none applied. Integrity flagged for T14: the loop must run
  the check after *both* night resolution and exile. Reports in
  `.reviews/20260519-1136-706debd/`.
- Verified: `pytest` 183/183, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T14 — full game-loop integration test (closes M2).

---

## 2026-05-19 — T12: day resolution

- Added `games/werewolf/day.py` — `DayActions`/`DayResult` and `resolve_day`,
  a pure derivation mirroring `resolve_night`: tally exile votes (abstentions
  excluded), exile the **plurality** target, a tie resolves to **no exile**.
  Takes no `GameRNG` — the §12 tie-break is deterministic, so the day phase is
  replayable by construction (invariant #4). Emits one public `EXILE_RESOLVED`
  draft. New event constants `EXILE_RESOLVED` + `ABSTAIN` in `events.py`.
- **Decision (review-driven, resolves a doc/code conflict):** "majority" in
  `WEREWOLF_DESIGN.md` §4 is implemented as **plurality** — a strict >50% rule
  would stall most 7-player days. The doc is updated with a flagged note: §4
  now says "plurality", §12 records the exile tie-break RESOLVED to no-exile.
- **Decision (review High):** the integrity reviewer flagged that `resolve_day`
  tallied every ballot with no living-voter check — a dead voter could tip a
  plurality. Fixed: `resolve_day` now raises `ValueError` on a vote from a
  non-living player (the engine-as-referee must reject an ineligible ballot).
  The **same guard was added to `resolve_night`** (T11's file) so the two
  resolvers are consistent referees — surfacing the asymmetry rather than
  leaving it. *Target* legality stays trusted (tool-validation's job, T15);
  only *voter* legality is enforced here. Both `NightActions`/`DayActions`
  docstrings now state the split contract.
- Tests `tests/games/werewolf/test_day.py` (11): plurality exile, tie → no
  exile, all-abstain → no exile, abstentions excluded from the tally, plurality
  below an absolute majority still exiles, dead-voter rejected, public
  `EXILE_RESOLVED` + payload (named / `None` on a tie), input non-mutation
  (whole-state equality), non-day-phase rejected, determinism. Plus one new
  `test_night.py` test (dead-werewolf kill vote rejected). Suite 172/172.
- `/sdb-review`: python + test reviewers PASS; integrity reviewer NEEDS FIXES
  (1 High — the living-voter check, now fixed). All Mediums addressed before
  commit: plurality-below-majority test added; purity test strengthened to
  whole-object equality; trusted-target vs enforced-voter contract documented.
  Reports in `.reviews/20260519-1123-1270d88/`.
- Verified: `pytest` 172/172, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T13 — win-condition checks.

---

## 2026-05-19 — T11: night resolution + private-event guard

- **Engine:** added `assert_recipients_present(type, recipients, private_types)`
  to `engine/events.py` (exported) — the shared, game-agnostic guard that
  enforces invariant #2. The engine sees `type` as an opaque string, so it
  cannot tell a broadcast from a private event that forgot its recipients; the
  guard rejects any declared-private type emitted with empty `recipients`. The
  declared set is a parameter — Werewolf / ONUW / Secret Hitler all reuse it.
- **Game:** `games/werewolf/events.py` — event-type constants (`KILL_RESOLVED`
  public; `SEER_INSPECT`/`DOCTOR_PROTECT`/`WEREWOLF_CHAT` private, bound by test
  to `PRIVATE_EVENT_TYPES`) and the frozen `EventDraft` spec (the transient
  event the loop turns into a logged `Event`).
- **Game:** `games/werewolf/night.py` — `NightActions`/`NightResult` and
  `resolve_night`, a pure derivation mirroring `advance_phase`: werewolf joint
  kill vote (plurality; tie broken by `rng.choice` over a *sorted* leader list
  — the single stochastic point, seed-derived per invariant #4), seer private
  inspect (`recipients=(seer_name,)`, faction result), doctor protect, and
  protection suppresses the kill iff `doctor_protect == kill_target`. Fixed
  draft order seer→doctor→kill-resolved for byte-identical replay.
- **Decision:** `resolve_night` trusts decided inputs — target legality (alive,
  real player, right role/phase) is tool-call validation's job at the loop
  boundary (T07/T15), not the resolver's. Documented on `NightActions`.
- **Decision:** the private-event guard is declared+enforceable now, but the
  guard and night code first *meet* in the T14 game loop, which appends drafts
  through it. T11 adds a direct test (`resolve_night` drafts all pass the
  guard) so a leak is caught at the resolver, not only at the loop.
- **Tooling note:** `tests/games/werewolf/test_events.py` would have collided
  by basename with `tests/engine/test_events.py` under pytest's package-less
  prepend import mode. Resolved surgically by naming the file
  `test_event_types.py` — no `pyproject.toml` / import-mode change (an
  `--import-mode=importlib` patch was considered and rejected as out-of-scope
  repo-wide behavior change for a game-rules task).
- Tests: `tests/engine/test_event_guard.py` (4), `test_event_types.py` (4),
  `test_night.py` (14). Cover: guard accept/reject/empty-set; constants bound
  to the declared set; `EventDraft` frozen+defaults; plurality kill, protection
  suppress / non-suppress, seed tie-break (golden pins + leader-set membership
  + divergence), private seer result + faction correctness, public
  `KILL_RESOLVED`, fixed draft order, no-action single-draft, every draft
  passes the guard, input non-mutation, non-night-phase + empty-votes rejected.
  Suite 160/160.
- `/sdb-review`: python + test + integrity reviewers all PASS (0 critical,
  0 high). Addressed the Mediums before commit: made the `NightActions`
  trusted-inputs contract explicit; strengthened the seed-tie test with a
  leader-set membership + divergence assertion; added the resolver-drafts-pass-
  the-guard test. Reports in `.reviews/20260519-1116-f4de78d/`.
- Verified: `pytest` 160/160, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T12 — day resolution.

---

## 2026-05-19 — T10: seeded role assignment

- Added `src/social_deduction_bench/games/werewolf/assignment.py` — the seeded
  name->role deal (invariant #4: the assignment derives entirely from the
  engine seed, so a recorded game replays from turn one):
  - `assign_roles(names, role_multiset, rng)` — shuffles the *roles* (not the
    names) via `GameRNG.shuffle` and zips them onto fixed-order names, so the
    transcript roster stays in caller order with only the role column varying
    by seed. Fail-loud `ValueError` on a name/role count mismatch (either
    direction) and on a role string that is not a real `Role`.
  - `assign_default_roles(names, rng)` — convenience wrapper over the 7-player
    `default_role_multiset()`.
- **Decision:** shuffle roles against fixed-order names (not vice versa) — keeps
  `StreamHeader.players` in caller order; only the role column is seed-derived.
- Tests `tests/games/werewolf/test_assignment.py` (12): same-seed identity,
  different-seed divergence, **golden seed-42 literal** (pins the exact deal
  against an RNG/shuffle-algorithm change, mirroring `test_rng.py`), dealt
  roles equal the multiset, every name seated once, count mismatch rejected
  both directions, no global-`random` leak, input list not mutated, output
  builds a `GameState`, wrapper matches the explicit call. Suite 138/138.
- `/sdb-review`: python + test + integrity reviewers all PASS (0 critical,
  0 high). Addressed the consolidated Mediums before commit: `assign_roles`
  now validates each role string against `Role` (a typo'd role fails loud at
  deal time, not as a corrupt T13 win count); added the more-names-than-roles
  and unknown-role tests; fixed a test docstring that wrongly claimed
  `GameState.initial` does no duplicate-name rejection. Reports in
  `.reviews/20260519-1105-5f7f4f9/`.
- Verified: `pytest` 138/138, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T11 — night resolution + private-event guard enforcement.

---

## 2026-05-19 — T09: Werewolf roles + config (M2 begins)

- Added `src/social_deduction_bench/games/werewolf/` — the first concrete game
  plugged into the M1 engine:
  - `roles.py` — `Faction` (`StrEnum`, WEREWOLVES/VILLAGERS), `Role`
    (`StrEnum`, WEREWOLF/SEER/DOCTOR/VILLAGER), a private `MappingProxyType`
    role->faction map, and `faction_of(role: str) -> Faction`. `faction_of`
    accepts the engine's opaque `str` role and fails loud (`ValueError`) on an
    unknown role — an unrecognized role must never default into a faction and
    silently corrupt the T13 win check.
  - `config.py` — `DEFAULT_PLAYER_COUNT = 7`, `DEFAULT_ROLE_COUNTS`
    (`MappingProxyType`, 2 WW / 1 Seer / 1 Doctor / 3 Villager),
    `PRIVATE_EVENT_TYPES` (`frozenset`: `seer_inspect`, `werewolf_chat`,
    `doctor_protect`), and `default_role_multiset()` — a deterministic 7-tuple
    of role string values, one per seat.
- **Decision (user):** flat modules, no `GameDefinition` bundle object. The
  T06-parked "game-module interface" question is resolved to: each game exposes
  plain constants/functions; a unifying interface is extracted only if
  ONUW/Secret Hitler later earn it (`no abstractions for single-use code`).
- `roles.py`/`config.py` use `MappingProxyType` + `frozenset` so the role map,
  role counts, and declared private-event set cannot be mutated mid-game.
- Tests `tests/games/werewolf/test_roles.py` (6) + `test_config.py` (8): role
  ->faction map pinned; role/faction string literals pinned (they cross into
  transcripts and `ToolRequirement.role`); `faction_of` fail-loud on unknown
  role; closed role/faction sets; 7-player count + 2/1/1/3 counts pinned;
  multiset totals/contents/determinism; `PRIVATE_EVENT_TYPES` is exactly the
  three declared types; the multiset builds a valid `GameState` with roles
  intact. Suite 126/126.
- `/sdb-review`: python + test + integrity reviewers all PASS (0 critical,
  0 high). Addressed the one actionable Medium — the `GameState` test now also
  asserts role survival, not just player count. Remaining Mediums are T11
  follow-ups (bind `PRIVATE_EVENT_TYPES` strings to the T11 emitter constants).
- Verified: `pytest` 126/126, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T10 — seeded role assignment.

---

## 2026-05-19 — T08: determinism harness

- Added `src/social_deduction_bench/engine/determinism.py` — the test utility
  that turns invariant #4 (same seed → identical, replayable event stream)
  from an aspiration into a gate. Two public functions:
  - `assert_streams_identical(actual, expected)` — raises `AssertionError` on
    the first divergence, in a fixed check order (header → event count →
    events pairwise → JSONL serialization) so the failure reason is itself
    deterministic. The message names the first divergent event's `seq`.
  - `assert_deterministic(produce, seed)` — runs a `Callable[[int],
    EventStream]` producer twice with `seed` and compares the two transcripts.
  Both exported from `engine/__init__.py`.
- **Source finding that shaped the design:** `EventLog` defines no `__eq__`,
  so a frozen `EventStream`'s generated `__eq__` compares its `log` field by
  *identity* — two streams with identical content are never `==`. The
  comparator deliberately bypasses `EventStream ==`, comparing `header`
  (frozen, structural) and the `log.events` tuples (`tuple[Event, ...]`,
  structural) instead. We did **not** add `EventLog.__eq__` (T04 code, out of
  scope; the user chose a dedicated comparator to own this gap).
- **Decision (user):** harness lives in `src/` (`engine/determinism.py`), not
  `tests/` — it is a public engine util that T14's integration test and the
  T28 replay tool both consume; "test utility" describes its purpose, not its
  location. **Decision (user):** expose both the producer-driven
  `assert_deterministic` and the lower-level `assert_streams_identical`
  comparator (T28's replay-vs-recorded check can reuse the latter).
- **Decision:** explicit `raise AssertionError`, never bare `assert` — survives
  `python -O`, matching the engine's fail-loud convention.
- Tests `tests/engine/test_determinism.py` (9): seeded producer passes; the
  critical negative test (a seed-ignoring producer is *caught*); seed
  pass-through pinned (producer called twice with the exact seed);
  independently-built equal streams compare equal *despite* raw `!=` (pins the
  `EventLog`-no-`__eq__` rationale); payload / header / event-count / recipient
  divergence each detected; first-divergent-`seq` pinpointing. Suite 112/112.
- `/sdb-review`: python + test + integrity reviewers all PASS (0 critical,
  0 high). Fixed both Mediums before commit — scoped the negative fixture's
  `itertools.count()` into the one test that uses it (no shared module state);
  added `match=` anchors to the header / event-count / recipient negative
  tests. Reports in `.reviews/20260519-1031-c00a741-T08/`.
- Verified: `pytest` 112/112, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T09 — Werewolf role definitions (M2 begins).

---

## 2026-05-19 — T07: tool-call validation

- Added `src/social_deduction_bench/engine/validation.py` — the game-agnostic
  primitive enforcing invariant #3 (agents change state only via validated
  tool calls). Three frozen, slotted dataclasses — `ToolCall`
  (`caller`/`tool`/`target`), `ToolRequirement` (`phase`/`role`/
  `requires_target`, each `None` = "no constraint"), `ValidationResult`
  (`valid`/`reason`) — plus `validate_tool_call(state, call, requirement)`.
- `validate_tool_call` is a pure read over `GameState`: fixed-order,
  first-failure checks (caller known → alive → role → phase → target present →
  target known → target alive). Returns a verdict + informative reason string;
  it does not build or append the error `Event`.
- Exported the four symbols from `engine/__init__.py`.
- Tests `tests/engine/test_validation.py` (17): happy path, each illegal-move
  class with paired positive/negative gate tests, empty-requirement passthrough,
  fixed check-order pins (caller-before-target, role-before-phase,
  phase-before-missing-target), state-purity, result immutability.
- **Decision:** T07 returns a verdict, not an `Event` — error-observation
  emission (round/phase/recipients, private to the caller) is deferred to the
  game loop / T17 tool wiring. Keeps M1 game-agnostic.
- **Decision:** the caller is always required alive (dead players never act)
  and dead targets are always rejected — no "dead target allowed" flag, since
  no tool needs one. Game-agnostic assumptions, not Werewolf rules.
- `/sdb-review`: all 3 reviewers PASS, 0 critical / 0 high. Two Medium
  test-polish items (placeholder tool name; unpinned role/phase order) both
  fixed before commit — added the two ordering tests. Reports in
  `.reviews/20260518-2207-04a732a-T07/`.
- Verified: `pytest` 103/103, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T08 — determinism harness.

---

## 2026-05-18 — T06: phase state machine

- Added `src/social_deduction_bench/engine/phase.py` — the engine-owned
  night/day transition (invariant #1: the engine, not a game, advances
  phase/round) plus the terminal-detection hook:
  - `advance_phase(state) -> GameState` — pure derivation: NIGHT→DAY keeps the
    round, DAY→NIGHT opens the next round. Reuses T03's `with_phase` /
    `advanced_round`; no RNG, no clock (invariant #4: deterministic,
    replayable). `match` over `Phase` with a fail-loud `case _`.
  - `TerminalCheck` (`Protocol`) — the seam a game's win condition plugs into.
    `__call__(self, state, /)` — the parameter is positional-only so a game's
    callable need not match the parameter name (Pyrefly enforces name match on
    `Protocol.__call__` otherwise).
  - `is_terminal(state, check)` — thin relay; the engine forms no opinion on
    what ends a game. Werewolf supplies the check in T13.
  Exported from `engine/__init__.py`.
- Tests `tests/engine/test_phase.py` (9): NIGHT→DAY same round, DAY→NIGHT next
  round, full cycle from `initial()` pinning the round-1 convention, input
  non-mutation, roster preservation, death persistence across a phase change,
  determinism across two independent runs (now also asserts the cycle visits
  the expected distinct `(round, phase)` positions), hook delegation, and the
  hook receiving the actual state by identity. Suite 86/86.
- **Decision (user):** minimal hook now, not a full `GameDefinition` ABC. T06
  ships only the one-method `TerminalCheck` Protocol; the broader game-module
  interface (terminal + declared private-event types + roles) emerges in T09
  where it earns being an interface.
- **Decision (user):** first night = round 1. `WEREWOLF_DESIGN.md` §4's loop
  increments `round` before the first night; T03 had shipped `initial()` with
  `round=0`. T06 amends `GameState.initial()` to `round=1` and updates
  `test_state.py` (one rename + 3 round assertions). Blast radius verified
  contained — `Event.round` is an independent int literal, no rating/metrics
  code keys on round 0.
- **Decision:** `advance_phase` does not consult the terminal hook — terminal
  is a query the game loop (T14) makes between resolutions, not a transition
  guard. The transition stays a pure, unconditional NIGHT↔DAY function.
- `/sdb-review`: python + test + integrity reviewers all PASS (0 critical,
  0 high). Addressed the 1 Medium (strengthened the determinism test to also
  pin the distinct `(round, phase)` sequence, so an identity-degenerate
  `advance_phase` fails it) and the 1 Low / surfaced conflict (`case _` arm:
  `AssertionError`→`ValueError` to match `state.py`'s fail-loud convention and
  avoid `python -O` stripping). Reports in
  `.reviews/20260518-1521-9c2f326-t06phase/`.
- Verified: `pytest` 86/86, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T07 — tool-call validation.

---

## 2026-05-18 — T05: observation routing

- Added `src/social_deduction_bench/engine/observation.py` — the read-side of
  invariant #2 (agents never read hidden state). T04 carried the `recipients`
  marker; T05 is the filter that consumes it. Two pure, stateless functions:
  - `observations_for(events, player)` — every public broadcast plus the
    private events naming `player` as a recipient, in original log order.
  - `public_events(events)` — public broadcasts only, the spectator / replay
    (T28) view.
  Both take `Iterable[Event]` (works with `EventLog`, the `.events` snapshot,
  or a bare list/generator) and return `tuple[Event, ...]`. Exported from
  `engine/__init__.py`.
- Tests `tests/engine/test_observation.py` (14): public broadcast reaches all,
  private delivered to its recipient, the core leak test (a non-recipient's
  observations contain zero private events), multi-recipient pack-chat routing,
  `seq`/order preservation, source-log non-mutation, determinism across two
  independent logs, blank-player rejection, unknown-player → public-only,
  exact recipient-name matching (case/whitespace significant), bare-iterable
  input, empty input. Suite 77/77.
- **Decision:** routed events keep their original engine-assigned `seq`, so a
  filtered observer sees non-contiguous `seq` (e.g. 0, 1, 3). Not a
  hidden-state leak — a gap reveals only that *some* private action occurred
  (already common knowledge from the public rules), never its content, type,
  or recipients. Renumbering would break replay (T28) / determinism (T08) /
  metrics (T24), which correlate events by the stable global `seq`. Residual
  gap-counting side channel accepted; a public `private_action_occurred`
  placeholder is a game-layer call, out of T05 scope. All three reviewers
  endorsed this trade-off.
- **Decision:** `observations_for` rejects a blank / non-`str` `player` with
  `ValueError` — silently degrading to public-only would mask a caller bug.
  No roster validation: the game-agnostic router has no roster; a non-blank
  unknown name correctly matches no `recipients` and yields public-only.
- **Decision:** recipient matching is exact (literal string) — no case-fold,
  no trim — so a near-miss name cannot leak a private event.
- `/sdb-review`: python + test + integrity reviewers all PASS (0 critical,
  0 high). Addressed the 2 Mediums + 1 Low test-hardening nits: `assert seen`
  guard on the leak test (was vacuous-safe on an empty stream), determinism
  test now routes two independent logs, added the exact recipient-name match
  test. Reports in `.reviews/20260518-1302-0fa9436-t05obs/`.
- Verified: `pytest` 77/77, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T06 — phase state machine.

---

## 2026-05-18 — T04 follow-up: multi-recipient private events

- Schema change to `engine/events.py`: replaced `Event.visibility`
  (`Visibility` enum) + `Event.recipient: str | None` with a single
  `recipients: tuple[str, ...]` field. Empty tuple = public broadcast;
  non-empty = private to exactly those players. Removed the `Visibility` enum.
  Added an `is_public` property. `__post_init__` validates recipients
  (non-empty `str`, no duplicates) and stores them in canonical sorted order.
- **Why:** `WEREWOLF_DESIGN.md` gives the werewolves a *shared private chat
  channel* (`werewolf_chat`). A single `recipient: str | None` could not
  address the pack; a private event must reach multiple players. Surfaced by a
  user question — the original T04 plan missed it.
- **Decision:** `recipients` is the single source of truth for public/private —
  no separate `Visibility` field. This eliminates the old
  public-with-recipient / private-without-recipient contradiction class by
  construction. Trade-off: a private event emitted with no recipients is
  indistinguishable from a broadcast; that check moves to the game layer (see
  the parked note above).
- **Decision:** recipients stored sorted (`tuple(sorted(...))`) so two
  semantically-equal events serialize byte-identically (invariant #4).
- Tests `tests/engine/test_events.py` updated to the new contract (33; suite
  63/63): blank/duplicate recipient rejection, multi-recipient round-trip,
  construction-level sort canonicalization, corrupt-`recipients` read-back
  (non-list, blank element, non-string element).
- `/sdb-review`: python + test + integrity reviewers all PASS (0 critical).
  Integrity raised 1 High (public ⟺ empty-recipients ambiguity) — resolved as a
  documented design trade-off + the T05/T11 parked note; the game-agnostic core
  cannot own it. python-reviewer's `TypeError` Medium was a misread (verified:
  non-`str` elements already raise `ValueError`). Reports in
  `.reviews/20260518-1225-28231c1-recipients/`.
- Verified: `pytest` 63/63, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T05 — observation routing.

---

## 2026-05-18 — T04: event stream

- Added `src/social_deduction_bench/engine/events.py` — the append-only event
  log + JSONL serialization:
  - `Visibility` (`StrEnum`, PUBLIC/PRIVATE) — invariant #2's routing marker.
  - `Event` (`frozen`, `slots`): `seq`, `round`, `phase`, `type` (opaque str),
    `payload`, `visibility`, `recipient`. `__post_init__` enforces
    `PRIVATE ⟺ recipient`, copies `payload` into a `MappingProxyType`, and
    rejects non-JSON-primitive payload values so the round-trip stays lossless.
  - `StreamHeader` (`frozen`): `seed`, `game_id`, `players` roster.
  - `EventLog` — mutable append-only class; `append()` assigns `seq`, `events`
    returns a read-only tuple snapshot, no remove/clear/insert.
  - `EventStream` (`frozen`): header + log; `to_jsonl_lines` /
    `from_jsonl_lines` (fail-loud on corruption); `write_jsonl` / `read_jsonl`.
  Exported from `engine/__init__.py`.
- Tests `tests/engine/test_events.py` (30): event frozen/immutable payload,
  routing-marker rejection, contiguous `seq`, append-only view, header
  seed/roster, lossless JSONL round-trip (incl. enums, empty log),
  byte-identical serialization, and fail-loud read-back (non-contiguous /
  out-of-order `seq`, missing recipient, unknown phase/visibility, non-int
  `seq`, malformed JSON line, malformed player pair, empty input). Suite 60/60.
- **Decision:** no Pydantic — frozen dataclasses + explicit `to_json_dict` /
  `from_json_dict`, consistent with `state.py`/`rng.py`, keeps the engine core
  dependency-light.
- **Decision:** `EventLog` is the deliberate exception to the engine's
  frozen-everywhere style — invariant #5 is "append-only", and a frozen log
  forces O(n²) rebuilds. Safety property is append-only (no remove/edit), not
  immutability; individual `Event`s stay frozen.
- **Decision:** `event.type` is an opaque `str`, not an enum — concrete event
  vocabularies belong to T06/T11/T12 and other games; the engine core stays
  game-agnostic. Same rationale as the plain-`str` role in T03.
- **Decision (user, plan):** whole-stream `write_jsonl` only — no incremental
  streaming writer yet (`to_jsonl_lines` is line-oriented so live-append drops
  in later). `game_id` is a caller-supplied `str` (no `uuid4`). Model-name
  header metadata deferred to a T27-era extension.
- `/sdb-review`: python-reviewer NEEDS FIXES (1 High), test-reviewer PASS,
  integrity-reviewer PASS. Fixed all 6 consolidated findings — High:
  reject non-JSON payload values (tuple→list round-trip lossiness); Mediums:
  `seq`/`round` type-check on read-back, line-located error on malformed JSON,
  strengthened the byte-identical test, added malformed-`visibility` +
  empty-log tests; Low: player-pair unpack guard. Reports in
  `.reviews/20260518-1022-b9a766d/`.
- Verified: `pytest` 60/60, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T05 — observation routing.

---

## 2026-05-18 — T03: engine game-state model

- Added `src/social_deduction_bench/engine/state.py` — `Phase` (`StrEnum`,
  NIGHT/DAY), `PlayerState` and `GameState` (`frozen=True, slots=True`
  dataclasses). Pure query helpers (`player`, `alive_players`, `alive_names`,
  `is_alive`) and pure derivation helpers (`with_player_killed`, `with_phase`,
  `advanced_round`) that return new instances. `GameState.initial()` classmethod
  builds the canonical start (all alive, round 0, NIGHT). Exported from
  `engine/__init__.py`.
- Tests `tests/engine/test_state.py` (16): start contract, duplicate-name
  rejection, empty-roster acceptance, frozen `GameState`/`PlayerState`, tuple
  container, derivation purity, idempotent kill, target-only kill, `KeyError`
  on unknown lookup/kill target, alive filtering, structural equality.
- **Decision:** role is a plain `str` — the engine stays game-agnostic; the
  Werewolf role set is defined in T09. No `engine → games` import.
- **Decision:** `Phase` is NIGHT/DAY only — no setup/terminal sentinel. "Setup"
  is the `initial()` output; "terminal" is a game property answered by T06 +
  M2 win conditions, not a phase.
- **Decision:** `GameState` does not carry the seed — that lives on `GameRNG`;
  T04 records it once in the event-stream header.
- **Decision:** `initial()` takes a `Sequence[(name, role)]`, not a `Mapping`,
  so duplicate names are visible and rejected with `ValueError` (a duplicate is
  a private-event misrouting / hidden-state-leak vector).
- `/sdb-review`: all 3 reviewers PASS, 0 critical / 0 high. Addressed 2 Medium
  test-coverage gaps (missing unknown-name kill test; strengthened idempotent
  test to full structural equality) and 1 Low (`with_player_killed` reuses
  `player()` for its existence check). Reports in
  `.reviews/20260518-0829-e4e91ab/`.
- Verified: `pytest` 30/30, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T04 — event stream.

---

## 2026-05-16 — T02: seeded RNG utility

- Added `src/social_deduction_bench/engine/rng.py` — `GameRNG`, a seed-owned
  random stream wrapping a private `random.Random`. Operations: `shuffle`
  (returns a new list, non-mutating), `choice`, `sample`. PEP 695 generics.
  Exported from `engine/__init__.py`.
- Tests `tests/engine/test_rng.py` (8): same-seed determinism, seed divergence,
  shuffle purity, no global-`random` state leak, instance independence, seed
  exposure, seed read-only, and a golden-literal sequence for seed 42.
- **Decision:** `seed` is a read-only property — a reassignable seed could
  silently desync replay from the recorded event stream (review High item).
- **Decision:** golden-literal test pins seed-42 draws so a future interpreter
  / algorithm change that breaks replay fails loudly (review Medium item).
- **Decision:** kept a single stream with only 3 ops — no checkpoint/restore,
  no labelled sub-streams. Both are speculative until a caller exists.
  - **Parked for T04/T05:** replay will need RNG stream checkpoint/restore.
  - **Parked for `games/werewolf`:** decide single-stream vs labelled
    sub-streams before the game module consumes `GameRNG`.
- `/sdb-review`: all 3 reviewers PASS, 0 critical. Reports in
  `.reviews/20260516-1603-f0a953f/`.
- Verified: `pytest` 14/14, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors, 1 intentional `# type: ignore` on the
  read-only-property assignment test).

**Next:** T03 — game state model.

---

## 2026-05-16 — T01: project scaffold

- Created `pyproject.toml` (Poetry 2.x / PEP 621), `src/social_deduction_bench/`
  with `engine/ games/ agents/ rating/` subpackages, `tests/`, `.gitignore`.
- Deps: `dspy ^3.2.1`, `litellm ^1.84.0`, `trueskill ^0.4.5`; dev: `pytest`,
  `ruff`, `pyrefly`, `pre-commit`.
- `[tool.ruff]` line-length 120, py313, rules `E F I LOG UP T10 ISC ICN G PIE
  PT Q RSE FURB RUF`; `[tool.pyrefly]` over `src/` + `tests/`.
- **Decision:** `requires-python` pinned to `>=3.13,<3.14`. Intended `^3.13`,
  but `litellm` caps at `<3.14` and `dspy` at `<3.15` — the tighter bound wins.
- **Decision:** `.pre-commit-config.yaml` runs ruff check / ruff format /
  pyrefly via `poetry run` (local hooks). No `pytest` hook — tests stay off the
  commit gate. Fresh clones must run `poetry run pre-commit install`.
- **Repo structure fix:** `social-benchmarks/` was sitting inside a
  home-directory-wide git repo (`/Users/pavluhin/.git`), which breaks the sdb
  workflow. Initialized a standalone repo here (`git init -b main`); the
  home-dir `.git` is untouched.
- Verified: `pytest` 6/6, `ruff check`, `ruff format --check`, `pyrefly check`
  all green.

**Next:** T02 — seeded `Random` threaded explicitly through the engine.

---

## 2026-05-16 — Project setup

- Wrote `CLAUDE.md` (adapted from KVARK): behavioral rules, TDD workflow, benchmark invariants, session loop.
- Created the `.claude/` review fleet: `sdb-orchestrator`, `sdb-consolidator`, `sdb-python-reviewer`, `sdb-test-reviewer`, `sdb-benchmark-integrity-reviewer`, `sdb-tdd-implementer`, plus `/sdb-review` and per-agent passthrough commands.
- **Decision:** code quality = Ruff + Pyrefly only. No standalone isort (Ruff's `I` rule), no mypy/pyright.
- **Decision:** package name `social_deduction_bench`; project "Social Deduction Benchmark".
- Set up file-based task management: `BACKLOG.md` (28 tasks, 5 milestones), this `PROGRESS.md`, and the `/sdb-plan` + `/sdb-next` skills.
- Authoritative game spec: `social-deduction-benchmarks/WEREWOLF_DESIGN.md` (research-only folder).

**Next:** start T01.
