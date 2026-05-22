# Progress Log — Social Deduction Benchmark

Append-only. Newest entry on top. Read this first when starting a session.

**Current state:** T25 done — the deceiver-vs-detector split metric
(WEREWOLF_DESIGN §10) is in `games/werewolf/metrics.py`:
`deceiver_detector_split(games)` pairs werewolf win rate (deceiver) with
villager exile accuracy (detector), and `GameMetrics` now carries per-game
`exiles_total` / `exiles_correct` / `exile_accuracy`. Exile accuracy is
read purely from `EXILE_RESOLVED` + the roster header (invariant #1);
ties / no-exile are excluded; a zero-exile game has `exile_accuracy=None`
(not 0.0). Built on T24's metric layer (`extract_game_metrics` /
`aggregate_metrics` / `extract_run_dir` + `rating/manifest.py`
`RunManifest`) and the faction-split CLI cross-play
(`--werewolf-model` / `--villager-model`). 722/722 default suite green,
ruff + pyrefly clean. **Next task:** T26 (TrueSkill rating — its first
unblocked M6 dep is now satisfied) and/or the static results site (folds
T31 replay + T28 leaderboard). With cross-play seating + the split metric
in place, the 10–20-runs-per-pair sweep is unblocked.

**Note (post-T30, unticked in BACKLOG):** commits `dadfd1e` +
`163a179` landed an agent-play overhaul (CLI `sdb-werewolf`, rich live
printer, finish-free ReAct, bid-budget economy, day reaction round with
`ACCUSATION`/`DEFENSE` events, seeded role assignment). These were
bundled ad-hoc, not as a numbered task; the run data they produced lives
untracked in `games/` (15 uniform-qwen self-play runs, wolves 12/15).

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

## 2026-05-22 — T25: deceiver-vs-detector split metric

- **`games/werewolf/metrics.py`** — the §10 first-class split. Surfaces
  the recurring upstream finding (LLMs deceive better than they detect):
  - Per-game: `GameMetrics` gains `exiles_total`, `exiles_correct`,
    `exile_accuracy: float | None`. `_exile_accuracy(events)` walks
    `EXILE_RESOLVED` events, counts a resolved exile (skips `exiled=None`
    ties — indecision is not a detection failure), and marks it correct
    when the exiled seat's role is a werewolf (`faction_of` over the
    roster header). Outcome-sourced from the event log only (invariant
    #1). Zero resolved exiles → `exile_accuracy=None` (a real None, NOT
    0.0 — no decision to judge vs. "exiled and missed").
  - Cross-game: `deceiver_detector_split(games) -> DeceiverDetectorReport`
    with per-model `FactionSplit` rows. Deceiver = game-level wolf win
    rate attributed to the unique werewolf-faction model; detector =
    village exile accuracy attributed to the unique village-faction model.
    `exile_accuracy` POOLS across exile decisions (Σcorrect / Σtotal),
    deliberately unlike `aggregate_metrics.mean_illegal_move_rate`
    (mean-of-per-game-rates) — each exile is the unit of detection, so a
    multi-exile game must outweigh a single-exile one. A pooling-guard
    test pins 1/3 (not the 0.25 a mean-of-rates would give).
  - `_unique_faction_model(seats, faction)` returns the lone model
    staffing a faction, else None (mixed faction → skip per-model
    attribution but still count the population pool).
- **Decision (review fix):** an unresolved seat model is the `_UNKNOWN_MODEL`
  sentinel; a faction staffed entirely by it is treated as unattributable
  (None), so post-hoc runs without a manifest never emit a bogus "unknown"
  leaderboard row — they still count toward the population pool. (Consensus
  Medium from all three reviewers; covered by a new test.)
- **`/sdb-review`**: python + test + benchmark-integrity reviewers all PASS
  (0 critical / 0 high). The one consensus Medium ("unknown"-model
  attribution) was fixed before commit.
- Verified: `pytest -q` 722 passed + 1 deselected (real-LLM smoke
  auto-skips); ruff + ruff format + pyrefly all clean. Pure metric only —
  no TrueSkill (T26), tournament (T27), or UI (T28/T31) surface yet.

---

## 2026-05-22 — Per-seat models: faction-split CLI cross-play

- **`cli.py`** — `sdb-werewolf` gained a second seating mode so runs are
  no longer forced self-play (the adapter already did per-seat LMs via
  T22; the gap was purely the CLI copying one `--model` to all seats):
  - `--model X` — uniform (unchanged).
  - `--werewolf-model X --villager-model Y` — faction split: werewolf
    seats play X, the village faction (villager/seer/doctor) plays Y.
  - New pure helper `_resolve_seat_models(roster, *, model,
    werewolf_model, villager_model)` maps each seat to its model by
    `faction_of(role)` over the **seed-dealt** roster, so the wolf-model
    follows whichever seats the seed made wolves — fair, deterministic,
    not seat-name-pinned (invariant #4). Exactly one faction flag set
    fails loud (`ValueError` → clean `SystemExit(2)` + stderr), BEFORE
    the API-key check, so a half-specified mode never silently seats a
    default model on the missing faction.
  - `_model_arg_summary(args)` feeds the header + the T24 manifest's
    `model_arg`; `_build_manifest` now takes the resolved `seat_models`
    map so `manifest.json > models` records the true per-seat models
    (or the `"scripted"` sentinel for `--dry-run`, which seats no LMs
    even if faction flags are passed). No metrics/manifest module change
    — they already consume the per-seat `models` map.
- **Decision (user):** keep it to two modes for now — uniform and a
  two-model faction split (deceivers vs detectors). 3+ models /
  count-based or positional seating / per-seat sampling deferred to the
  tournament runner (T27), which will seat models programmatically.
- **`/sdb-review`** (`.reviews/...-faction`): python + integrity
  reviewers PASS; test reviewer 1 High — the bad-combo test deleted the
  API key and asserted only exit 2, which the missing-key path also
  produces, so it couldn't prove the faction error fires first. Fixed:
  set the key + assert the faction stderr message (proves ordering).
  Low (stale module docstring) fixed too.
- Verified: `pytest -q` 712 passed + 1 deselected; ruff + ruff format +
  pyrefly clean; manual `--dry-run` faction smoke records `"scripted"`;
  half-specified `--werewolf-model` alone exits 2 with the guidance.

---

## 2026-05-22 — T24: metric extraction + run manifest (M6 begins)

- **New `rating/manifest.py`** — `RunManifest` (frozen, slots): the
  game-agnostic per-run provenance record the leaderboard/TrueSkill
  (T26+) needs. Fields: `game_id, seed, players, models` (seat→model,
  stored sorted), `model_arg, temperature, max_tokens, max_iters,
  reasoning, git_sha, created_at, winner, rounds`. `to_json_dict` /
  `from_json_dict` (fail-loud: missing key → `KeyError`, malformed pair
  → `ValueError`), `write_json` / `read_json` (sorted keys; malformed
  file wrapped to a stable `"malformed manifest JSON"` `ValueError`,
  mirroring the engine/trajectory readers). Solves the seat→model gap:
  model identity previously lived only inside `trajectories.jsonl >
  lm_calls[].model`, and a seat that never calls an LM had none.
- **New `games/werewolf/metrics.py`** — werewolf-aware extraction
  (placed here, not `rating/`, because it depends on roles/factions and
  the game-action tool set; `rating/` stays for game-agnostic TrueSkill).
  - `SeatMetrics` / `GameMetrics` / `RoleStats` / `ModelStats` /
    `AggregateMetrics` frozen value types (computed rates stored as
    fields, no `@property`, so `dataclasses.asdict` + equality work).
  - `extract_game_metrics(events, trajectories, manifest=None)`:
    winner/rounds/deaths from the **event log only** (invariant #1);
    per-seat `won` via `faction_of(role).value == winner`, `survived`
    from `KILL_RESOLVED`/`EXILE_RESOLVED` (null-safe); tokens/tool-calls
    from trajectories grouped by `caller`; model from the manifest when
    present else inferred from the seat's first `lm_call` else
    `"unknown"`. Game-action tool set = `set(WEREWOLF_TOOL_REQUIREMENTS)`
    (reused, not redefined); `tool_usage` = successful (`"ok:"`)
    game-action steps, sorted.
  - `aggregate_metrics` rolls up over **seat-games** bucketed by model
    then role (per-model/per-role win rate, token totals); sorted output.
  - `extract_run_dir(path)` reads the `events`/`trajectories`/(optional
    `manifest`) triplet — the backfill path for the existing
    manifest-less `games/` runs.
- **CLI** (`cli.py`): builds a `RunManifest` after each run and writes
  `manifest.json` as the 4th artifact (`_build_manifest`, `_git_sha`
  via `git rev-parse HEAD` → `None` outside a checkout, `created_at` =
  UTC now). `--dry-run` records the sentinel model `"scripted"`.
  `printer.print_summary` gained an optional `manifest.json` row. The
  manifest's `git_sha`/`created_at` are run bookkeeping, NOT game logic
  (consistent with the existing UTC `game_id`); the event/trajectory
  determinism contracts are untouched.
- **Decisions (user, this session):** static pre-aggregated JSON +
  zero-build vanilla front-end for the eventual results site (no DB,
  no CRUD); add `manifest.json` going forward and infer-backfill the
  existing runs; start with the metric layer before any UI.
- **`/sdb-review`** (`.reviews/20260522-1511-163a179-worktree/`):
  python + test reviewers PASS; integrity reviewer NEEDS FIXES (0
  critical / 1 high). All addressed before commit:
  - **HIGH** — illegal-move count must equal the engine's
    `TOOL_REJECTED` events, but a tool that *raises* during execution
    yields an `"error:"` observation with no rejection event (a tool
    fault, not an illegal move). Switched `total_illegal_moves` to be
    sourced from `TOOL_REJECTED` events (per-seat via recipients), not
    trajectory `"error:"` prefixes. Regression test adds an
    execution-error step and pins that it does NOT inflate the count.
  - **Mediums** — tightened `dict[str, list[Trajectory]]`; documented
    `mean_illegal_move_rate` as a mean-of-per-game-rates (pooled rate
    derivable from `ModelStats`); added a fail-capable invariant-#1 test
    (a manifest that lies about winner/rounds is ignored); made the
    `tool_usage` + aggregate-model `sorted(...)` load-bearing under test
    (keys fed out of order, exact ordered-tuple assertions).
  - **Lows** — `_git_sha` docstring (full sha); aligned the winner
    convention (metrics uses first/`next` `GAME_OVER`, like the CLI).
- Verified: `pytest -q` 702 passed + 1 deselected; `ruff check`, `ruff
  format --check`, `pyrefly check` (0 errors); `sdb-werewolf --dry-run`
  writes `manifest.json`; `extract_run_dir` on a real manifest-less run
  yields real numbers (backfill path).

**M6 (Rating, metrics & tournament) progress:** T24 done. T25, T26,
T27, T28 remain. T31 (replay UI, M5) also still open.

**Next:** T25 (deceiver-vs-detector split) or the static results site.

---

## 2026-05-21 — T30: per-decision trajectory + LM metadata sidecar (M5 progress)

- **New module `agents/trajectory.py`** — the sidecar's value types,
  mirroring `engine/events.py`'s shape so the two files share their
  integrity story:
  - `ReActStep(frozen, slots)` — `iter, thought, tool, args,
    observation`. `args` is wrapped in `MappingProxyType` at
    construction (post-review High fix) so the audit step is genuinely
    immutable, not just "frozen with a mutable inner dict."
  - `LMCallRecord(frozen, slots)` — `model, prompt_tokens,
    completion_tokens, latency_ms, cost_usd`. No `cached` field —
    `cost_usd is None` already conveys cache-hit-or-non-billable; T31
    can render accordingly.
  - `Trajectory(frozen, slots)` — `decision_seq, round, phase, caller,
    role, terminal_tool, committed_value, react_trajectory, lm_calls`.
    `phase` stored as the `Phase.value` string for JSON-friendliness.
  - `TrajectoryStream(frozen)` — `header: StreamHeader, trajectories:
    tuple[Trajectory, ...]`. Reuses the engine's `StreamHeader` so the
    sidecar is self-contained (seed + game_id + roster pinned at the
    top of its own file). Mirror methods to `EventStream`:
    `to_jsonl_lines` (sort_keys=True), `from_jsonl_lines` (fail-loud
    on malformed JSON, gapped `decision_seq`, missing header),
    module-level `write_jsonl` / `read_jsonl`.
  - `sanitize_arg` (hoisted from `decisions.py`) — single sanitizer
    used by both the `TOOL_REJECTED` draft path and the new sidecar
    value types.
- **`agents/react.py`** — `trace_sink: TraceSink | None = None`
  callback added to `react_decide`. Fires once after a successful
  commit with `(steps: tuple[ReActStep, ...], lm_calls:
  tuple[LMCallRecord, ...])`. On `RuntimeError` (no commit) the sink
  is not called — one trajectory == one committed decision. The
  per-iteration projection captures:
  - **Steps**: thought / tool / args / observation, with `args`
    sanitized + frozen via `MappingProxyType`.
  - **LM calls**: built from `lm.history` entries appended during the
    iteration, paired by **uuid-marker scheme** (post-review High
    fix). The original `len(lm.history)`-slice approach was fragile
    against DSPy's bounded `settings.max_history_size` rotation and
    silently dropped under `disable_history`; the uuid-set comparison
    is robust to both.
  - **Latency**: `time.monotonic()` delta around `react.react(...)`
    divided across the iteration's new history entries (almost always
    one; multi-entry warns via `logger.warning`).
  - **Token coercion**: a new `_coerce_int` helper (post-review High
    fix) downgrades non-numeric provider responses (`None`, missing,
    bool, string) to `0`. Telemetry is observability, never a
    kill-switch — a quirky provider response must not abort a game
    mid-decision.
- **`agents/decisions.py`** — `ReActDecisionSource` accumulates
  `Trajectory` records in `self._trajectories` (private list);
  exposed via the new `trajectories` read-only property as a snapshot
  tuple. `_invoke_react` builds the sink closure once per call and
  appends the assembled `Trajectory` with `decision_seq` (gap-free,
  starting at 0), `round` / `phase` captured at decision time. The
  local `_sanitize_arg` was removed; both call sites (`TOOL_REJECTED`
  draft and sidecar value types) now use the shared
  `agents.trajectory.sanitize_arg`. `_role_by_name` indexed once at
  `__init__` for O(1) role lookup during trajectory assembly.
- **`agents/__init__.py`** — re-exports `LMCallRecord`, `ReActStep`,
  `Trajectory`, `TrajectoryStream` so consumers (T31 visualizer, T24
  metrics) import from `social_deduction_bench.agents` directly.
- **Decisions (user, plan):**
  - **Adapter property + caller-wires-IO**, not `run_game`-extended.
    `run_game` keeps returning `EventStream` unchanged; the caller
    reads `source.trajectories` and wraps in `TrajectoryStream(...)`.
    The scripted `DecisionSource` Protocol stays the same — only
    agent-backed sources have trajectories.
  - **No `cached` field on `LMCallRecord`.** `cost_usd is None`
    encodes cache-hit-or-non-billable; adding a separate flag would
    need a `model == "dummy"` carve-out and conflate the two cases
    anyway.
  - **Sidecar is agent-side, NOT part of determinism.** The engine's
    `EventStream` determinism contract is unaffected.
    `assert_streams_identical` continues to compare only event
    streams. Wall-clock fields (`latency_ms`, `cost_usd`) vary
    run-to-run even with the same seed — by design.
  - **Sink does NOT fire on failed loops.** Keeps the invariant "one
    sidecar line per committed decision" simple. T24's illegal-move
    metric will count failures via `TOOL_REJECTED` events and the
    loop's `RuntimeError` boundary, not via partial trajectories.
- **Test surface added:**
  - `tests/agents/test_trajectory.py` (new, 22 cases) — value-type
    frozen + structurally equatable, JSON round-trip, JSONL stream
    round-trip, header-first / one-line-per-trajectory invariants,
    empty-trajectories handling, byte-identical serialization for
    equal streams, `write_jsonl` / `read_jsonl` through tmp_path,
    fail-loud reads (empty input, malformed JSON, gapped
    `decision_seq`, out-of-order `decision_seq`), sanitization at
    construction, latency type/sign pin.
  - `tests/agents/test_react.py` (+9 cases) — `trace_sink` default
    no-op back-compat, sink fires once with structured steps + LM
    calls, sink does NOT fire on failed commit, rejection observation
    lands in the same trajectory as the eventual success, the new
    `_coerce_int` helper handles `None`/`bool`/`str`/`float`,
    `_lm_calls_for_iteration` warns on empty + multi-entry slices,
    **bounded-history-rotation regression** via a `_RotatingLM` that
    pops `lm.history` between calls (the test would fail under the
    pre-fix `len(lm.history)` slicing), `ReActStep.args` immutability
    pin (`step.args["k"] = ...` raises `TypeError`).
  - `tests/agents/test_decisions.py` (+9 cases) — empty
    `trajectories` on construction, one trajectory per acting living
    player at night, `terminal_tool` / `role` / `phase` / `round`
    correctness, `committed_value` matches the terminal's parsed
    value, `lm_calls` non-empty per loop, werewolf-chat intermediate
    lands BEFORE the kill terminal in the same trajectory (chat-then-
    kill is one trajectory, not two), rejected-terminal shows as
    `error:` step in the same trajectory whose retry committed,
    `decision_seq` gap-free across a full `run_game`, snapshot
    semantics (mutating returned tuple does not affect later growth),
    bid + speech loops each contribute one trajectory each.
  - `tests/agents/test_smoke_game.py` (+1 assertion under the
    existing `pytest.mark.smoke`) — after `run_game`, the source's
    `trajectories` is non-empty,
    `TrajectoryStream(stream.header, source.trajectories)` round-trips
    through `to_jsonl_lines` → `from_jsonl_lines` losslessly, and
    every `LMCallRecord.latency_ms` is a non-negative float.
- `/sdb-review`: python reviewer **NEEDS FIXES** (0 critical / 3 high
  / 4 medium); test + integrity reviewers **PASS** (0 critical / 0
  high). All three Highs addressed before commit:
  - **HIGH (python #1, integrity-medium #1)** — `int(usage.get(...))`
    on `None` / non-numeric provider response was a crash-the-game
    hazard. Replaced with `_coerce_int` helper that downgrades to `0`.
  - **HIGH (python #2)** — `len(lm.history)` slicing was fragile under
    DSPy's bounded-history rotation and broken under
    `settings.disable_history`. Replaced with a uuid-marker set:
    snapshot pre-iteration uuids, slice on entries whose uuid wasn't
    in the snapshot. Regression test installs a `_RotatingLM` that
    pops history between iterations.
  - **HIGH (python #3)** — `ReActStep.args` was a plain `dict`,
    leaving the "frozen audit trail" mutable. Wrapped in
    `MappingProxyType` at `__post_init__`. Pin test:
    `step.args["k"] = ...` raises `TypeError`.
  - **Mediums addressed**: warning logs on empty / multi-entry LM
    history slices; per-iteration `len(steps) == len(calls)` symmetry
    assertion in the trace-sink test; dropped the redundant `tuple(...)`
    cast in `_invoke_react` (Trajectory.__post_init__ normalizes
    once). The `from_json_dict` `KeyError`-vs-`ValueError`
    consistency item left as-is to match the engine's existing
    `Event.from_json_dict` convention (CLAUDE.md rule 10).
  - **Low items** (cosmetic — TYPE_CHECKING moves, `dict(...)` outer
    copy in step construction) — bundled into the High #2 fix path
    (the outer copy was dropped when wiring the new arg-passing
    shape).
  - Reports in `.reviews/20260521-0805-c28f9ac-T30/`.
- Verified: `pytest -q` 498 passed + 1 deselected; `pytest -m smoke -q`
  (no key) 1 skipped + 498 deselected; `ruff check`, `ruff format
  --check`, `pyrefly check` (0 errors).

**M5 (Observability & analysis) progress:** T29 + T30 done. T31
(replay UI) remains.

**Next:** T31 — Replay UI for audience and behavior analysis.
Standalone HTML viewer consuming `events.jsonl` +
`trajectories.jsonl`, joined by `(round, phase, caller,
decision_seq)`. Audience-facing.

---

## 2026-05-21 — T29: dialogue, ballot attribution, rejected-call events (M5 begins)

- **Engine event vocabulary** (`games/werewolf/events.py`,
  `games/werewolf/config.py`):
  - Added five new event-type constants. `DISCUSSION_RESOLVED` and
    `SPEECH` are public; `BID`, `TOOL_REJECTED`, and `KILL_BALLOTS` are
    private and bound into `PRIVATE_EVENT_TYPES` so the engine guard
    rejects each one when emitted with empty recipients.
  - `KILL_BALLOTS` did not appear in the original T29 plan — the
    integrity reviewer flagged the originally-planned
    `KILL_RESOLVED.payload["ballots"]` shape as a Critical leak (kill
    ballots are werewolves' private coordination data, but
    `KILL_RESOLVED` is broadcast to the whole village). The fix was to
    split into two events: `KILL_RESOLVED` (public, `{"victim"}` only,
    pre-T29 shape) and `KILL_BALLOTS` (private, recipients = sorted
    living werewolf pack, `{"ballots"}`). `EXILE_RESOLVED.payload["ballots"]`
    stays public — exile votes are public by design.
- **`DecisionSource` Protocol** (`games/werewolf/loop.py`): three new
  methods on top of `night_actions` / `day_actions` / `observe`:
  - `bids(state) -> dict[str, int]` — one ReAct loop per alive player
    in the agent-backed source; agent-side stages one `BID` draft per
    bidder, private to that bidder.
  - `speeches(state, speakers) -> tuple[(speaker, message), ...]` —
    one ReAct loop per chosen speaker; stages one public `SPEECH`
    draft per commit. Driver cross-checks the returned order against
    the resolver's `discussion.speakers` (must be a prefix) — a
    divergent script fails loud rather than logging a contradictory
    transcript.
  - `drain_drafts() -> tuple[EventDraft, ...]` — drains any staged
    drafts (chat / bid / speech / tool-rejected) since the last drain.
    The driver drains after each agent-facing step and routes the
    drafts through the same `_log_drafts` + private-event guard as the
    resolvers' drafts.
- **`react.py` intermediate-tool category**: new `intermediate_tools`
  mapping on `react_decide` for game-action tools that emit a side
  effect but do **not** terminate the loop (only `werewolf_chat`
  uses it today — wolves can chat zero-to-N times before committing a
  kill vote). New `on_reject(tool, args, reason)` callback fires once
  per `ToolResult(valid=False)` from either category; default `None`
  is a quiet no-op (back-compat with the M4 tests).
- **`ReActDecisionSource` wiring** (`agents/decisions.py`):
  - Per-player ReAct loops drive `submit_bid`, `speak`, and the
    werewolf-chat intermediate. Sub-helper `_invoke_react` centralises
    the (cognitive tools, intermediates, terminals, on_reject) build.
  - `on_reject` callbacks are **cached per caller** in `__init__`
    (`self._on_reject_by_caller`) — reviewer caught that the previous
    code built a fresh closure per `_invoke_react` call. Each
    callback also sanitises the LLM's tool-call kwargs through
    `_sanitize_arg` before staging the `TOOL_REJECTED` payload — a
    non-JSON value from the LLM would otherwise crash
    `Event.__post_init__` in mid-rejection.
  - `_bind_werewolf_chat` builds the intermediate wrapper that stages
    `WEREWOLF_CHAT` drafts to the **currently-living** werewolf pack
    on every valid call. The living pack is computed once at the top
    of `night_actions` (it cannot change during a single night —
    no kills resolve until `resolve_night`).
- **`run_game` loop** (`games/werewolf/loop.py`): two new helpers
  `_run_night` and `_run_day` keep the loop body to a thin
  termination + advance-phase wrapper (reviewer high finding).
  `DISCUSSION_RESOLVED` is now built as an `EventDraft` and routed
  through `_log_drafts` like every other event — the previous
  inline `log.append` bypassed the recipients guard.
- **`ScriptedDecisions`** (`games/werewolf/scripted.py`): three
  default-empty staging fields — `night_chats`, `day_bids`,
  `day_speeches` — let a fixed script supply chat lines / bids /
  speeches without breaking the legacy `nights=` + `days=`-only
  construction. Staged drafts flush through `drain_drafts()`.
- **Decisions (review-driven):**
  - **Three reviewers ran in parallel.** python: `NEEDS FIXES` (0
    critical / 4 high / 4 medium). test: `PASS` (0 / 0 / 7 medium —
    advisory). integrity: `NEEDS FIXES` (1 critical — stream stalled
    at 600s but the critical was captured before the stall).
  - **All Critical + High items addressed before commit.** Mediums
    addressed except for Low cosmetic items (per CLAUDE.md rule 2).
  - **`speeches` cross-check is prefix-based, not strict equality.**
    A script may supply fewer than `K_DISCUSSION_SLOTS` speeches
    (silent slots), but their order must match the start of the
    resolver's chosen speakers. A full divergence is a caller bug
    and raises `RuntimeError`.
- **Test surface added:**
  - `tests/games/werewolf/test_event_types.py` — parametrized
    private-event-guard test now covers all six private types.
  - `tests/games/werewolf/test_night.py` — `KILL_BALLOTS` is private
    to the living pack (incl. dead-wolf exclusion); `KILL_RESOLVED`
    carries `{"victim"}` only.
  - `tests/games/werewolf/test_day.py` — `EXILE_RESOLVED.ballots`
    payload, including all-abstain case.
  - `tests/games/werewolf/test_scripted.py` (new) — 7 cases pinning
    `ScriptedDecisions` chat / bid / speech staging through
    `drain_drafts()`.
  - `tests/agents/test_react.py` — `intermediate_tools` does not
    terminate; `on_reject` fires for both categories; default `None`
    is a quiet no-op.
  - `tests/agents/test_decisions.py` — per-seat scripts now include
    bids + speeches; new cases pin werewolf-chat intermediate, dead-
    werewolf exclusion from chat recipients, per-seat BID drafts with
    distinct amounts, speech-order pinning, `TOOL_REJECTED` draft
    shape, `drain_drafts` idempotency, init-time emptiness; new
    end-to-end test drives a self-target rejection through `run_game`
    and asserts the `TOOL_REJECTED` event lands private to the
    caller.
  - `tests/games/werewolf/test_game_loop.py` — dialogue-rich scripted
    game pins all five new event types end-to-end, JSONL round-trip,
    determinism, the villager observation closure, and the
    `KILL_BALLOTS` recipients (incl. a dedicated dead-werewolf-not-
    a-recipient test).
- `/sdb-review`: see `.reviews/20260520-2202-6818a74-T29/` for the
  three reviewer reports + consolidated triage. Verdict: **all
  Critical + High items resolved before commit.**
- Verified: `pytest -q` 458 passed + 1 deselected; `pytest -m smoke -q`
  (no key) 1 skipped + 458 deselected; `ruff check`, `ruff format
  --check`, `pyrefly check` (0 errors).

**M5 (Observability & analysis) progress:** T29 done. T30 (per-decision
trajectory + LM metadata sidecar) and T31 (replay UI) remain.

**Next:** T30 — per-decision trajectory + LM metadata sidecar
`<game_id>.trajectories.jsonl` written next to `events.jsonl`.

---

## 2026-05-20 — T23: real-LLM smoke game via OpenRouter (M4 close)

- Added the first end-to-end test that drives a real LLM through the
  full Werewolf engine + adapter loop:
  - `src/social_deduction_bench/settings.py` (new) — the single home
    for `os.getenv` reads in the project (CLAUDE.md "single settings
    module" rule). Frozen `Settings(openrouter_api_key, smoke_model)`
    dataclass + `load()` factory. `load()` re-reads env on every call
    so tests can `monkeypatch.setenv` and observe the change without
    reload tricks.
  - `tests/test_settings.py` (new, 5 cases) — env-unset → both None;
    `OPENROUTER_API_KEY` read pinned by string (catches a rename);
    `SDB_SMOKE_MODEL` read pinned; frozen instance rejects mutation;
    review-driven "no extra fields" pin asserts the field set is
    exactly `{openrouter_api_key, smoke_model}` (a new field added
    without a matching `os.getenv` in `load()` would default to None
    forever — caught here).
  - `tests/agents/test_smoke_game.py` (new, 1 case) — file-level
    `pytestmark = pytest.mark.smoke`; in-test `pytest.skip` when
    `OPENROUTER_API_KEY` is absent. Builds 7 uniform
    `dspy.LM("openrouter/{model}", api_key=..., temperature=0.0,
    cache=False, max_tokens=512)` seats, runs `run_game(ROSTER,
    seed=42, decisions=source, max_rounds=4)`, asserts:
    `events[-1].type == GAME_OVER`; every event whose type is in
    `PRIVATE_EVENT_TYPES` has non-empty `recipients` (invariant #2
    end-to-end under real-LLM traffic); JSONL round-trip via
    `EventStream.from_jsonl_lines` (invariant #5).
  - `pyproject.toml` — extended `[tool.pytest.ini_options]`:
    `markers = ["smoke: real-LLM smoke games (cost money; opt in
    with -m smoke)"]` and `addopts = "-m 'not smoke'"`. Default
    `poetry run pytest` does not collect the smoke test; explicit
    `poetry run pytest -m smoke` is required.
- **Decisions (user, plan):**
  - **Two-lock cost gate**: pytest marker (default-deselected) AND
    in-test `OPENROUTER_API_KEY` skip. A globally-set key on a
    developer's machine cannot accidentally burn money on a plain
    `poetry run pytest`.
  - **`qwen/qwen3.5-9b` as default seat model** (user choice;
    hardcoded verbatim per the "use the user's literal value" rule).
    `SDB_SMOKE_MODEL` env var overrides.
  - **Uniform single model across all 7 seats** in T23. Per-seat
    cross-play is already pinned by T22's unit tests; T23's job is
    to prove the loop survives a real LLM round-trip.
  - **No determinism assertion.** Real LLMs do not replay
    byte-identically without an exact prompt-cache hit; pinning
    determinism here would make the test flaky. Engine-side
    determinism is already covered by `assert_deterministic` in
    T08/T22 (with `DummyLM`).
  - **`cache=False` on every LM** (review-driven). DSPy's default
    on-disk cache would let a second run be a 100% cache-hit, making
    the test useless as a real-LLM gate. Every smoke invocation now
    actually hits OpenRouter.
  - **`max_iters=20`** (review-driven). Up from the adapter's default
    of 10; a 9B open model on DSPy's ReAct loop needs more room to
    reliably emit `finish`. A loop that exhausts iters raises
    `RuntimeError` (T21 fail-loud), which would conflate model
    discipline with adapter regressions — bump reduces that surface.
- **Out of scope** (per CLAUDE.md rule 2): no CLI / runnable script,
  no metric extraction (T24), no tournament loop (T27).
- `/sdb-review`: python reviewer **PASS** (0/0/0); test reviewer
  **NEEDS FIXES** (1 critical, 2 high, 3 medium); integrity reviewer
  **NEEDS FIXES** (0 critical, 4 high). All findings addressed
  before commit:
  - **CRITICAL (test) / HIGH (integrity)** — the original recipients
    assertion (`recipients == () or len >= 1`) was a tautology that
    matches every tuple. Replaced with `if event.type in
    PRIVATE_EVENT_TYPES: assert event.recipients` — pins invariant
    #2 at the declared-private event types, end-to-end.
  - **HIGH (test + integrity)** — `KILL_RESOLVED >= 1` was
    structurally guaranteed (`night.py:121` drafts it every night,
    even on doctor-save). Dropped — the `GAME_OVER` terminal
    assertion + `max_rounds`-driven `RuntimeError` already pin
    liveness.
  - **HIGH (test)** — skip message expanded to name both the env
    var to set AND the `-m smoke` invocation.
  - **HIGH (integrity)** — `dspy.LM(cache=False)` so every smoke
    run actually exercises the LLM.
  - **HIGH (integrity)** — `max_iters=20` (was 12) to reduce
    9B-model dead-end false failures.
  - **MEDIUM (test)** — added the "no extra fields" pin on the
    `Settings` dataclass.
  - **NOT applied:** moving `_DEFAULT_MODEL` into `settings` —
    that's test config, not env-var config; the `SDB_SMOKE_MODEL`
    override path through `settings` is what the contract needs.
  Reports in `.reviews/20260520-2026-1c08b9c-T23/`.
- Verified: `pytest -q` 413 passed + 1 deselected; `pytest -m smoke
  -q` (no key) 1 skipped + 413 deselected; `ruff check`, `ruff
  format --check`, `pyrefly check` (0 errors).

**M4 (DSPy ReAct agent + memory) is complete** — T19, T20, T21, T22,
T23 all done. The benchmark can now host real LLMs at every seat,
seated per-roster, behind a hard cost gate.

**Next:** T24 — metric extraction from the event stream (M5 begins).

---

## 2026-05-20 — T22: per-player LM seating in `ReActDecisionSource` (M4 progress)

- Extended the adapter so a single game can seat a mix of LLMs across
  seats — the cross-play setup WEREWOLF_DESIGN.md §10 requires:
  - `src/social_deduction_bench/agents/decisions.py` — `__init__`
    swaps `lm: BaseLM` for `lms: Mapping[str, BaseLM]`; computes
    `missing = roster_names - lm_names` and `extra = lm_names - roster_names`
    and raises `ValueError` naming both sides (sorted, for message
    determinism) on any mismatch. Stores `MappingProxyType(dict(lms))`
    (defensive copy + immutable view). New `lms` read-only property
    symmetric with `memories`. `_run_one` looks up
    `self._lms[caller]` per ReAct loop.
- **Decisions (user, plan):**
  - **Strict `Mapping[str, BaseLM]`, one entry per seat.** No
    `BaseLM | Mapping[...]` union shorthand, no `with_uniform_lm`
    class method, no `Callable[[str], BaseLM]` factory. Each seat
    is explicit; mismatch fails loud (CLAUDE.md rule 11).
  - **Defensive copy on store.** `MappingProxyType(dict(lms))`
    means the caller cannot mutate the source dict post-construction
    and cannot reach in via `source.lms` either.
  - **Sorted error names** (`sorted(missing)`, `sorted(extra)`) so the
    `ValueError` message is byte-identical run-to-run regardless of
    Python set iteration order — small but matches the codebase's
    determinism discipline.
  - **`react_decide` stays single-LM.** Per-seat is purely an adapter
    concern. The loop primitive still takes one `lm`.
- **Test migrations (11 existing tests):** introduced two helpers in
  `tests/agents/test_decisions.py` — `_uniform_lms(lm, roster=ROSTER)`
  (shares one `DummyLM` across all seats, preserving the legacy
  single-cursor answer-queue semantics) and `_empty_lms(roster=ROSTER)`
  (no-act tests). Every `lm=DummyLM(...)` call site became
  `lms=_uniform_lms(DummyLM(...))` or `lms=_empty_lms()`; the swapped-
  roster test threads its own roster through `_uniform_lms`.
- **Tests added (`tests/agents/test_decisions.py`, 9 net new cases):**
  missing-name rejected; extra-name rejected; exact-coverage accepted;
  `lms` read-only mapping (mutation raises `TypeError`); per-seat LM
  isolation in `night_actions` (each acting seat scripted with a
  *distinct* target so a Wolf1↔Wolf2 swap would mis-attribute the
  vote, with `DummyLM.history` length pinned at 2 per acting seat and
  `[]` for villagers); dead seat does not consume its LM during
  `day_actions`; full werewolf-win sweep with per-seat scripted LMs
  reaches `GAME_OVER`; two-source determinism via
  `assert_deterministic`; lms-dict insertion-order does not affect the
  event stream (forward vs. reversed-key dict; both `run_game`
  outputs byte-identical via `assert_streams_identical`). Suite
  408/408.
- `/sdb-review`: python + integrity reviewers **all PASS** (0
  critical, 0 high, 0 medium); test reviewer PASS with 2 advisory
  mediums — both addressed before commit:
  - Per-seat isolation test now uses distinct targets per seat
    (Wolf1 → Vil1, Wolf2 → Vil2, Seer1 → Wolf1, Doc1 → Vil3) so a
    seat-key swap surfaces in `actions.kill_votes` /
    `actions.seer_inspect` / `actions.doctor_protect`; the
    `history`-length assertion alone would have missed it.
  - Added `test_lms_dict_insertion_order_does_not_affect_event_stream`
    — the regression guard that pins iteration through
    `self._roster` (tuple) rather than `self._lms.keys()` (dict
    insertion order). Two runs with the same per-seat mapping but
    forward vs. reversed key order yield byte-identical streams.
  Reports in `.reviews/20260520-1952-1c08b9c-T22/`.
- Verified: `pytest` 408/408, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T23 — smoke game with real LLM agents (auto-skips when no
API key). M4 (DSPy ReAct agent + memory) closes.

---

## 2026-05-20 — T21: DSPy ReAct loop + `DecisionSource` adapter (M4 entry)

- Added the M4 capstone: the per-decision-point ReAct loop primitive that
  turns a seated LLM into a Werewolf player, plus the adapter that wires
  those loops into the existing `run_game` driver:
  - `src/social_deduction_bench/agents/react.py` (new) — `Commit` frozen
    public value type + `react_decide(*, caller, cognitive_tools,
    terminal_tools, decision_brief, lm, max_iters=10) -> Commit`. Drives
    `dspy.ReAct.react` (not `forward`) in a manual loop so each iteration
    is exactly one LM call. Terminal-tool wrappers write `(name, value)`
    into a `_Slot` on a valid `ToolResult`; the LLM then emits `finish` to
    close the loop, and we return the slot as a `Commit`. An empty slot
    after the loop is fail-loud (`RuntimeError` chained from any swallowed
    upstream `ValueError`).
  - `src/social_deduction_bench/agents/decisions.py` (new) —
    `ReActDecisionSource`: per-player `GameMemory` table; roster-order
    iteration; `_NIGHT_TERMINAL_BY_ROLE` table picks `submit_kill_vote` /
    `seer_inspect` / `doctor_protect` for night, `submit_exile_vote` for
    day; `observe(state, new_events)` routes through `observations_for`
    into each *living* player's memory; `memories` exposes a
    `MappingProxyType`.
  - `src/social_deduction_bench/games/werewolf/loop.py` — extended the
    `DecisionSource` Protocol with `observe(state, new_events)`; driver
    calls it after each `_log_drafts` with `log.events[before:]`. The
    terminal `GAME_OVER` event is NOT routed (game is over — documented
    on the Protocol).
  - `src/social_deduction_bench/games/werewolf/scripted.py` and inline
    `_StallingDecisions` in `tests/games/werewolf/test_game_loop.py`
    gained no-op `observe`.
  - `src/social_deduction_bench/games/werewolf/cognitive.py` — one-line
    import-path fix (`agents.memory` instead of `agents` package init) to
    break a circular import surfaced by the new re-exports.
  - `src/social_deduction_bench/agents/__init__.py` — re-exports
    `Commit`, `ReActDecisionSource`, `react_decide`.
- **Decisions (user, plan):**
  - **`dspy.ReAct` with its `finish` terminator** rather than a custom
    loop. Side-band `_Slot` captures the committed value when the
    terminal returns a valid `ToolResult`; the LLM then calls `finish`.
    Empty slot → `RuntimeError`. Trade-off: one extra LM round-trip per
    decision (the `finish` hop) in exchange for not coupling to ReAct's
    private structure.
  - **Drive `react.react` directly, skip the extract step.** Avoids the
    extra LM call `ReAct.forward` would append; the commitment lives in
    the slot, not in the LM's `committed_action` output field.
  - **Loop primitive + `DecisionSource` adapter together in T21.** T22
    will extend the adapter for per-player LM seating; T23 is the
    real-LLM smoke game.
  - **Single LM across all seats** in T21. The adapter's constructor
    takes one `lm: BaseLM`.
  - **Protocol extension on `DecisionSource`.** One method:
    `observe(state, new_events)`. Driver calls after each phase with
    the freshly-appended slice. The scripted source is no-op.
  - **Roster-order iteration.** `night_actions` and `day_actions` walk
    `self._roster` (not `state.alive_players()`) so two independent
    `ReActDecisionSource` instances with the same `(roster, lm-queue)`
    inputs decide identically (invariant #4).
  - **Dead players are skipped in `observe`** (review-driven): they will
    never be asked for actions again, so growing their memory is wasted
    work and confuses post-game inspection.
  - **Closure binding for tools.** `_bind_cognitive` / `_bind_terminal`
    use `inspect.signature(fn).replace(parameters=trailing)` +
    `__signature__` assignment so DSPy's `Tool` infers the LLM-facing
    schema from only the trailing args; `functools.wraps` carries
    `__name__` / `__doc__` so the LLM sees the original tool name and
    docstring.
  - **`ScriptedLM` lives in tests** (no shared helper yet). The tests
    use `dspy.utils.dummies.DummyLM` (a real DSPy testing facility);
    CLAUDE.md rule 2 — promote when T22/T23 need it.
- **Out of scope (per plan):** per-player LM seating (T22), real-LLM
  smoke game (T23), werewolf-chat sub-loop, bidding/discussion
  terminals, illegal-move-rate metric (T24), Tier 1/2 retrieval.
- **Tests, react primitive** (`tests/agents/test_react.py`, 12 cases):
  trivial-terminal-then-finish; cognitive-before-terminal trajectory
  (4 LM calls pinned); invalid-then-valid retry; finish-without-commit
  raises (regex pins `"finished without a committed"`); `max_iters`
  exhausted without finish raises; `remember` and `set_belief` memory
  side-effects; cognitive `"ok: ..."`-shaped observation does not
  terminate; two-identical-runs determinism; LM-context restoration
  with a sentinel `DummyLM` (not the `None is None` tautology);
  exile-vote terminator passes `abstain` through; `Commit` frozen,
  hashable, structurally equatable.
- **Tests, adapter** (`tests/agents/test_decisions.py`, 12 cases):
  `observe` routes public + private events correctly; non-duplication
  across two slices; `night_actions` aggregates kill votes / seer
  inspect / doctor protect; dead seer / dead doctor → `None`;
  `day_actions` covers every alive player and passes ABSTAIN through;
  roster-order pins LM call order; end-to-end `run_game` reaches
  `GAME_OVER`; invariant #2 read-side closure (every recorded event is
  public or names the player); `memories` is a read-only `Mapping`.
- `/sdb-review`: python + test + integrity reviewers **all PASS** (0
  critical, 0 high). 11 mediums addressed before commit (per T15/T16
  precedent):
  - `react.py`: weakened `_format_trajectory` docstring overclaim;
    captured the upstream `ReAct.react` `ValueError` and chained it
    into the `RuntimeError` (`raise ... from err`).
  - `decisions.py`: replaced three `assert isinstance(value, str)` with
    one `_require_str_target` helper that raises `TypeError` (not
    stripped under `python -O`); added `-> Commit` annotation on
    `_run_one`; `observe` now filters by `state.is_alive(name)`.
  - `loop.py`: tightened the `DecisionSource` docstring to state
    `GAME_OVER` is not routed through `observe`.
  - `test_react.py`: removed dead `_extract()` scaffolding from all
    scripts (the loop drives `react.react`, never the extract step);
    rewrote stale comments; tightened the finish-without-commit and
    max-iters regex to `r"Wolf1.*finished without a committed"`;
    `test_lm_context_is_restored_after_react_decide` now installs a
    sentinel `DummyLM` via `dspy.configure` and asserts that sentinel
    survives, instead of `None is None`.
- Verified: `pytest` 399/399, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**M4 (DSPy ReAct agent + memory) is half-closed** — T19, T20, T21 all
done. T22 (per-player LM seating) and T23 (real-LLM smoke game) remain.

**Next:** T22 — multi-model config: seat different LLMs as different
players. The adapter currently takes a single `lm`; T22 extends to a
mapping or a `Callable[[str], BaseLM]`.

---

## 2026-05-20 — T16: cognitive tools (M3 close)

- Added the eight cognitive tools from WEREWOLF_DESIGN.md §6.2 — the
  read-only / private-scratchpad surface the T21 DSPy ReAct loop will
  call between game actions:
  - `src/social_deduction_bench/agents/cognitive.py` (new) — six
    **game-agnostic** wrappers over `GameMemory`: `recall`, `remember`,
    `get_beliefs`, `set_belief`, `get_plan`, `set_plan`. Each is a
    thin str-returning delegation; ONUW / Secret Hitler reuse them
    unchanged. A private `_validate_caller(state, caller)` runs
    `state.player(caller)` up front so an unknown caller fails loud
    (`KeyError`) before any memory mutation.
  - `src/social_deduction_bench/games/werewolf/cognitive.py` (new) —
    two **werewolf-specific** readers: `get_public_state` (round /
    phase / alive / dead list, roles deliberately omitted) and
    `get_private_info` (self-line for every caller; living-pack
    `partners:` line for werewolves; `inspections:` history for the
    seer, filtered out of `memory.events` to `SEER_INSPECT`).
- **Decisions (user, plan):**
  - **Split by game-agnosticism.** Six wrappers in `agents/`, two
    readers in `games/werewolf/`. Mirrors T19/T20 storage placement.
  - **Uniform `(state, memory, caller, ...)` positional signature**
    on all eight tools; all return `str`. No `AgentContext` bundle —
    a DSPy ReAct loop sees text observations either way.
  - **Writers raise `ValueError`** (`remember` / `set_belief` /
    `set_plan` let `GameMemory` / `Note` / `Belief` `__post_init__`
    bubble up; CLAUDE.md rule 11 fail-loud). DSPy ReAct catches and
    surfaces as an Observation; we do not synthesize an error string.
  - **Label key=value render style** with sentinel `(none)` for
    empty sections. `get_beliefs` sorts by player name (deliberately
    reversed insertion to detect a missing sort); `get_public_state`
    keeps engine player order; `get_private_info` sorts werewolf
    partners by name and renders seer inspections in memory
    insertion order. **Living-pack partners only** — dead werewolves
    are excluded so the LLM coordinates with who is around to act.
  - **No `query` arg on `recall` yet.** WEREWOLF_DESIGN.md §6.2 lists
    `recall(query, last_n_rounds)`, but T19's `GameMemory.recall` is
    Tier 0; T16 matches storage. Tier 1 grows the parameter at both
    layers together.
  - **Dead caller still reads private/public.** Cognitive layer
    never gates on alive (mirrors `available_tools` (T17) split:
    semantic emptiness for dead, structural fail-loud for unknown).
  - **No `__init__.py` re-export.** Werewolf sub-package convention
    is direct module imports (T15 precedent); `agents/__init__.py`
    keeps its existing storage re-exports (`Belief`, `GameMemory`,
    `Note`) but does not add the six cognitive functions.
- **Tooling note:** `tests/games/werewolf/test_cognitive.py` would have
  collided by basename with `tests/agents/test_cognitive.py` under
  pytest's package-less prepend import mode (same issue T11 hit on
  `test_events.py`). Resolved surgically by naming the werewolf-side
  file `test_cognitive_views.py` — no import-mode change.
- Tests `tests/agents/test_cognitive.py` (31): empty-memory recall,
  delegation passthrough + `last_n_rounds` forwarding, `KeyError` on
  unknown caller across all six wrappers, `ValueError` propagation
  for blank note / bad confidence (parametrized over
  `["uncertain", "LOW", "Medium", "extreme", "", "medum"]`) / blank
  plan, `state.round` binding for `remember` with end-to-end `recall`
  closure, `get_beliefs` empty sentinel + literal one-row pin +
  sorted-by-name multi-row (deliberately reversed insertion) +
  `(no evidence)` placeholder, **determinism as a function of write
  sequence** (two independent memories built from same op sequence
  render byte-identical), state non-mutation on a read call.
- Tests `tests/games/werewolf/test_cognitive_views.py` (17):
  `get_public_state` literal layout + breadcrumb on `phase=day`,
  kill moves player to dead list, **dead-list follows engine player
  order** (reverse-roster-order kills still render in roster order),
  all-dead alive-(none), **invariant-#2 role leak guard** iterating
  `Role` directly (auto-covers a future fifth role), state purity,
  **determinism as a function of state** (two independent states from
  same op sequence). `get_private_info`: werewolf living-partner list,
  last-werewolf-alive `(none)`, seer no-inspections / inspection-history
  pin / **memory-insertion-order pin with reversed writes** /
  no-`partners:`-line guard / non-`SEER_INSPECT` event ignored
  (full-string equality, not `.endswith`), villager + doctor self-line
  only, **villager/doctor leak guard** with word-boundary regex
  (`\b{role}\b` avoids the `villager` ⊂ `villagers` faction-substring
  false positive), werewolf does-not-leak-non-pack-names, **dead-werewolf
  spectator** still sees living pack, state purity, unknown caller →
  `KeyError`. Suite 375/375.
- `/sdb-review`: python + integrity reviewers **all PASS** (0 critical,
  0 high, 0 medium); test reviewer PASS with 7 medium hardening items
  + 5 missing-coverage items. **All 12 addressed before commit**: drop
  the `_state(round_, phase)` foot-gun helper; rewrite both
  "determinism" tests as functions-of-input (two independent
  instances); iterate `Role` directly in the public-state leak guard;
  add word-boundary regex to villager/doctor leak guards (catches the
  `Doc`/`doctor` substring collision the consolidator flagged); switch
  the non-`SEER_INSPECT` test from `.endswith` to full equality; add
  `phase=day` breadcrumb to the literal-layout test; close the
  `remember` test with an end-to-end `recall` assertion; parametrize
  `set_belief` bad-confidence over six values; add the seer reversed-
  insertion-order test, seer no-`partners:`-line guard, dead-list
  non-roster-order test, dead-werewolf still-sees-living-pack test.
  Reports in `.reviews/20260520-1646-57abfc5-T16/`.
- Verified: `pytest` 375/375, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**M3 (Tool set) is complete** — T15, T17, T18, T16 all done. The
agent-facing surface (game-action tools + role/phase gating + bidding +
cognitive tools) is now fully in place. T21 (DSPy ReAct agent loop) is
the next unblocker — its dependencies T16 and T19 are both `[x]`.

**Next:** T21 — DSPy ReAct agent: one decision-point loop, cognitive
tools as intermediate steps, one game-action tool terminates the loop.

---

## 2026-05-20 — T20: persistent `plan` string on `GameMemory`

- Added a per-agent persistent strategic plan (WEREWOLF_DESIGN.md §9) to
  `agents/memory.py` — the storage T16's `get_plan` / `set_plan` cognitive
  tools will read/write:
  - `GameMemory._plan: str = ""` (initialized inside `__init__`, not at
    class scope, so per-agent isolation holds).
  - `set_plan(text: str) -> None` — overwrites the prior plan. Validates
    `isinstance(text, str)` and `text.strip()`, raises `ValueError`
    naming `Plan` on rejection. Blanks fail loud (mirrors `remember` and
    `Note.text`); §9 frames the plan as a strategic statement, so an
    explicit clear belongs as meaningful text, not a blank.
  - `plan -> str` read-only property. ``""`` until first `set_plan`.
- `recall()` is **unchanged** — events + notes only, per §7 Tier 0. The
  plan is a separate read surface T16 reads directly. A regression test
  (`test_plan_is_not_surfaced_through_recall`) pins the contract so any
  future fold-in must rewrite that test deliberately.
- **Decision (user, plan):** T20 lands before T16. The cognitive-tools
  brief (T16) lists `get_plan` / `set_plan`, but the underlying storage
  (this plan field) was T20's. Doing T20 first keeps T16 a thin
  interface layer over a final storage shape.
- **Decision (plan):** **no belief-table refinement.** T19 already
  shipped the §8 row shape verbatim (`player`/`guess`/`confidence`/
  `evidence`, fail-loud `__post_init__`, `MappingProxyType` view). The
  spec does not call for `round` / history / faction vocab; adding any
  would violate CLAUDE.md rule 2. T20 is plan-only.
- **Decision (plan):** default `plan = ""` (not `None`). Keeps the
  `str` invariant on the read accessor, so T16's `get_plan(state,
  memory, caller) -> str` becomes a one-liner with no `Optional`
  plumbing.
- Tests `tests/agents/test_memory.py` (+18 cases): default-empty
  (`""` + `isinstance(plan, str)`); round-trip; overwrite (mirrors
  `set_belief`); blank rejection parametrized over `["", "   ", "\n",
  "\t"]`; non-`str` rejection parametrized over
  `[None, 123, 1.5, b"bytes", ["list"], {"k": "v"}]` (review-driven —
  pins that a future "relax to truthy-only" refactor cannot silently
  let `None` through); read-only property shape; `plan` not surfaced
  through `recall()`. Two existing tests extended:
  `test_two_memories_independent` now writes a plan on `m1` and asserts
  `m2.plan == ""` (per-agent isolation, invariant #2); the determinism
  test (now `test_identical_operation_sequence_yields_identical_state`)
  writes two plans inside the build sequence and asserts both memories
  converge on the second. Suite 318/318.
- `/sdb-review`: python + test + integrity reviewers **all PASS** (0
  critical, 0 high, 0 medium on python+integrity; 2 mediums + 1 coverage
  on test, all addressed). Addressed before commit: capitalized the
  error-message prefix to `"Plan"` (consistency with `"Note"` / `"Belief"`,
  test matches updated); expanded the non-`str` rejection test to cover
  `None`/`bytes`/`list`/`dict`/`float`; renamed the determinism test to
  `_yields_identical_state` (the assertions now cover plan and beliefs,
  not just recall output); added the `recall()` no-plan pin. Reports in
  `.reviews/20260520-1627-97ba9ec-T20/`.
- Verified: `pytest` 318/318, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T16 — cognitive tools (`get_public_state`, `get_private_info`,
`recall`, `remember`, `get_beliefs`, `set_belief`, `get_plan`, `set_plan`).
Plain str-returning readers, plain positional args `(state, memory, caller)`,
no `AgentContext` bundle (user decisions during T20 plan).

---

## 2026-05-20 — T19: `GameMemory` (Tier 0) — M4 entry point

- Added `src/social_deduction_bench/agents/memory.py` — the per-agent
  in-process memory the (future) DSPy ReAct agent (T21) and cognitive
  tools (T16) consume:
  - `Note` (`frozen=True, slots=True`): `round: int, text: str`.
    `__post_init__` rejects negative rounds and blank text — fail-loud
    at write time so `recall` cannot render empty lines as if real.
  - `Belief` (`frozen=True, slots=True`): `player, guess, confidence,
    evidence`. `confidence: Literal["low", "medium", "high"]` — a typo
    (`"medum"`) or case shift (`"HIGH"`) fails fast at `set_belief`,
    not later via drift in the structured-table contract.
  - `GameMemory` — owns three append-only / overwriting buffers
    (`_events`, `_notes`, `_beliefs`). Methods: `record_event`,
    `remember`, `set_belief`. Read-only accessors return tuple
    snapshots (`events`, `notes`) and a `MappingProxyType` (`beliefs`).
    Tier 0 `recall(last_n_rounds: int | None = None) -> str` returns
    `[R{round}] {event.type} {sorted-json-payload}` and `[R{round}]
    note: {text}` lines sorted by `(round, kind, insertion_index)` —
    events kind 0, notes kind 1, so same-round chronology is
    "engine acted, then agent reflected." `last_n_rounds < 0` raises
    `ValueError`; `last_n_rounds = 0` returns `""`.
- `src/social_deduction_bench/agents/__init__.py` — re-exports
  `Belief`, `GameMemory`, `Note` via `__all__` (sets the convention
  for `agents/` before T16/T20/T21 land; mirrors `engine/__init__.py`).
- **Decision (user, plan):** `Belief.confidence` is the ordinal
  `Literal["low", "medium", "high"]`, not a float. The LLM ergonomics
  argument won — three buckets are easier to emit consistently than a
  free `0..1` probability. Migration risk to float later is small (add
  a derived field); reverse direction would be lossy.
- **Decision (user, plan):** `recall` is strict Tier 0 — no `query`
  parameter. Tier 1's substring filter lands with its implementation;
  adding a no-op `query: str | None = None` now would be the speculative
  parameter CLAUDE.md rule 2 rejects.
- **Decision (user, plan):** `agents/__init__.py` re-exports the public
  surface via `__all__` (engine-style). `agents/` is the boundary
  T16/T20/T21 cross, closer to `engine/` than to the flat-module
  `games/werewolf/` convention.
- **Decision (plan):** `record_event` is the ingestion seam (a method
  on `GameMemory`), not a free function. The T21 agent loop calls it
  after `observations_for(events, player)` once per decision point.
  Memory does **not** dedup — a double-push is a loop bug surfaced by
  loop tests, not silently absorbed here.
- **Decision (review-driven):** dropped the `default=str` fallback in
  `recall`'s `json.dumps`. The engine's `Event.__post_init__` already
  gates payloads to JSON primitives — `default=str` would have been
  dead defense that masks a future widening of the `Event` contract.
  Fail-loud now matches `events.py::to_jsonl_lines` exactly.
- Tests `tests/agents/test_memory.py` (36): construction + read-only
  accessor contracts (tuple snapshots, `MappingProxyType` immutability);
  `record_event` insertion order + no-dedup; `remember` blank-text +
  negative-round rejection; `set_belief` overwrite, the three-bucket
  acceptance, parametrized rejection of `"uncertain"`/`"LOW"`/etc.,
  blank-player/guess rejection, empty-evidence acceptance; `recall`
  empty-memory empty string, mixed events+notes round order, line
  format (regex + literal — payload keys inserted in reverse order so
  removing `sort_keys=True` would flip the rendered output), event-
  before-note same-round ordering pin (new, from review), the
  `last_n_rounds` filter incl. edge cases (zero, larger-than-history,
  empty memory, negative → `ValueError`), notes contribute to
  `latest`-round cutoff, stable order at same round; determinism +
  isolation (two memories from identical op sequences yield byte-
  identical `recall`; cross-instance independence). Suite 303/303.
- `/sdb-review`: python + test + integrity reviewers all PASS
  (0 critical, 0 high). Addressed three mediums before commit: dropped
  `default=str`; strengthened the sort-keys test by switching to
  `victim`/`actor` keys whose insertion order disagrees visibly with
  sorted order; added the event-before-note same-round ordering test.
  Reports in `.reviews/20260520-1444-7e5f852-T19/`.
- Verified: `pytest` 303/303, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T16 (cognitive tools) or T20 (belief table refinement +
persistent plan string). Both are unblocked; T16 is the wider
unblocker for T21 and the M4 agent loop.

---

## 2026-05-20 — T18: bidding-based speech ordering

- Added `games/werewolf/discussion.py` — `BiddingActions` /
  `DiscussionResult` frozen dataclasses and `resolve_discussion(state,
  actions, rng)`, the pure resolver picking the top `K_DISCUSSION_SLOTS = 3`
  bidders in descending bid order. Tie-break is one `rng.shuffle` per tied
  amount group over a `sorted(...)` leader list — sorted before the draw so
  the RNG sees byte-identical input regardless of bid-dict iteration order
  (invariant #4). Voter-alive guard mirrors `resolve_day` / `resolve_night`:
  a bid from a non-living (or unknown) player raises `ValueError` rather
  than seating a phantom speaker.
- Added `K_DISCUSSION_SLOTS: Final[int] = 3` (Werewolf Arena baseline for
  7-player games) and `MAX_BID: Final[int] = 100` (the §6.1 `0..N` upper
  bound parked since T15) to `games/werewolf/config.py`. `submit_bid` now
  clamps to `[0, MAX_BID]` — the rejection reason names both the cap and
  the offending amount for the agent's self-correction.
- **Decision (user, plan):** `K_DISCUSSION_SLOTS = 3` (Werewolf Arena
  baseline); `MAX_BID = 100` (bounded to deny an unbounded-bid griefing
  surface — RNG cost, prompt-token bloat, integer overflow); **defer the
  `run_game` wiring + `SPEECH` / per-bid event emission to T21** when the
  DSPy agent layer drives speech turns and constrains the event-privacy
  question. The T17-flagged "filter `speak` out of the day menu for
  non-bid-winners" follow-up rides on the same T21 wiring (it needs a
  sub-phase or "active speakers" slot in `GameState`).
- **Decision (plan):** the resolver intentionally accepts an arbitrary or
  empty bid subset rather than enforcing "every alive agent submits a bid"
  (§4) — that invariant belongs to the loop (T21), which maps a missing or
  timed-out bid to 0 before calling the resolver. Documented on
  `BiddingActions`.
- **Decision (plan):** no event emission yet. The pure resolver returns the
  speaker tuple; `SPEECH` / `DISCUSSION_RESOLVED` event drafts and the
  `DecisionSource` Protocol extension land with the loop wiring (T21), where
  the event-privacy question is constrained by the agent caller.
- **Decision (integrity review):** `K_DISCUSSION_SLOTS` and `MAX_BID` are
  `Final[int]` so Pyrefly flags any reassignment — mirrors the
  `MappingProxyType` / `frozenset` immutability discipline of the adjacent
  constants.
- Tests: `tests/games/werewolf/test_discussion.py` (11) — descending-bid
  order pin (zero bid excluded only because K is full), fewer-than-K clamp,
  zero bids can win a slot (§5 `0..N`), seeded tie-break (hand-verified
  seed pair (0,1) diverges; permutation-set membership), tie-break replayable
  across two fresh `GameRNG(7)` runs, mixed-tie test (unique top is always
  slot 0, RNG only governs tie scopes), empty bids → empty speakers, input
  non-mutation via whole-object equality, non-day-phase `ValueError`,
  dead-bidder `ValueError`, unknown-bidder `ValueError` (defense against a
  future refactor swapping `state.alive_names()` for `state.player(...).alive`).
  `tests/games/werewolf/test_config.py` (2) — `K_DISCUSSION_SLOTS == 3`,
  `MAX_BID == 100`. `tests/games/werewolf/test_tools.py` (2) — `submit_bid`
  accepts the `MAX_BID` boundary, rejects `MAX_BID + 1` with a reason naming
  both. Suite 267/267.
- `/sdb-review`: python + test + integrity reviewers **all PASS** (0
  critical, 0 high, 6 medium). All 6 mediums addressed before commit:
  loop-responsibility note on `BiddingActions`; inline why-comment on the
  sort-then-shuffle determinism rationale; `Final[int]` on both new config
  constants; hand-verified-seed comment on the tie-break test; deleted the
  duplicate `K_DISCUSSION_SLOTS` test; tightened the phase-rejection match
  from `"day"` to `"day phase"`; added the unknown-bidder test from the
  missing-coverage list. Reports in
  `.reviews/20260520-1241-a26a3b1-T18/`.
- Verified: `pytest` 267/267, `ruff check`, `ruff format --check`,
  `pyrefly check` (0 errors).

**Next:** T19 — `GameMemory` (Tier 0): events, notes, beliefs;
`remember`/`recall`/`set_belief` (M4 entry point). T16 (cognitive tools)
unblocks once T19 lands.

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
