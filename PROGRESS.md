# Progress Log — Social Deduction Benchmark

Append-only. Newest entry on top. Read this first when starting a session.

**Current state:** T04 done — append-only event stream + JSONL landed, all checks green.
**Next task:** T05 — observation routing (public to all, private to one agent).

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
