# Social Deduction Benchmark

A multi-agent LLM benchmark for social deduction games. Werewolf is the core game (ONUW and Secret Hitler follow). DSPy + ReAct agents play; a seeded **engine acts as referee**; different LLMs compete on a TrueSkill leaderboard.

Stack: Python 3.13, Poetry, DSPy + `litellm`, TrueSkill, pytest, Ruff. Pinned versions live in `pyproject.toml` once scaffolded.

Bias toward caution over speed on non-trivial work. Use judgment on trivial tasks.

---

## Repo layout

```
social-benchmarks/
  CLAUDE.md
  pyproject.toml
  src/social_deduction_bench/
    engine/      # game-agnostic referee: state, phases, event log, RNG
    games/       # per-game rules, roles, phases (werewolf/ first)
    agents/      # DSPy ReAct agents, memory, cognitive + game-action tools
    rating/      # TrueSkill, leaderboard, metrics
  tests/
  social-deduction-benchmarks/   # RESEARCH ONLY — survey, report.md, WEREWOLF_DESIGN.md
  .claude/                       # CLAUDE.md skills: agents + commands
  .reviews/                      # review outputs (gitignored)
```

`social-deduction-benchmarks/` holds research and design artifacts only. Implementation code never lives there and never imports from it. **`social-deduction-benchmarks/WEREWOLF_DESIGN.md` is the authoritative spec for game rules, tools, and the memory design** — when code and that doc disagree, the doc wins (or the doc gets a flagged update).

---

## Benchmark invariants (non-negotiable)

A benchmark is only worth running if results are trustworthy. These hold everywhere:

1. **The engine is the single source of truth.** All game state lives in the engine.
2. **Agents never read hidden state.** They receive only observations the engine emits — public events to everyone, private events to the relevant agent.
3. **Agents change state only via validated tool calls.** Illegal moves (dead target, wrong phase, wrong role) are rejected with an error observation; they are never silently applied.
4. **Everything is seeded → deterministic → replayable.** Role assignment, kill resolution, tie-breaks, speech order — all derive from the engine seed. Same seed + same models → same game.
5. **Every game is an append-only event stream** (JSONL): replayable, debuggable, the basis for post-hoc metrics.

A change that weakens any of these is a Critical issue, not a style nit.

---

## Behavioral rules

### 1. Think before coding
State assumptions explicitly. Present multiple interpretations when ambiguity exists. Push back when a simpler approach exists. Stop when confused; name what's unclear.

### 2. Simplicity first
Minimum code that solves the problem. Nothing speculative. No abstractions for single-use code. Would a senior engineer call this overcomplicated? Then simplify.

### 3. Surgical changes
Touch only what you must. Don't "improve" adjacent code, comments, or formatting. Match existing style.

### 4. Goal-driven execution
Define success criteria. Loop until verified. Don't follow steps blindly — define success and iterate.

### 5. Token budgets are not advisory
If a task is approaching context limits, summarize and start fresh. Don't push through silently.

### 6. Surface conflicts, don't average them
If two existing patterns in this repo contradict, pick the more recent / more tested one, explain why, and flag the other for cleanup. Average code that satisfies both rules is the worst code. When game behavior is in question, `WEREWOLF_DESIGN.md` and the engine's existing invariants win.

### 7. Read before you write
Before adding code to a file, read its exports, the immediate caller of any function you're touching, and any obvious shared utilities. "Looks orthogonal to me" is the most dangerous phrase in this codebase.

### 8. Tests verify intent, not just behavior
Every test must encode WHY the behavior matters, not just WHAT it does. A test that can't fail when game logic changes is wrong. Determinism and the benchmark invariants above must have explicit tests.

### 9. Checkpoint after every significant step
After each step in a multi-step task: state what was done, what's verified, what's left. If you lose track, stop and restate.

### 10. Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase. If you genuinely think a convention is harmful, surface it. Don't fork silently.

### 11. Fail loud
"Engine works" is wrong if you didn't verify the requested edge case. "Tests pass" is wrong if any were skipped. Surface uncertainty in an explicit `Unverified` section, not hidden.

---

## Workflow

Test-Driven Development. Tests first, then implementation.

1. Clarify — ask if requirements are ambiguous
2. Write tests — in `tests/`, mirroring the `src/` package layout
3. Implement — minimum code to make the tests pass
4. Verify — `poetry run pytest`, `poetry run ruff check`, `poetry run pyrefly check`
5. Review — run `/sdb-review` before pushing
6. Commit — focused commits

Don't skip tests. Don't guess requirements — ask.

---

## Task & session workflow

Task state lives **on disk**, not in the conversation — it must survive compaction and `/clear`. Two tiers:

- **`BACKLOG.md`** — the durable task list, grouped by milestone. Each task has an ID, a status checkbox, and dependencies. Derived from `social-deduction-benchmarks/WEREWOLF_DESIGN.md` via `/sdb-plan`.
- **`PROGRESS.md`** — append-only session log: what shipped, decisions made, what's next. The "read me first" file for any new session.
- The in-session Task tool tracks only the **current** task's sub-steps — never rely on it across sessions.

**The session loop:**

1. **Orient** — read `PROGRESS.md`, `BACKLOG.md`, and `git log`; run `poetry run pytest` as a smoke test. Agents mark work "done" prematurely — verify before trusting. (`/sdb-next` automates this.)
2. **Pick** — choose the next unblocked task from `BACKLOG.md`.
3. **Plan** — enter plan mode, decompose, get approval. Skip planning only for one-sentence changes.
4. **Build** — TDD: write tests, hand to `sdb-tdd-implementer`, iterate to green.
5. **Review** — `/sdb-review`, fix findings.
6. **Close** — tick the task in `BACKLOG.md`, append a `PROGRESS.md` entry, commit. If context is running low, write the `PROGRESS.md` entry *before* compaction.

Task IDs (`T01`…) may appear in `BACKLOG.md`, `PROGRESS.md`, and commit messages — never in source code or docstrings.

---

## Sub-Agents

Read-only reviewers and one implementer (the only agent that writes). Each reviewer persists its output to `.reviews/` (gitignored) for later grep. The orchestrator picks the right reviewers from the diff and runs them in parallel.

| Agent | When to invoke | Role |
|---|---|---|
| `sdb-orchestrator` | Single entry point (`/sdb-review`) | Picks reviewers from the diff, dispatches in parallel, pipes to consolidator |
| `sdb-consolidator` | Auto-invoked by orchestrator | Merges reviewer outputs, dedupes, surfaces conflicts |
| `sdb-python-reviewer` | After any `src/` Python change | Bug / type / async / convention review |
| `sdb-test-reviewer` | After test changes | Test quality and meaningfulness |
| `sdb-benchmark-integrity-reviewer` | After engine, games, RNG, memory, or tool-plumbing changes | Determinism, seeding, hidden-state leaks, engine-as-referee invariants |
| `sdb-tdd-implementer` | When you have failing tests and want minimal implementations | The only agent with write access |

Diff scope: reviewers compare branch vs `main` (via `git merge-base`), falling back to `HEAD~1` on `main` itself.

---

## Coding Conventions (Python 3.13)

- Modern syntax: `str | None`, `list[str]`; never `Optional` or `List` from `typing`
- Absolute imports: `from social_deduction_bench.engine import ...`
- Naming: `snake_case` (functions), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants)
- Pydantic models (or frozen dataclasses) for structured data crossing module boundaries; no raw dicts at boundaries
- Logs: `logger.info("game=%s round=%s", game_id, round_)` — never f-strings
- `raise ... from e` when re-raising
- All config / env vars in one settings module — no `os.getenv()` elsewhere
- Ruff line-length 120, target py313
- Code is type-checked with Pyrefly — keep `poetry run pyrefly check` clean; no new type errors
- **Determinism:** anything stochastic (RNG, sampling, shuffles, tie-breaks) must derive from the engine seed. Never call `random` / `np.random` global state directly — thread a seeded generator. No wall-clock or `uuid4()` in game logic.
- **Time:** measuring elapsed duration uses `time.monotonic()`, never `time.time()`.
- **LLM calls:** all model access goes through DSPy/`litellm`; no direct provider SDK calls. Unit tests never hit a real LLM.

---

## Testing Strategy

- Tests first (TDD). Tests define the contract.
- Location: `tests/`, mirroring the `src/social_deduction_bench/` layout
- Test through the public API (engine, agent, rating), not internals
- Naming: `test_<what>_<condition>_<expected>`
- Mock only external I/O. The engine is pure and seeded — test it directly, no mocks.
- Determinism tests are mandatory for the engine: same seed → identical event stream.
- Tests that require a real LLM API key auto-skip when the key is absent — they never run by default in CI.

---

## Code Quality

Two tools, nothing else — no standalone isort, no flake8, no mypy/pyright.

- **Ruff** — lint + format + import sorting. The `I` rule provides import sorting natively, so no separate isort. Rules `E F I LOG UP T10 ISC ICN G PIE PT Q RSE FURB RUF`.
  - Check: `poetry run ruff check`
  - Format: `poetry run ruff format`
- **Pyrefly** — static type checking ([pyrefly.org](https://pyrefly.org)).
  - Check: `poetry run pyrefly check`
  - Config lives in `[tool.pyrefly]` in `pyproject.toml` (`pyrefly init` scaffolds it).

---

## Documentation Lookup

Use **Context7** before writing code that depends on a library — DSPy, `litellm`, Pydantic v2, `trueskill`, pytest, etc. Don't guess library APIs.

---

## Safety & Permissions

**Allowed without asking:** read/list files in the repo; run lints and tests; create/edit files in the repo.

**Ask first:** add/remove dependencies (`pyproject.toml`); change game rules, win conditions, or any benchmark invariant above; run benchmark games against **paid** LLM APIs (this spends real money — confirm scope and model first); modify CI.

**Never:** `git push`, force-push, delete branches, skip hooks (`--no-verify`), bypass signing.
