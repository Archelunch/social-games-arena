# Progress Log — Social Deduction Benchmark

Append-only. Newest entry on top. Read this first when starting a session.

**Current state:** T01 done — project scaffolded, all checks green.
**Next task:** T02 — seeded RNG utility.

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
