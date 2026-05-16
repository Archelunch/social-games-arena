Start the next backlog task: orient, pick, and plan.

## Input

`$ARGUMENTS` — optional task ID (e.g. `T05`) to start a specific task instead of the next one.

## Procedure

### 1. Orient
- Read `PROGRESS.md` (newest entry) and `CLAUDE.md`.
- Read `BACKLOG.md`.
- Run `git log --oneline -10` to see recent work.
- Run `poetry run pytest` as a smoke test. **Agents mark work "done" prematurely** — if a task is marked `[x]` in `BACKLOG.md` but its tests fail or are missing, say so loudly and stop for the user to decide.
- If `pyproject.toml` does not exist yet, the project is unscaffolded — the only valid task is T01.

### 2. Pick
- If `$ARGUMENTS` names a task, use it (verify its dependencies are all `[x]`; if not, report the blockers and stop).
- Otherwise pick the **lowest-numbered task** whose status is `[ ]` and whose dependencies are all `[x]`.
- If no task is unblocked, report what is blocking and stop.
- Mark the chosen task `[~]` in `BACKLOG.md`.

### 3. Plan
- Read the relevant section of `social-deduction-benchmarks/WEREWOLF_DESIGN.md` for this task.
- Enter plan mode. Produce a TDD plan: which tests to write first, then the minimal implementation, then which reviewers `/sdb-review` will select.
- Present the plan and the chosen task to the user for approval. **Do not write code before approval.**

## After the task (reminder for the closing steps)

When the task is complete, verified, and reviewed: set it `[x]` in `BACKLOG.md`, append a dated entry to `PROGRESS.md` (what shipped, decisions, next task), and commit. If context is running low, write the `PROGRESS.md` entry before compaction.

## Rules

- Never skip the smoke test. A green `BACKLOG.md` checkbox is a claim, not proof.
- Respect dependency order — do not start a task with an unsatisfied dependency.
- One task at a time. Do not mark two tasks `[~]` at once.
