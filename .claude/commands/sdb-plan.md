Decompose a design/spec document into `BACKLOG.md` tasks.

## Input

`$ARGUMENTS` — path to the design doc to decompose. If empty, default to `social-deduction-benchmarks/WEREWOLF_DESIGN.md`.

## Procedure

1. Read `CLAUDE.md` (Task & session workflow section) and the target design doc in full.
2. Read the existing `BACKLOG.md` if it exists — you are **extending or revising**, not blindly overwriting. Never drop or renumber existing `[x]` done tasks.
3. Decompose the design into **milestones** (coarse, dependency-ordered phases) and **tasks** under each milestone. A good task is:
   - One TDD cycle of work — small enough to finish and review in one sitting.
   - Independently testable — it has a clear "done" condition.
   - Explicit about dependencies (which task IDs must be `[x]` first).
4. Assign sequential IDs (`T01`, `T02`, …) continuing from the highest existing ID. Never reuse an ID.
5. Write `BACKLOG.md` using the existing format: milestone headers, `[ ]`/`[~]`/`[x]`/`[!]` checkboxes, `**Tnn** — <description>. _Depends: …_`.
6. Present the proposed backlog (or the diff, if revising) to the user for confirmation **before** finalizing. Do not start implementing.

## Rules

- Decompose to the design doc's intent — do not invent scope it does not describe.
- Game-agnostic work and game-specific work go in separate milestones (the engine must not depend on a specific game).
- If the design doc has open questions, do NOT create tasks that depend on unresolved decisions — note them and surface the blockers.
- Keep task descriptions to one line. Detail belongs in the design doc, not the backlog.
