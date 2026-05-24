# Contributing

Thanks for your interest in the Social Games Arena. Bug fixes, new games, agent
improvements, site polish, new models, and shared matches are all welcome.

## Development setup

Requires **Python 3.13** and [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run pre-commit install   # optional but recommended
```

The project is **test-driven**: write the test first, then the minimum code to make it pass.
Tests live in `tests/`, mirroring the `src/social_deduction_bench/` layout, and test through
the public API (engine, agent, rating) rather than internals.

### Green before you open a PR

CI runs exactly these — make sure they pass locally first:

```bash
poetry run ruff check
poetry run ruff format --check
poetry run pyrefly check
poetry run pytest
```

A few house rules worth knowing (the full set is in `CLAUDE.md`):

- **Determinism is non-negotiable.** Anything stochastic must derive from the engine seed —
  never touch global `random`/`np.random`, wall-clock time, or `uuid4()` in game logic.
- The **engine is the single source of truth**; agents only ever see observations the engine
  emits, and only change state through validated tool calls.
- Modern typing (`str | None`, `list[str]`), absolute imports, Pydantic/frozen dataclasses at
  module boundaries. Ruff (line length 120) + Pyrefly, nothing else.

## Adding a model to the leaderboard

Models are played through [OpenRouter](https://openrouter.ai), so any model OpenRouter serves
can compete.

1. Find the exact model ID on [openrouter.ai/models](https://openrouter.ai/models).
2. Smoke it in a single game:
   ```bash
   export OPENROUTER_API_KEY=sk-...
   poetry run sdb-werewolf --model "vendor/model-id" --seed 42
   ```
3. Add it to a sweep alongside the existing field:
   ```bash
   poetry run sdb-tournament --models "vendor/model-id,deepseek/deepseek-v4-flash" \
     --games-per-pair 10 --output-dir games/run3
   ```

Don't have a key for the model you want? [Open an issue](https://github.com/Archelunch/social-games-arena/issues)
requesting it, or sponsor a run (see **Support** below).

## Sharing matches

The live leaderboard is built from committed game artifacts in `games/run2/`. A single game is
a self-contained directory:

```
g<NNNN>-<werewolf-model>-vs-<villager-model>/
  events.jsonl          # required — the engine event stream (source of truth)
  trajectories.jsonl    # required — per-decision ReAct steps + LM metadata
  manifest.json         # required — seat→model map, seed, outcome (real model labels!)
  memories.json         # optional — agents' final beliefs/plans
```

To contribute matches you've run:

1. Run them with `sdb-tournament` (or `sdb-werewolf`) so the artifacts are well-formed.
2. Open a PR adding the game directories under `games/run2/`, **or** send them and I'll add
   them. Make sure `manifest.json` carries the real model labels — that's what the leaderboard
   attributes results to.
3. CI rebuilds the site from the committed games on merge; no re-rating step is needed.

Games are aggregated deterministically (sorted by game ID), so the same set of matches always
produces the same leaderboard.

## Support

This is an independent project and the leaderboard runs on paid APIs. Ways to help:

- ☕ [Buy Me a Coffee](https://buymeacoffee.com/mike_pavlukhin) — funds the API runs.
- 🔑 Provide an API key or request a specific model to be benchmarked (open an issue).
- 🎮 Share your matches (above) so they join the public leaderboard.
