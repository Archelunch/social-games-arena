# Social Games Arena

[![CI](https://github.com/Archelunch/social-games-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/Archelunch/social-games-arena/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-orange?logo=buy-me-a-coffee)](https://buymeacoffee.com/mike_pavlukhin)

**A multi-agent LLM benchmark for social deduction games.** Language models play
[Werewolf](https://en.wikipedia.org/wiki/Mafia_(party_game)) against each other — some lie,
some hunt the liars — and a seeded engine referees every game. Models are ranked on a
[TrueSkill](https://trueskill.org/) leaderboard built from cross-play.

**Live leaderboard → [arena.pavlukhinlab.com](https://arena.pavlukhinlab.com)**

Werewolf is the first game; Avalon and Secret Hitler are on the roadmap.

---

## Why this benchmark

Most LLM benchmarks measure a model alone against a fixed answer key. Social deduction
measures something different: can a model **model other minds** — lie convincingly, read
deception, build and break trust, and coordinate — under hidden information and pressure?
Werewolf makes that measurable, and cross-play turns it into a ranking.

The benchmark only matters if its results are trustworthy, so it is built around a few
non-negotiable invariants:

1. **The engine is the single source of truth.** All game state lives in the engine.
2. **Agents never read hidden state.** They receive only the observations the engine emits —
   public events to everyone, private events to the relevant player.
3. **Agents change state only via validated tool calls.** Illegal moves (dead target, wrong
   phase, wrong role) are rejected with an error observation, never silently applied.
4. **Everything is seeded → deterministic → replayable.** Same seed + same models → the same
   game, every time.
5. **Every game is an append-only event stream** (JSONL): replayable, debuggable, and the
   basis for every post-hoc metric.

---

## Quick start

Requires **Python 3.13** and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/Archelunch/social-games-arena.git
cd social-games-arena
poetry install
```

Run a **free** scripted game (no API key, no network — useful to confirm the install):

```bash
poetry run sdb-werewolf --dry-run
```

Run a **real** game (models played through [OpenRouter](https://openrouter.ai)):

```bash
export OPENROUTER_API_KEY=sk-...
poetry run sdb-werewolf --model "qwen/qwen3.5-9b"
```

> Real runs call paid model APIs. The runner **fails loud and spends nothing** if
> `OPENROUTER_API_KEY` is unset — only `--dry-run` works without a key.

---

## Running a benchmark

Three commands, one per stage: play a game, run a sweep, build the site.

### One game — `sdb-werewolf`

```bash
poetry run sdb-werewolf --model "qwen/qwen3.5-9b" --seed 42
poetry run sdb-werewolf --werewolf-model "model-A" --villager-model "model-B"   # faction cross-play
poetry run sdb-werewolf --dry-run                                               # scripted, no key
```

Writes `events.jsonl`, `trajectories.jsonl`, `manifest.json`, and `memories.json` to
`games/<game-id>/` and streams the game live in your terminal.

### A tournament — `sdb-tournament`

Schedules every model pair (including self-pairs, as a diagnostic), runs `--games-per-pair`
seeded games each, and writes a TrueSkill `summary.json`.

```bash
export OPENROUTER_API_KEY=sk-...
poetry run sdb-tournament \
  --models "deepseek/deepseek-v4-flash,google/gemma-4-31b-it,qwen/qwen3.5-9b" \
  --games-per-pair 10 \
  --seed 2026 \
  --concurrency 10 \
  --output-dir games/run3
```

Useful flags: `--resume` (recover a crashed sweep without re-paying for finished games),
`--reasoning` (enable model thinking tokens), `--temperature`, `--max-tokens`, `--max-iters`,
and `--dry-run` (scripted, no key). Run `poetry run sdb-tournament --help` for the full list.

### The site — `sdb-site`

Builds a static, zero-dependency leaderboard + replay viewer from completed games:

```bash
poetry run sdb-site --run games/run2 --out site
(cd site && python -m http.server)   # → http://localhost:8000
```

Pass `--run` more than once to aggregate several runs into one site. The output is
byte-identical for the same set of games (no timestamps or paths baked in).

---

## How it's organized

```
src/social_deduction_bench/
  engine/      # game-agnostic seeded referee: state, phases, event log, RNG, validation
  games/       # per-game rules, roles, phases (werewolf/ first)
  agents/      # DSPy ReAct agents, per-seat memory, cognitive + game-action tools
  rating/      # TrueSkill ratings, leaderboard, run manifest
```

| CLI | Entry point | Does |
|---|---|---|
| `sdb-werewolf` | `cli.py` | Run and stream a single Werewolf game |
| `sdb-tournament` | `tournament_cli.py` | Run a seeded pairwise cross-play sweep |
| `sdb-site` | `site_cli.py` | Build the static leaderboard + replay site |

Models reach the game through DSPy + `litellm` via OpenRouter — no direct provider SDK calls.
The full rules, tool set, and memory design live in
[`social-deduction-benchmarks/WEREWOLF_DESIGN.md`](social-deduction-benchmarks/WEREWOLF_DESIGN.md),
which is the authoritative spec.

### Game artifacts

Each game produces an append-only record under its own directory:

| File | Contents |
|---|---|
| `events.jsonl` | The engine event stream — the source of truth |
| `trajectories.jsonl` | Per-decision ReAct steps + per-call LM metadata (tokens, latency, cost) |
| `manifest.json` | Provenance: seat→model map, seed, sampling config, outcome |
| `memories.json` | Each agent's final beliefs, plan, and notes |

A tournament adds a `summary.json` with the leaderboard and aggregate metrics.

---

## Development

```bash
poetry install
poetry run pre-commit install   # optional: ruff + pyrefly on commit

poetry run pytest               # test suite (real-LLM "smoke" tests are opt-in)
poetry run ruff check
poetry run ruff format --check
poetry run pyrefly check
```

The project is test-driven; tests mirror the `src/` layout under `tests/`. Smoke tests that
call a real model are deselected by default and only run with `-m smoke` and a key set, so
the suite never spends money in CI. See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

---

## Roadmap

- [ ] More models on the leaderboard
- [ ] Optimize the agents with [GEPA](https://github.com/gepa-ai/gepa)
- [ ] Build a reinforcement-learning environment from the engine
- [ ] More games: Avalon, Secret Hitler

---

## Contributing

Contributions are very welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The short version:

- **Add a model** to the leaderboard, or **share matches** you've run (the per-game artifact
  directory is self-contained and merges cleanly into a sweep).
- **Give an API key or sponsor a model** you'd like benchmarked, and I'll run it.
- File issues and PRs for bugs, new games, agent improvements, or site polish.

## Support

This is an independent research project. If it's useful to you, here's how to help it grow:

- ☕ **[Buy Me a Coffee](https://buymeacoffee.com/mike_pavlukhin)** — directly funds the paid
  API runs behind the leaderboard.
- 🔑 **Provide API access for a model** you want benchmarked — [DM me](https://x.com/mike_pavlukhin)
  and I'll run it.
- 🎮 **Share your matches** so they can join the public leaderboard.
- ⭐ **Star the repo** to help others find it.

## License

[MIT](LICENSE) © Michael Pavlukhin
