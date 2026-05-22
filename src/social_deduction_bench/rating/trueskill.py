"""Game-agnostic TrueSkill rating core.

Turns a sequence of `GameResult`s (model-name teams + the winning team index) into
a per-model `Leaderboard`. Social deduction games are team games with asymmetric
roles, so TrueSkill updates *individual* model ratings from *team* outcomes. This
module imports only `trueskill` (and stdlib) — never the engine or any game — so
ONUW / Secret Hitler reuse it.

Determinism: `trueskill.rate` is pure analytic math with no RNG, and groups are
built in sorted-model order, so identical input → identical `Leaderboard`. The
caller owns game ordering (TrueSkill is sequential/online); `rate_games` does not
re-sort the games.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import trueskill


@dataclass(frozen=True, slots=True)
class GameResult:
    """One decisive game: model-name teams and the winning team's index."""

    teams: tuple[tuple[str, ...], ...]
    winner: int

    def __post_init__(self) -> None:
        if len(self.teams) < 2:
            raise ValueError(f"a game needs at least 2 teams, got {len(self.teams)}")
        if any(len(team) == 0 for team in self.teams):
            raise ValueError("every team must have at least one model")
        if not 0 <= self.winner < len(self.teams):
            raise ValueError(f"winner index {self.winner} out of range for {len(self.teams)} teams")


@dataclass(frozen=True, slots=True)
class ModelRating:
    """One model's rating and win/loss record after rating a batch of games."""

    model: str
    mu: float
    sigma: float
    games: int
    wins: int
    losses: int
    skill: float


@dataclass(frozen=True, slots=True)
class Leaderboard:
    """Rated-game count, skipped-game count, and per-model ratings (skill desc)."""

    n_games: int
    n_skipped: int
    ratings: tuple[ModelRating, ...]


def rate_games(results: Sequence[GameResult], *, env: trueskill.TrueSkill | None = None) -> Leaderboard:
    """Rate `results` in order, returning a per-model `Leaderboard`.

    Werewolf outcomes are decisive (a faction always wins; no ties), so the default
    draw probability is 0.0; the `env` param lets a caller override it. Games where
    a model appears on more than one team (self-play / overlap) carry no cross-model
    signal and are skipped.
    """
    env = env or trueskill.TrueSkill(draw_probability=0.0)

    store: dict[str, trueskill.Rating] = {}
    games_played: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    n_rated = 0
    n_skipped = 0

    for result in results:
        team_models = [sorted(set(team)) for team in result.teams]
        all_models = [model for team in team_models for model in team]
        if len(all_models) != len(set(all_models)):
            n_skipped += 1
            continue

        groups = [{model: store.get(model, env.create_rating()) for model in team} for team in team_models]
        ranks = [0 if i == result.winner else 1 for i in range(len(result.teams))]
        rated = env.rate(groups, ranks=ranks)

        for i, group in enumerate(rated):
            for model, rating in group.items():
                store[model] = rating
                games_played[model] += 1
                if i == result.winner:
                    wins[model] += 1
        n_rated += 1

    entries = [
        ModelRating(
            model=model,
            mu=rating.mu,
            sigma=rating.sigma,
            games=games_played[model],
            wins=wins[model],
            losses=games_played[model] - wins[model],
            skill=env.expose(rating),
        )
        for model, rating in store.items()
    ]
    entries.sort(key=lambda r: (-r.skill, r.model))
    return Leaderboard(n_games=n_rated, n_skipped=n_skipped, ratings=tuple(entries))
