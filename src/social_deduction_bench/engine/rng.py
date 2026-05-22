"""Seeded engine RNG.

Upholds benchmark invariant #4: every stochastic operation derives from the
engine seed so the same seed yields an identical, replayable game. `GameRNG`
wraps a private `random.Random` instance and never touches global `random`
state.
"""

import random


class GameRNG:
    """A seeded random stream owned by the engine.

    Each instance holds its own `random.Random`; two instances with the same
    seed produce identical sequences, and draws on one never affect another.
    """

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._random = random.Random(seed)

    @property
    def seed(self) -> int:
        """The seed this stream was created from; read-only for replay safety."""
        return self._seed

    def shuffle[T](self, items: list[T]) -> list[T]:
        """Return a new shuffled list; the input is not mutated."""
        result = list(items)
        self._random.shuffle(result)
        return result

    def choice[T](self, items: list[T]) -> T:
        """Return one element drawn from `items`."""
        return self._random.choice(items)

    def sample[T](self, items: list[T], k: int) -> list[T]:
        """Return `k` elements drawn without replacement from `items`."""
        return self._random.sample(items, k)

    def randrange(self, stop: int) -> int:
        """Return a seed-derived int in [0, stop); upholds invariant #4 (no global random)."""
        return self._random.randrange(stop)
