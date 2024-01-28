from dataclasses import dataclass
import jax.numpy as jnp
from jax import random, Array
from jax.tree_util import Partial
from collections.abc import Sequence
from jax.typing import ArrayLike


@dataclass
class Prior:
    """Represents a prior using `jax.random` distributions."""

    dist: str
    kwargs: dict[str, float]

    def __post_init__(self):
        dist_func = globals().get(self.dist, getattr(random, self.dist, None))
        self.dist_func = Partial(dist_func, **self.kwargs)

    def __hash__(self):
        return hash((self.dist, repr(self.kwargs)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def sample(self, key: Array, shape: Sequence[int]) -> ArrayLike:
        return self.dist_func(key, shape=shape)


# JAX doesn't have a lambda parameterized exponential (28-01-2024)
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.exponential.html
def exponential(
    key: Array,
    lam: float,
    shape: Sequence[int],
) -> ArrayLike:
    """Exponential parameterized by lambda `lam`."""
    return 1 / lam * random.exponential(key, shape)


def fixed(
    key: Array,
    value: float,
    shape: Sequence[int],
) -> ArrayLike:
    """Fixed distribution."""
    return jnp.full(shape, value)
