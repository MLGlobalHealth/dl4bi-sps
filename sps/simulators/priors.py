from dataclasses import dataclass
import jax.numpy as jnp
from jax import random, jit
from jax.tree_util import Partial
from typing import Any, Sequence
from jax.random import PRNGKey
from jaxtyping import Float, Array, PRNGKeyArray, Num
from . import shared_types as T


@dataclass
class Prior:
    dist: str
    kwargs: dict[str, Num]

    def __post_init__(self):
        dist_func = getattr(globals(), self.dist, getattr(random, self.dist))
        self.dist_func = jit(Partial(dist_func, **self.kwargs))

    def sample(self, key: PRNGKeyArray, shape: Sequence[int]) -> Float[Array, "..."]:
        return self.dist_func(key, shape=shape)


# JAX doesn't have a lambda parameterized exponential (28-01-2024)
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.exponential.html
def exponential(
    key: PRNGKeyArray,
    lam: Float,
    shape: Sequence[int],
) -> Float[Array, "..."]:
    return 1 / lam * random.exponential(key, shape)


def fixed(
    key: PRNGKeyArray,
    value: Float,
    shape: Sequence[int],
) -> Float[Array, "..."]:
    return jnp.full(shape, value)
