from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, random
from jax.tree_util import Partial


@dataclass
class Prior:
    """Represents a prior using `jax.random` distributions.

    Args:
        dist: A distribution name from `jax.random` or `kernels` submodule.
        kwargs: A dict of the parameters for the given distribution.

    Returns:
        An instance of the Prior dataclass.
    """

    dist: str
    kwargs: dict[str, float]

    def __post_init__(self):
        dist_func = globals().get(self.dist, getattr(random, self.dist, None))
        self.dist_func = Partial(dist_func, **self.kwargs)

    def __hash__(self):
        return hash((self.dist, repr(self.kwargs)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def sample(self, key: Array, shape: Sequence[int] = (1,)) -> Array:
        """Samples this prior.

        Args:
            key: A psuedo-random number generator from `jax.random`.
            shape: Output shape of sample(s).

        Returns:
            A sample of shape `shape`.
        """
        return self.dist_func(key, shape=shape)


# JAX doesn't have a lambda parameterized exponential (28-01-2024)
# https://jax.readthedocs.io/en/latest/_autosummary/jax.random.exponential.html
def exponential(
    key: Array,
    lam: float,
    shape: Sequence[int],
) -> Array:
    """Exponential parameterized by lambda `lam`.

    Args:
        key: A psuedo-random number generator from `jax.random`.
        lam: Lambda of exponential distribution.
        shape: Output shape of sample(s).

    Returns:
        A sample of shape `shape`.
    """
    return 1 / lam * random.exponential(key, shape)


def fixed(
    key: Array,
    value: float,
    shape: Sequence[int],
) -> Array:
    """Fixed distribution.

    Args:
        key: A psuedo-random number generator from `jax.random`.
        value: A fixed value to return for all samples.
        shape: Output shape of sample(s).

    Returns:
        A fixed sample of shape `shape`.
    """
    return jnp.full(shape, value)
