from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, config, lax, random, vmap
from jax.random import PRNGKey
from jax.typing import ArrayLike

from . import kernels
from .priors import Prior

# improves numerical stability for small lengthscales
config.update("jax_enable_x64", True)


@dataclass
class GP:
    """Gaussian Process simulator."""

    kernel: str = "matern_3_2"
    variance: Prior = Prior("fixed", {"value": 1})
    lengthscale: Prior = Prior("beta", {"a": 2.5, "b": 6.0})
    seed: int = 0

    def __post_init__(self):
        self.kernel_func = getattr(kernels, self.kernel)
        self.key = PRNGKey(self.seed)

    def simulate(
        self,
        locations: ArrayLike,  # [..., D]
        batch_size: int = 1,
        approx: bool = False,
    ) -> tuple[Array, Array, Array]:
        """Simulate `batch_size` realizations of the GP at `locations`."""
        self.key, rng_var, rng_ls, rng_z = random.split(self.key, 4)
        factorize = vmap(kronecker if approx else cholesky, in_axes=(None, None, 0, 0))
        num_locations = locations.size // locations.shape[-1]
        var = self.variance.sample(rng_var, (batch_size,))
        ls = self.lengthscale.sample(rng_ls, (batch_size,))
        Ls = factorize(self.kernel_func, locations, var, ls)
        zs = random.normal(rng_z, shape=(batch_size, num_locations))
        mu = vmap(jnp.dot)(Ls, zs).reshape(-1, *locations.shape[:-1], 1)
        return var, ls, mu


def kronecker(
    kernel: kernels.Kernel,
    locations: ArrayLike,  # [..., D]
    var: float,
    ls: float,
    noise: float = 1e-5,
) -> Array:
    """Kronecker kernel covariance factorization."""
    D, Ks = locations.shape[-1], []
    start, stop = jnp.zeros(D, dtype=int), jnp.ones(D, dtype=int)
    for dim, dim_size in enumerate(locations.shape[:-1]):
        _stop = stop.at[dim].set(dim_size)
        axis = lax.slice(locations[..., dim], start, _stop).squeeze()[..., jnp.newaxis]
        Ks += [kernel(axis, axis, var, ls) + noise * jnp.eye(dim_size)]
    L = jnp.linalg.cholesky(Ks[0])
    for K_i in Ks[1:]:
        L_i = jnp.linalg.cholesky(K_i)
        L = jnp.kron(L, L_i)
    return L


def cholesky(
    kernel: kernels.Kernel,
    locations: ArrayLike,  # [..., D]
    var: float,
    ls: float,
    noise: float = 1e-5,
) -> Array:
    """Cholesky kernel covariance factorization."""
    num_locations = locations.size // locations.shape[-1]
    K = kernel(locations, locations, var, ls) + noise * jnp.eye(num_locations)
    return jnp.linalg.cholesky(K)
