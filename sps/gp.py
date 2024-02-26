from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, config, lax, random, vmap
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

    def __post_init__(self):
        self.kernel_func = getattr(kernels, self.kernel)

    def simulate(
        self,
        key: Array,
        locations: ArrayLike,  # [..., D]
        batch_size: int = 1,
        approx: bool = False,
    ) -> tuple[Array, Array, Array]:
        """Simulate `batch_size` realizations of the GP at `locations`."""
        rng_var, rng_ls, rng_z = random.split(key, 3)
        num_locations = locations.size // locations.shape[-1]
        var = self.variance.sample(rng_var, (batch_size,))
        ls = self.lengthscale.sample(rng_ls, (batch_size,))
        z = random.normal(rng_z, shape=(batch_size, num_locations))
        print(var[:5], ls[:5], z[:5])
        vsample = vmap(kronecker if approx else cholesky, in_axes=(None, None, 0, 0, 0))
        mu = vsample(self.kernel_func, locations, var, ls, z)
        mu = mu.reshape(-1, *locations.shape[:-1], 1)  # batch x grid x 1
        return var, ls, mu


def cholesky(
    kernel: kernels.Kernel,
    locations: ArrayLike,  # [..., D]
    var: float,
    ls: float,
    z: Array,
    noise: float = 1e-5,
) -> Array:
    """Cholesky kernel covariance factorization."""
    num_locations = locations.size // locations.shape[-1]
    K = kernel(locations, locations, var, ls) + noise * jnp.eye(num_locations)
    L = jnp.linalg.cholesky(K)
    return L @ z


def kronecker(
    kernel: kernels.Kernel,
    locations: ArrayLike,  # [..., D]
    var: float,
    ls: float,
    z: Array,
    noise: float = 1e-5,
) -> Array:
    """Kronecker kernel covariance factorization."""
    Ls = _kronecker_Ls(kernel, locations, var, ls, noise)
    return _kronecker_mvprod(Ls, z)


def _kronecker_Ls(
    kernel: kernels.Kernel,
    locations: ArrayLike,
    var: float,
    ls: float,
    noise: float = 1e-5,
) -> Sequence[Array]:
    D, Ls = locations.shape[-1], []
    start, stop = jnp.zeros(D, dtype=int), jnp.ones(D, dtype=int)
    for dim, dim_size in enumerate(locations.shape[:-1]):
        _stop = stop.at[dim].set(dim_size)
        axis = lax.slice(locations[..., dim], start, _stop).squeeze()[..., jnp.newaxis]
        K = kernel(axis, axis, var, ls) + noise * jnp.eye(dim_size)
        Ls += [jnp.linalg.cholesky(K)]
    return Ls


def _kronecker_mvprod(Ls: Sequence[Array], z: Array) -> Array:
    """Linear Kronecker product of `Ls` with vector `z`.

    Source: https://mlg.eng.cam.ac.uk/pub/pdf/Saa11.pdf p137
    """
    x, N = z, z.size
    for L in reversed(Ls):
        D = L.shape[0]
        X = x.reshape(D, N // D)
        Z = L @ X
        x = Z.T.flatten()  # does this need to be column-wise? Z.flatten()
    return x
