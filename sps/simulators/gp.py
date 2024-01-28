import jax.numpy as jnp
from jax import config, random, vmap, jit
from dataclasses import dataclass
from jax.random import PRNGKey
from . import kernels
from .priors import Prior
from jax.typing import ArrayLike

# improves numerical stability for small lengthscales
config.update("jax_enable_x64", True)


# TODO:
# test creation from hydra/omegaconf
# implement kronecker method
# implement MMD


@dataclass
class GP:
    kernel: str = "rbf"
    variance: Prior = Prior("fixed", {"value": 1})
    lengthscale: Prior = Prior("beta", {"a": 2.5, "b": 6.0})
    seed: int = 0

    def __post_init__(self):
        self.kernel_func = getattr(kernels, self.kernel)
        self.key = PRNGKey(self.seed)

    def simulate(
        self,
        locations: ArrayLike,
        batch_size: int = 1,
        approx: bool = False,
    ):
        self.key, rng_var, rng_ls, rng_z = random.split(self.key, 4)
        var = self.variance.sample(rng_var, (batch_size,))
        ls = self.lengthscale.sample(rng_ls, (batch_size,))
        factorize = vmap(kronecker if approx else cholesky, in_axes=(None, None, 0, 0))
        Ls = factorize(self.kernel_func, locations, var, ls)
        zs = random.normal(rng_z, shape=(batch_size, locations.size))
        return vmap(jnp.dot)(Ls, zs)


def kronecker(
    kernel: kernels.Kernel,
    locations: ArrayLike,
    var: float,
    ls: float,
    noise: float = 1e-5,
) -> ArrayLike:
    # vmap(kernel, in_axes=(-1, None, None))
    # pass
    K = kernel(locations, locations, var, ls)
    return jnp.linalg.cholesky(K)


def cholesky(
    kernel: kernels.Kernel,
    locations: ArrayLike,
    var: float,
    ls: float,
    noise: float = 1e-5,
) -> ArrayLike:
    K = kernel(locations, locations, var, ls) + noise * jnp.eye(locations.size)
    return jnp.linalg.cholesky(K)
