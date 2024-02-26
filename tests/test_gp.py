from functools import reduce

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jax.random import PRNGKey

from sps.gp import GP, _kronecker_Ls, _kronecker_mvprod
from sps.kernels import rbf
from sps.utils import build_grid


def test_factorizations(var=1.0, ls=0.1, num_dims=2, dim_size=32, noise=1e-6):
    locations = build_grid([{"start": 0, "stop": 1, "num": dim_size}] * num_dims)
    K = rbf(locations, locations, var, ls) + noise * jnp.eye(dim_size * dim_size)
    L_ch = jnp.linalg.cholesky(K)
    Ls_kr = _kronecker_Ls(rbf, locations, var, ls, noise / num_dims)
    L_kr = reduce(jnp.kron, Ls_kr)
    K_ch = L_ch @ L_ch.T
    K_kr = L_kr @ L_kr.T
    assert jnp.allclose(K, K_ch)
    assert jnp.allclose(K, K_kr)


def test_kronecker_mvprod(seed=0, var=1.0, ls=0.1, num_dims=2, dim_size=32):
    locations = build_grid([{"start": 0, "stop": 1, "num": dim_size}] * num_dims)
    num_locations = locations.size // locations.shape[-1]
    Ls = _kronecker_Ls(rbf, locations, var, ls)
    L = reduce(jnp.kron, Ls)
    z = random.normal(random.key(seed), (num_locations,))
    Lz = L @ z
    Lz_mvprod = _kronecker_mvprod(Ls, z)
    assert jnp.allclose(Lz, Lz_mvprod)


def test_1D_gp_approx(seed=0, dim_size=32, batch_size=3):
    grid = build_grid([{"start": 0, "stop": 1, "num": dim_size}])
    gp = GP()
    key = PRNGKey(seed)
    _, _, mu = gp.simulate(key, grid, batch_size)
    _, _, mu_approx = gp.simulate(key, grid, batch_size, approx=True)
    assert jnp.allclose(mu, mu_approx)


def test_2D_gp_approx(seed=0, num_dims=2, dim_size=32, batch_size=3):
    locations = build_grid([{"start": 0, "stop": 1, "num": dim_size}] * num_dims)
    gp = GP()
    key = PRNGKey(seed)
    _, _, mu = gp.simulate(key, locations, batch_size)
    _, _, mu_approx = gp.simulate(key, locations, batch_size, approx=True)
    for i in range(1, batch_size + 1):
        plt.imshow(mu[i] - mu_approx[i], cmap="inferno")
        plt.colorbar()
        plt.savefig(f"mu_diff_2D_{i}.png")
        plt.clf()
    assert jnp.allclose(mu, mu_approx)
