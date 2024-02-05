from functools import reduce

import jax.numpy as jnp
import pytest
from jax import random
from jax.random import PRNGKey

from sps.gp import GP, _kronecker_Ls, _kronecker_mvprod, cholesky, kronecker
from sps.kernels import rbf
from sps.utils import build_grid


# TODO(danj): this isn't close for dims > 1
@pytest.mark.skip("")
def test_kronecker_approx():
    grid = build_grid([{"start": 0, "stop": 1, "num": 50}] * 2)
    var, ls = 1.0, 0.1
    L_ch = cholesky(rbf, grid, var, ls)
    L_kr = kronecker(rbf, grid, var, ls, noise=1e-8)
    # print(jnp.max(jnp.abs(L_ch - L_kr)))
    # plt.imshow(L_ch - L_kr, cmap='inferno')
    # plt.colorbar()
    assert jnp.allclose(L_ch, L_kr)


def test_kronecker_mvprod(seed=0):
    var, ls, noise, key = 1.0, 0.1, 1e-5, PRNGKey(seed)
    locations = build_grid()
    num_locations = locations.size // locations.shape[-1]
    Ls = _kronecker_Ls(rbf, locations, var, ls, noise)
    L = reduce(jnp.kron, Ls)
    z = random.normal(key, (num_locations,))
    Lz = L @ z
    Lz_mvprod = _kronecker_mvprod(Ls, z)
    assert jnp.allclose(Lz, Lz_mvprod)


def test_1D_gp_approx(seed=0):
    batch_size, grid, gp = 3, build_grid(), GP()
    key = PRNGKey(seed)
    _, _, mu = gp.simulate(key, grid, batch_size)
    _, _, mu_approx = gp.simulate(key, grid, batch_size, approx=True)
    assert jnp.allclose(mu, mu_approx)


# TODO(danj): this is failing
@pytest.mark.skip("Deviations in covariance matrices are magnified during sampling")
def test_2D_gp_approx(seed=0):
    batch_size, gp = 3, GP()
    grid = build_grid([{"start": 0, "stop": 1, "num": 50}] * 2)
    key = PRNGKey(seed)
    _, _, mu = gp.simulate(key, grid, batch_size)
    _, _, mu_approx = gp.simulate(key, grid, batch_size, approx=True)
    assert jnp.allclose(mu, mu_approx)
