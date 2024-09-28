#!/usr/bin/env python3
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from jax import random

from sps.gp import GP
from sps.kernels import matern_1_2, matern_3_2, matern_5_2, periodic, rbf
from sps.priors import Prior
from sps.utils import build_grid


# Parameters are based on Heaton, et al. benchmark: https://tinyurl.com/heaton-params
def main():
    rng = random.key(42)
    rng_sim, rng_noise = random.split(rng)
    s = build_grid(
        [
            {"start": 34.30, "stop": 37.07, "num": 300},
            {"start": -95.91, "stop": -91.28, "num": 500},
        ]
    )
    batch_size = 1
    approx = True
    f_mu = 44.49105
    lengthscales = [0.1, 0.15, 0.2]
    n = len(lengthscales)
    fig, axes = plt.subplots(n, 1, figsize=(5, 5 * n))
    for i, ls in enumerate(lengthscales):
        gp = GP(
            matern_1_2,
            var=Prior("fixed", {"value": 16.41}),
            ls=Prior("fixed", {"value": ls}),
        )
        f, *_ = gp.simulate(rng_sim, s, batch_size, approx)
        f += f_mu + jnp.sqrt(0.05) * random.normal(rng_noise, f.shape)
        axes[i].set_title(f"ls={ls}")
        axes[i].imshow(f.squeeze().reshape(300, 500), cmap="Spectral_r")
    plt.tight_layout()
    plt.savefig("2d_gp.png", dpi=150)
    plt.clf()
    df = pd.DataFrame(
        jnp.hstack([s.reshape(-1, 2), f[0].reshape(-1, 1)]),
        columns=["Lat", "Lon", "TrueTemp"],
    )
    df.to_csv("mat.csv", index=False)


if __name__ == "__main__":
    main()
