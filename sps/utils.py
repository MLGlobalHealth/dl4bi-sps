from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def build_grid(
    axes: Sequence[dict[str, float]] = [{"start": 0, "stop": 1, "num": 128}],
) -> Array:
    """Builds a grid of shape `[..., D]` along the axes using `jnp.linspace`."""
    pts = [jnp.linspace(**axis) for axis in axes]
    return jnp.stack(jnp.meshgrid(*pts, indexing="ij"), axis=-1)


def scale_grid(grid: ArrayLike, factor: int) -> Array:
    """Scales the `grid` of shape `[..., D]` by `factor` along all axes."""
    axes = [
        jnp.linspace(grid[..., dim].min(), grid[..., dim].max(), int(n * factor))
        for dim, n in enumerate(grid.shape[:-1])
    ]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
