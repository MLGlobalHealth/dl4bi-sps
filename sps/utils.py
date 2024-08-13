from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def build_grid(
    axes: Sequence[dict[str, float]] = [{"start": 0, "stop": 1, "num": 128}],
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Builds a grid of shape `[..., D]` along the axes using `jnp.linspace`.

    Args:
        axes: A list of dicts, each with keys `start`, `stop`, and `num`, which
            are passed to `jnp.linspace`.

    Returns:
        A mesh grid across those axes.
    """
    pts = [jnp.linspace(**axis, dtype=dtype) for axis in axes]
    return jnp.stack(jnp.meshgrid(*pts, indexing="ij"), axis=-1)


def scale_grid(grid: ArrayLike, factor: int) -> Array:
    """Scales the `grid` of shape `[..., D]` by `factor` along all axes.

    Args:
        grid: A mesh grid.
        factor: A factor by which to scale each dimension of the grid.

    Returns:
        A scaled grid.
    """
    axes = [
        jnp.linspace(grid[..., dim].min(), grid[..., dim].max(), int(n * factor))
        for dim, n in enumerate(grid.shape[:-1])
    ]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
