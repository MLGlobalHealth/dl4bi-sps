import jax.numpy as jnp
from jaxtyping import Num, Float, Array
from collections.abc import Sequence


def build_grid(
    axes: Sequence[dict[str, Num]] = [{"start": 0, "stop": 1, "num": 128}],
) -> Float[Array, "..."]:
    pts = [jnp.linspace(**axis) for axis in axes]
    return jnp.stack(jnp.meshgrid(*pts, indexing="ij"), axis=-1)


def scale_grid(grid: Float[Array, "..."], factor: Num) -> Float[Array, "..."]:
    axes = [
        jnp.linspace(grid[..., axis].min(), grid[..., axis].max(), int(n * factor))
        for axis, n in enumerate(grid.shape[:-1])
    ]
    return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
