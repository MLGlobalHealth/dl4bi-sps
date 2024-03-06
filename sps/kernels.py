from collections.abc import Callable

import jax.numpy as jnp
from jax import config, jit
from jax.typing import ArrayLike

# improves numerical stability for small lengthscales
config.update("jax_enable_x64", True)

Kernel = Callable[[ArrayLike, ArrayLike, float, float], ArrayLike]


@jit
def _prepare_dims(x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Prepares dims for use in kernel functions.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Two `[N, D]` dimensional arrays.
    """
    if x.ndim == 1:
        x = x[:, jnp.newaxis]
    if y.ndim == 1:
        y = y[:, jnp.newaxis]
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    return x, y


@jit
def l2_dist_sq(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """L2 distance between two [..., D] arrays.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Matrix of all pairwise distances.
    """
    x, y = _prepare_dims(x, y)
    dsq = (x**2).sum(-1)[:, None] + (y**2).sum(-1).T - 2 * x @ y.T
    # can produce small (negative) values on the diagonal,
    # e.g. distance d(x_i, x_i) = -1.2384e-8, so fill with 0s
    return jnp.fill_diagonal(dsq, 0, inplace=False)


@jit
def rbf(
    x: ArrayLike,
    y: ArrayLike,
    var: float,
    ls: float,
) -> ArrayLike:
    """Radial Basis kernel, aka Squared Exponential kernel.

    K(x, y) = var * exp{-||x-y||^2 / (2 * ls^2)}

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    return var * jnp.exp(-l2_dist_sq(x, y) / (2 * ls**2))


@jit
def matern_3_2(
    x: ArrayLike,
    y: ArrayLike,
    variance: float,
    lengthscale: float,
) -> ArrayLike:
    """Matern 3/2 kernel.

    K(x, y) = var * (1 + √3 * ||x-y|| / ls) * exp{-√3 * ||x-y|| / ls}

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    d = l2_dist_sq(x, y) ** (1 / 2)
    sqrt3 = 3.0 ** (1 / 2)
    return variance * (1 + sqrt3 * d / lengthscale) * jnp.exp(-sqrt3 * d / lengthscale)


@jit
def matern_5_2(
    x: ArrayLike,
    y: ArrayLike,
    variance: float,
    lengthscale: float,
) -> ArrayLike:
    """Matern 5/2 kernel.

    K(x, y) = var * (1 + √5 * ||x-y|| / ls + 5/3 * ||x-y||^2 / ls^2) * exp{-√5 * ||x-y|| / ls}

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        A covariance matrix.
    """
    dsq = l2_dist_sq(x, y)
    d = jnp.sqrt(dsq)
    sqrt5 = jnp.sqrt(5.0)
    return (
        variance
        * (1 + sqrt5 * d / lengthscale + 5 / 3 * dsq / lengthscale**2)
        * jnp.exp(-sqrt5 * d / lengthscale)
    )
