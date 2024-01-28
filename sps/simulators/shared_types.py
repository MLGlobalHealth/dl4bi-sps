from jaxtyping import Float, Array
from collections.abc import Callable

type Locations = Float[Array, "... D"]
type Variance = Float
type Lengthscale = Float
type Covariance = Float[Array, "N N"]
type Kernel = Callable[[Locations, Locations, Variance, Lengthscale], Covariance]
