from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array, jit, lax, random

from .priors import Prior


@dataclass(frozen=True)
class MalariaState:
    """Immutable state for the malaria transmission simulator.

    Attributes:
        chi: Superinfection probability for each simulation, shape `[B]`.
        mu: Per-allele per-generation mutation rate, shape `[B]`.
        sw: Within-host selection coefficient, shape `[B]`.
        st: Transmission selection coefficient, shape `[B]`.
        p_init: Founding mutant allele frequency, shape `[B]`.
        mutant_count: Number of mutant alleles per host, shape `[B, Nh]`.
        n_alleles: Number of active allele slots per host, shape `[B, Nh]`.
    """

    chi: Array
    mu: Array
    sw: Array
    st: Array
    p_init: Array
    mutant_count: Array
    n_alleles: Array

    def parameter_values(self, Nh: int, Q: int) -> dict[str, Array]:
        """Return sampled parameters as an ordered mapping."""
        B = self.chi.shape[0]
        return {
            "Nh": jnp.full((B,), Nh, dtype=jnp.float32),
            "Q": jnp.full((B,), Q, dtype=jnp.float32),
            "chi": self.chi,
            "mu": self.mu,
            "sw": self.sw,
            "st": self.st,
            "p_init": self.p_init,
        }

    def parameter_matrix(self, Nh: int, Q: int) -> tuple[Array, tuple[str, ...]]:
        """Return sampled parameters as `[B, parameter]` rows plus names."""
        values = self.parameter_values(Nh, Q)
        return jnp.stack(tuple(values.values()), axis=-1), tuple(values.keys())


jax.tree_util.register_pytree_node(
    MalariaState,
    lambda d: (
        (d.chi, d.mu, d.sw, d.st, d.p_init, d.mutant_count, d.n_alleles),
        None,
    ),
    lambda _aux, children: MalariaState(*children),
)


@dataclass
class MalariaTransmission:
    """Forward-in-time malaria parasite transmission simulator.

    Args:
        Nh: Prior for the effective number of hosts. Used only when `Nh` is
            not supplied to `simulate`.
        Q: Prior for alleles transmitted per vector bite. Used only when `Q`
            is not supplied to `simulate`.
        chi: Prior over superinfection probabilities.
        mu: Prior over per-allele per-generation mutation rates.
        sw: Prior over within-host selection coefficients.
        st: Prior over transmission selection coefficients.
        p_init: Prior over founding mutant allele frequencies.
    """

    Nh: Prior = Prior("fixed", {"value": 1024})
    Q: Prior = Prior("randint", {"minval": 1, "maxval": 11})
    chi: Prior = Prior("uniform", {"minval": 0.0, "maxval": 1.0})
    mu: Prior = Prior("loguniform", {"minval": 1e-9, "maxval": 1e-7})
    sw: Prior = Prior("uniform", {"minval": -0.999, "maxval": 0.0})
    st: Prior = Prior("uniform", {"minval": 0.0, "maxval": 1.0})
    p_init: Prior = Prior("uniform", {"minval": 0.0, "maxval": 1.0})

    def sample_shape(self, rng: Array) -> tuple[int, int]:
        """Sample static `(Nh, Q)` values from their configured priors."""
        rng_nh, rng_q = random.split(rng)
        Nh = int(self.Nh.sample(rng_nh, (1,))[0])
        Q = int(self.Q.sample(rng_q, (1,))[0])
        return Nh, Q

    def initial_state(
        self,
        rng: Array,
        batch_size: int = 32,
        Nh: int = 1024,
        Q: int = 4,
    ) -> MalariaState:
        """Sample parameters and initialise the founding population."""
        if Nh < 2:
            raise ValueError("Nh must be at least 2 when superinfection is possible.")
        if Q < 1:
            raise ValueError("Q must be at least 1.")

        rng_chi, rng_mu, rng_sw, rng_st, rng_init, rng_founder = random.split(rng, 6)
        chi = self.chi.sample(rng_chi, (batch_size,)).astype(jnp.float32)
        mu = self.mu.sample(rng_mu, (batch_size,)).astype(jnp.float32)
        sw = self.sw.sample(rng_sw, (batch_size,)).astype(jnp.float32)
        st = self.st.sample(rng_st, (batch_size,)).astype(jnp.float32)
        p_init = self.p_init.sample(rng_init, (batch_size,)).astype(jnp.float32)

        max_alleles = 2 * Q
        founder = random.bernoulli(
            rng_founder,
            p_init[:, None, None],
            shape=(batch_size, Nh, Q),
        ).astype(jnp.float32)
        mutant_count = jnp.sum(founder, axis=-1)
        n_alleles = jnp.full((batch_size, Nh), Q, dtype=jnp.int32)
        return MalariaState(chi, mu, sw, st, p_init, mutant_count, n_alleles)

    def simulate(
        self,
        rng: Array,
        num_generations: int = 512,
        batch_size: int = 32,
        Nh: Optional[int] = None,
        Q: Optional[int] = None,
        state: Optional[MalariaState] = None,
        include_initial: bool = False,
    ) -> tuple[Array, MalariaState]:
        """Simulate per-host allele-frequency trajectories.

        Args:
            rng: Pseudo-random key.
            num_generations: Number of generation transitions to simulate.
            batch_size: Number of independent simulations in the batch.
            Nh: Static number of hosts. If omitted, sampled from `self.Nh`.
            Q: Static transmission bottleneck size. If omitted, sampled from
                `self.Q`.
            state: Optional state to continue from.
            include_initial: Whether to include the founding generation before
                transmission transitions in the returned trajectory.

        Returns:
            Tuple of allele frequencies with shape `[B, T, Nh]`, or
            `[B, T + 1, Nh]` when `include_initial=True`, and the final state.
        """
        if state is None:
            rng_shape, rng_init, rng_sim = random.split(rng, 3)
            if Nh is None or Q is None:
                sampled_Nh, sampled_Q = self.sample_shape(rng_shape)
                Nh = sampled_Nh if Nh is None else Nh
                Q = sampled_Q if Q is None else Q
            state = self.initial_state(rng_init, batch_size, Nh, Q)
        else:
            rng_sim = rng
            if Q is None:
                Q = int(jnp.max(state.n_alleles)) // 2

        return _simulate(rng_sim, state, num_generations, Q, include_initial)


@partial(jit, static_argnames=("num_generations", "Q", "include_initial"))
def _simulate(
    rng: Array,
    state: MalariaState,
    num_generations: int,
    Q: int,
    include_initial: bool,
) -> tuple[Array, MalariaState]:
    """Run the jitted simulator loop."""
    rng_steps = random.split(rng, num_generations)

    def step(carry: MalariaState, rng_step: Array):
        next_state = _transmit_generation(rng_step, carry, Q)
        return next_state, _allele_frequency(next_state)

    final_state, freq_steps = lax.scan(step, state, rng_steps)
    steps = jnp.moveaxis(freq_steps, 0, 1)
    allele_freq = (
        jnp.concatenate([_allele_frequency(state)[:, None, :], steps], axis=1)
        if include_initial
        else steps
    )
    return allele_freq, final_state


def _allele_frequency(state: MalariaState) -> Array:
    return state.mutant_count / state.n_alleles.astype(jnp.float32)


def _sample_second_parent(
    rng: Array,
    probs: Array,
    pa: Array,
    superinfected: Array,
) -> Array:
    Nh = probs.shape[-1]
    rng_excluded, rng_unrestricted = random.split(rng)
    cdf = jnp.cumsum(probs, axis=-1)
    p_pa = jnp.take_along_axis(probs, pa, axis=-1)
    cdf_pa = jnp.take_along_axis(cdf, pa, axis=-1)
    cdf_before_pa = cdf_pa - p_pa
    u = random.uniform(rng_excluded, pa.shape) * (1.0 - p_pa)
    u = jnp.where(u < cdf_before_pa, u, u + p_pa)
    excluded = jnp.minimum(_searchsorted_batched(cdf, u), Nh - 1)
    unrestricted = _sample_parent_indices(rng_unrestricted, probs)
    return jnp.where(superinfected, excluded, unrestricted)


def _sample_parent_indices(rng: Array, probs: Array) -> Array:
    Nh = probs.shape[-1]
    u = random.uniform(rng, probs.shape)
    return jnp.minimum(_searchsorted_batched(jnp.cumsum(probs, axis=-1), u), Nh - 1)


def _searchsorted_batched(cdf: Array, values: Array) -> Array:
    return jax.vmap(lambda c, v: jnp.searchsorted(c, v, side="right"))(cdf, values)


def _gather_hosts(values: Array, host_idx: Array) -> Array:
    B = values.shape[0]
    batch_idx = jnp.arange(B)[:, None]
    return values[batch_idx, host_idx]


def _sample_parent_mutants(
    rng: Array,
    parent_mutants: Array,
    parent_n_alleles: Array,
    Q: int,
) -> Array:
    p = parent_mutants / parent_n_alleles.astype(jnp.float32)
    return random.binomial(rng, Q, p).astype(jnp.float32)


def _within_host_resample(
    rng: Array,
    mutant_count: Array,
    n_alleles: Array,
    sw: Array,
) -> Array:
    wild_count = n_alleles.astype(jnp.float32) - mutant_count
    mutant_weight = (1.0 + sw[:, None]) * mutant_count
    total_weight = wild_count + mutant_weight
    p = jnp.where(total_weight > 0.0, mutant_weight / total_weight, 0.0)
    resampled = random.binomial(rng, n_alleles, p).astype(jnp.float32)
    return jnp.where(sw[:, None] == 0.0, mutant_count, resampled)


def _transmit_generation(rng: Array, state: MalariaState, Q: int) -> MalariaState:
    rng_super, rng_pa, rng_pb, rng_pool_a, rng_pool_b, rng_resample, rng_mut = (
        random.split(rng, 7)
    )
    B, Nh = state.mutant_count.shape

    allele_freq = _allele_frequency(state)
    host_weights = 1.0 + state.st[:, None] * allele_freq
    host_probs = host_weights / jnp.sum(host_weights, axis=-1, keepdims=True)

    superinfected = random.bernoulli(rng_super, state.chi[:, None], (B, Nh))
    pa = _sample_parent_indices(rng_pa, host_probs)
    pb = _sample_second_parent(rng_pb, host_probs, pa, superinfected)

    parent_a = _gather_hosts(state.mutant_count, pa)
    parent_b = _gather_hosts(state.mutant_count, pb)
    parent_a_n = _gather_hosts(state.n_alleles, pa)
    parent_b_n = _gather_hosts(state.n_alleles, pb)

    pool_a = _sample_parent_mutants(rng_pool_a, parent_a, parent_a_n, Q)
    pool_b = _sample_parent_mutants(rng_pool_b, parent_b, parent_b_n, Q)

    mutant_count = pool_a + jnp.where(superinfected, pool_b, 0.0)
    n_alleles = jnp.where(superinfected, 2 * Q, Q).astype(jnp.int32)

    mutant_count = _within_host_resample(
        rng_resample,
        mutant_count,
        n_alleles,
        state.sw,
    )
    rng_mutant, rng_wild = random.split(rng_mut)
    mutant_flips = random.binomial(rng_mutant, mutant_count, state.mu[:, None])
    wild_flips = random.binomial(
        rng_wild,
        n_alleles.astype(jnp.float32) - mutant_count,
        state.mu[:, None],
    )
    mutant_count = mutant_count - mutant_flips + wild_flips

    return MalariaState(
        state.chi,
        state.mu,
        state.sw,
        state.st,
        state.p_init,
        mutant_count,
        n_alleles,
    )
