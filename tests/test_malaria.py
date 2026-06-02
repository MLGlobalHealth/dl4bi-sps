import numpy as np
import pytest
from jax import random

from dl4bi_sps.malaria import MalariaTransmission
from dl4bi_sps.priors import Prior


def test_malaria_shapes_and_bounds():
    """Verify malaria simulator trajectories have expected shape and bounds."""
    rng = random.key(42)
    sim = MalariaTransmission()
    allele_freq, state = sim.simulate(
        rng,
        num_generations=5,
        batch_size=4,
        Nh=16,
        Q=2,
    )

    allele_freq = np.asarray(allele_freq)
    assert allele_freq.shape == (4, 5, 16)
    assert state.mutant_count.shape == (4, 16)
    assert state.n_alleles.shape == (4, 16)
    assert np.isfinite(allele_freq).all()
    assert ((allele_freq >= 0.0) & (allele_freq <= 1.0)).all()


def test_malaria_deterministic_repeat():
    """The same key and configuration should reproduce identical trajectories."""
    rng = random.key(123)
    sim = MalariaTransmission()
    freq_a, state_a = sim.simulate(rng, num_generations=4, batch_size=3, Nh=12, Q=2)
    freq_b, state_b = sim.simulate(rng, num_generations=4, batch_size=3, Nh=12, Q=2)

    np.testing.assert_array_equal(np.asarray(freq_a), np.asarray(freq_b))
    np.testing.assert_array_equal(
        np.asarray(state_a.mutant_count),
        np.asarray(state_b.mutant_count),
    )
    np.testing.assert_array_equal(
        np.asarray(state_a.n_alleles),
        np.asarray(state_b.n_alleles),
    )


def test_malaria_can_include_initial_generation():
    """Initial-state output is opt-in and adds one recorded generation."""
    freq, _ = MalariaTransmission().simulate(
        random.key(7),
        num_generations=5,
        batch_size=2,
        Nh=8,
        Q=2,
        include_initial=True,
    )
    assert np.asarray(freq).shape == (2, 6, 8)


def test_malaria_fixed_priors_in_parameters():
    """Fixed priors should be reflected in the state parameter matrix."""
    sim = MalariaTransmission(
        chi=Prior("fixed", {"value": 0.25}),
        mu=Prior("fixed", {"value": 1e-8}),
        sw=Prior("fixed", {"value": -0.1}),
        st=Prior("fixed", {"value": 0.2}),
        p_init=Prior("fixed", {"value": 0.3}),
    )
    _, state = sim.simulate(random.key(0), num_generations=2, batch_size=5, Nh=10, Q=3)
    params_jax, param_names = state.parameter_matrix(10, 3)
    params = np.asarray(params_jax)

    assert param_names == ("Nh", "Q", "chi", "mu", "sw", "st", "p_init")
    assert params.shape == (5, len(param_names))
    np.testing.assert_array_equal(params[:, 0], np.full(5, 10.0, dtype=np.float32))
    np.testing.assert_array_equal(params[:, 1], np.full(5, 3.0, dtype=np.float32))
    np.testing.assert_allclose(params[:, 2], 0.25)
    np.testing.assert_allclose(params[:, 3], 1e-8)
    np.testing.assert_allclose(params[:, 4], -0.1)
    np.testing.assert_allclose(params[:, 5], 0.2)
    np.testing.assert_allclose(params[:, 6], 0.3)


def test_malaria_shape_sampling_uses_choice_prior():
    """Static host counts can be sampled from a finite configured set."""
    sim = MalariaTransmission(Nh=Prior("choice", {"values": [1024, 2048]}))
    samples = {sim.sample_shape(random.key(seed))[0] for seed in range(16)}

    assert samples <= {1024, 2048}
    assert samples


def test_malaria_zarr_smoke(tmp_path):
    """Verify the intended xarray/Zarr schema can be written and read."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")

    sim = MalariaTransmission()
    allele_freq, state = sim.simulate(
        random.key(1),
        num_generations=3,
        batch_size=2,
        Nh=8,
        Q=2,
    )
    allele_freq = np.asarray(allele_freq, dtype=np.float32)
    params_jax, param_names = state.parameter_matrix(8, 2)
    params = np.asarray(params_jax, dtype=np.float32)
    store = tmp_path / "malaria.zarr"
    group = "nh=8/q=2"

    ds = xr.Dataset(
        data_vars={
            "allele_freq": (("simulation", "generation", "host"), allele_freq),
            "parameters": (("simulation", "parameter"), params),
        },
        coords={
            "simulation": np.arange(2, dtype=np.int64),
            "generation": np.arange(3, dtype=np.int32),
            "host": np.arange(8, dtype=np.int32),
            "parameter": np.asarray(param_names, dtype=object),
        },
    )
    ds.to_zarr(store, group=group, mode="a")

    opened = xr.open_zarr(store, group=group)
    assert opened["allele_freq"].shape == (2, 3, 8)
    assert opened["parameters"].shape == (2, len(param_names))
