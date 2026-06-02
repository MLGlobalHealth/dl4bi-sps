import numpy as np
import pytest
from jax import random
from omegaconf import OmegaConf

from dl4bi_sps.malaria import (
    MalariaHistogramState,
    MalariaSimulationBatch,
    MalariaTransmission,
    host_state_table,
)
from dl4bi_sps.priors import Prior


def test_malaria_shapes_and_bounds():
    """Verify malaria histogram trajectories have expected shape and bounds."""
    rng = random.key(42)
    sim = MalariaTransmission()
    num_hosts_by_state, state_num_mutants, state_num_alleles, state = (
        sim.simulate(
            rng,
            num_generations=5,
            num_simulations=4,
            num_hosts=16,
            num_transmitted_alleles=2,
        )
    )

    num_hosts_by_state = np.asarray(num_hosts_by_state)
    assert num_hosts_by_state.shape == (4, 5, 8)
    assert state_num_mutants.shape == (8,)
    assert state_num_alleles.shape == (8,)
    assert state.num_hosts_by_state.shape == (4, 8)
    assert state.num_transmitted_alleles == 2
    np.testing.assert_array_equal(
        num_hosts_by_state.sum(axis=-1),
        np.full((4, 5), 16),
    )


def test_malaria_deterministic_repeat():
    """The same key and configuration should reproduce identical histograms."""
    rng = random.key(123)
    sim = MalariaTransmission()
    hist_a, _, _, state_a = sim.simulate(
        rng,
        num_generations=4,
        num_simulations=3,
        num_hosts=12,
        num_transmitted_alleles=2,
    )
    hist_b, _, _, state_b = sim.simulate(
        rng,
        num_generations=4,
        num_simulations=3,
        num_hosts=12,
        num_transmitted_alleles=2,
    )

    np.testing.assert_array_equal(np.asarray(hist_a), np.asarray(hist_b))
    np.testing.assert_array_equal(
        np.asarray(state_a.num_hosts_by_state),
        np.asarray(state_b.num_hosts_by_state),
    )


def test_malaria_always_starts_with_initial_generation():
    """The first returned generation is the founding histogram."""
    rng = random.key(7)
    rng_init, _ = random.split(rng)
    sim = MalariaTransmission()
    initial = sim.initial_state(
        rng_init,
        num_simulations=2,
        num_hosts=8,
        num_transmitted_alleles=2,
    )

    num_hosts_by_state, _, _, state = sim.simulate(
        rng,
        num_generations=1,
        num_simulations=2,
        num_hosts=8,
        num_transmitted_alleles=2,
    )

    np.testing.assert_array_equal(
        np.asarray(num_hosts_by_state[:, 0, :]),
        np.asarray(initial.num_hosts_by_state),
    )
    np.testing.assert_array_equal(
        np.asarray(state.num_hosts_by_state),
        np.asarray(initial.num_hosts_by_state),
    )


def test_malaria_fixed_priors_in_parameters():
    """Fixed priors should be reflected in the state parameter matrix."""
    sim = MalariaTransmission(
        chi=Prior("fixed", {"value": 0.25}),
        mu=Prior("fixed", {"value": 1e-8}),
        sw=Prior("fixed", {"value": -0.1}),
        st=Prior("fixed", {"value": 0.2}),
        p_init=Prior("fixed", {"value": 0.3}),
    )
    _, _, _, state = sim.simulate(
        random.key(0),
        num_generations=2,
        num_simulations=5,
        num_hosts=10,
        num_transmitted_alleles=3,
    )
    params_jax, param_names = state.parameter_matrix(10)
    params = np.asarray(params_jax)

    assert param_names == (
        "num_hosts",
        "num_transmitted_alleles",
        "chi",
        "mu",
        "sw",
        "st",
        "p_init",
    )
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
    sim = MalariaTransmission(num_hosts=Prior("choice", {"values": [1024, 2048]}))
    samples = {sim.sample_shape(random.key(seed))[0] for seed in range(16)}

    assert samples <= {1024, 2048}
    assert samples


def test_malaria_zarr_smoke(tmp_path):
    """Verify the intended xarray/Zarr data can be written and read."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    from examples.build_malaria_dataset import make_histogram_dataset

    sim = MalariaTransmission()
    batch = sim.simulate_batch(
        random.key(1),
        num_generations=3,
        num_simulations=2,
        num_hosts=8,
        num_transmitted_alleles=2,
    )
    store = tmp_path / "malaria.zarr"
    group = "num_hosts=8/num_transmitted_alleles=2"

    ds = make_histogram_dataset(
        batch,
        np.arange(2, dtype=np.int64),
        np.arange(3, dtype=np.int32),
        seed=1,
    )
    ds.to_zarr(store, group=group, mode="a", consolidated=False)

    opened = xr.open_zarr(store, group=group, consolidated=False)
    assert opened["num_hosts_by_state"].shape == (2, 3, 8)
    assert opened["parameters"].shape == (2, len(batch.parameter_names))
    assert opened["num_hosts_by_state"].dtype == np.dtype("uint16")
    np.testing.assert_array_equal(
        opened["num_hosts_by_state"].sum("state").values,
        np.full((2, 3), 8, dtype=np.uint16),
    )
    allele_freq = (
        opened["num_hosts_by_state"] * opened["state_allele_freq"]
    ).sum("state") / opened.attrs["num_hosts"]
    assert allele_freq.shape == (2, 3)


def test_malaria_histogram_first_generation_matches_founder_distribution():
    """The first histogram generation should represent the founding population."""
    sim = MalariaTransmission(
        chi=Prior("fixed", {"value": 0.0}),
        mu=Prior("fixed", {"value": 0.0}),
        sw=Prior("fixed", {"value": 0.0}),
        st=Prior("fixed", {"value": 0.0}),
        p_init=Prior("fixed", {"value": 0.25}),
    )
    num_hosts_by_state, _, _, _ = (
        sim.simulate(
            random.key(1),
            num_generations=1,
            num_simulations=256,
            num_hosts=256,
            num_transmitted_alleles=2,
        )
    )

    observed = np.asarray(num_hosts_by_state[:, 0, :]).mean(axis=0) / 256
    expected = np.asarray([0.5625, 0.375, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(observed, expected, atol=0.03)


def test_malaria_zarr_append_reuses_existing_encoding(tmp_path):
    """Appending a second batch should not resubmit encoding for existing variables."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    from examples.build_malaria_dataset import _write_dataset, make_histogram_dataset

    param_names = MalariaHistogramState.parameter_names()
    state_num_mutants, state_num_alleles = host_state_table(2)
    num_hosts_by_state = np.zeros((1, 3, state_num_mutants.size), dtype=np.uint16)
    num_hosts_by_state[..., 0] = 8
    first = make_histogram_dataset(
        _batch_from_counts(num_hosts_by_state, state_num_mutants, state_num_alleles),
        np.asarray([0], dtype=np.int64),
        np.arange(3, dtype=np.int32),
        seed=0,
    )
    second_num_hosts_by_state = np.zeros_like(num_hosts_by_state)
    second_num_hosts_by_state[..., 1] = 8
    second = make_histogram_dataset(
        _batch_from_counts(
            second_num_hosts_by_state,
            state_num_mutants,
            state_num_alleles,
        ),
        np.asarray([1], dtype=np.int64),
        np.arange(3, dtype=np.int32),
        seed=0,
    )
    store = tmp_path / "malaria.zarr"
    group = "num_hosts=8/num_transmitted_alleles=2"
    output_cfg = OmegaConf.create(
        {
            "chunks": {
                "num_simulations": 1,
                "num_generations": 2,
                "num_states": 4,
            },
            "compression": {
                "enabled": True,
                "codec": "blosc",
                "cname": "zstd",
                "clevel": 5,
                "shuffle": "shuffle",
            },
            "write_empty_chunks": False,
        }
    )

    _write_dataset(first, store, group, append=False, output_cfg=output_cfg)
    _write_dataset(second, store, group, append=True, output_cfg=output_cfg)

    opened = xr.open_zarr(store, group=group, consolidated=False)
    assert opened["num_hosts_by_state"].shape == (2, 3, state_num_mutants.size)
    assert opened["parameters"].shape == (2, len(param_names))
    np.testing.assert_array_equal(opened["simulation"].values, np.asarray([0, 1]))
    np.testing.assert_array_equal(
        opened["num_hosts_by_state"].sum("state").values,
        np.full((2, 3), 8, dtype=np.uint16),
    )
    allele_freq = (
        opened["num_hosts_by_state"] * opened["state_allele_freq"]
    ).sum("state") / opened.attrs["num_hosts"]
    assert allele_freq.shape == (2, 3)


def _batch_from_counts(
    num_hosts_by_state: np.ndarray,
    state_num_mutants: np.ndarray,
    state_num_alleles: np.ndarray,
) -> MalariaSimulationBatch:
    return MalariaSimulationBatch(
        num_hosts_by_state=num_hosts_by_state,
        state_num_mutants=state_num_mutants,
        state_num_alleles=state_num_alleles,
        parameters=np.asarray(
            [[8.0, 2.0, 0.1, 1e-8, -0.1, 0.2, 0.3]],
            dtype=np.float32,
        ),
        parameter_names=MalariaHistogramState.parameter_names(),
        num_hosts=8,
        num_transmitted_alleles=2,
    )
