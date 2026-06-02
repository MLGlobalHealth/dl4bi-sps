#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

import hydra
import jax
import numpy as np
import xarray as xr
import zarr
from jax import random
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dl4bi_sps.malaria import MalariaTransmission
from dl4bi_sps.priors import Prior


@hydra.main(version_base=None, config_path="configs", config_name="malaria_dataset")
def main(cfg: DictConfig) -> None:
    """Generate malaria transmission simulations and write them to Zarr."""
    store = Path(cfg.output.store)
    if store.exists():
        if cfg.output.overwrite:
            shutil.rmtree(store)
        else:
            raise FileExistsError(
                f"{store} already exists. Set output.overwrite=true to replace it."
            )
    store.parent.mkdir(parents=True, exist_ok=True)

    simulator = _build_simulator(cfg)
    rng = random.key(cfg.seed)
    next_sample_id = 0
    group_counts: dict[str, int] = {}
    group_records: list[dict] = []
    param_names: tuple[str, ...] | None = None
    total_chunks = int(cfg.num_chunks)
    progress = tqdm(total=total_chunks, desc="Generating malaria dataset", unit="chunk")

    for chunk_idx in range(total_chunks):
        rng, rng_shape = random.split(rng)
        Nh, Q = simulator.sample_shape(rng_shape)
        group = _shape_group(Nh, Q)

        rng, rng_chunk = random.split(rng)
        allele_freq, state = simulator.simulate(
            rng_chunk,
            num_generations=cfg.num_generations,
            batch_size=cfg.batch_size,
            Nh=Nh,
            Q=Q,
            include_initial=cfg.include_initial,
        )
        allele_freq = np.asarray(jax.device_get(allele_freq), dtype=np.float32)
        params_jax, names = state.parameter_matrix(Nh, Q)
        if param_names is None:
            param_names = names
        elif param_names != names:
            raise ValueError(f"Parameter names changed from {param_names} to {names}")
        params = np.asarray(jax.device_get(params_jax), dtype=np.float32)
        sample_ids = np.arange(
            next_sample_id,
            next_sample_id + cfg.batch_size,
            dtype=np.int64,
        )
        next_sample_id += cfg.batch_size

        ds = _make_dataset(allele_freq, params, names, sample_ids, Nh, Q, cfg.seed)
        append = group_counts.get(group, 0) > 0
        _write_chunk(
            ds,
            store,
            group,
            append=append,
            sample_chunk=cfg.output.sample_chunk,
        )
        group_counts[group] = group_counts.get(group, 0) + cfg.batch_size

        group_records.append(
            {
                "group": group,
                "Nh": Nh,
                "Q": Q,
                "chunk_idx": int(chunk_idx),
                "sample_start": int(sample_ids[0]),
                "sample_stop": int(sample_ids[-1]) + 1,
            }
        )
        progress.update(1)
        progress.set_postfix_str(
            f"{group} samples={group_counts[group]}",
            refresh=False,
        )

    if param_names is None:
        raise ValueError("No samples were generated. Check num_chunks.")
    progress.close()
    _write_root_attrs(store, cfg, group_records, param_names)
    print(f"Done. Wrote {next_sample_id} samples to {store}")


def _build_simulator(cfg: DictConfig) -> MalariaTransmission:
    priors = cfg.priors
    return MalariaTransmission(
        Nh=_prior_from_config(priors.Nh),
        Q=_prior_from_config(priors.Q),
        chi=_prior_from_config(priors.chi),
        mu=_prior_from_config(priors.mu),
        sw=_prior_from_config(priors.sw),
        st=_prior_from_config(priors.st),
        p_init=_prior_from_config(priors.p_init),
    )


def _prior_from_config(cfg: DictConfig) -> Prior:
    """Build a `Prior` from a Hydra config node."""
    kwargs = OmegaConf.to_container(cfg.get("kwargs", {}), resolve=True)
    return Prior(cfg.dist, dict(kwargs))


def _shape_group(Nh: int, Q: int) -> str:
    return f"nh={int(Nh)}/q={int(Q)}"


def _make_dataset(
    allele_freq: np.ndarray,
    parameters: np.ndarray,
    parameter_names: tuple[str, ...],
    sample_ids: np.ndarray,
    Nh: int,
    Q: int,
    seed: int,
) -> xr.Dataset:
    generations = np.arange(allele_freq.shape[1], dtype=np.int32)
    hosts = np.arange(Nh, dtype=np.int32)
    return xr.Dataset(
        data_vars={
            "allele_freq": (
                ("simulation", "generation", "host"),
                allele_freq.astype(np.float32, copy=False),
            ),
            "parameters": (
                ("simulation", "parameter"),
                parameters.astype(np.float32, copy=False),
            ),
        },
        coords={
            "simulation": sample_ids.astype(np.int64, copy=False),
            "generation": generations,
            "host": hosts,
            "parameter": np.asarray(parameter_names, dtype=object),
        },
        attrs={
            "simulator": "malaria",
            "Nh": int(Nh),
            "Q": int(Q),
            "seed": int(seed),
            "parameter_names": list(parameter_names),
        },
    )


def _write_chunk(
    ds: xr.Dataset,
    store: Path,
    group: str,
    append: bool,
    sample_chunk: int,
) -> None:
    sample_chunk = min(sample_chunk, ds.sizes["simulation"])
    encoding = {
        "allele_freq": {
            "chunks": (sample_chunk, ds.sizes["generation"], ds.sizes["host"])
        },
        "parameters": {"chunks": (sample_chunk, ds.sizes["parameter"])},
    }
    kwargs = {"store": store, "group": group, "mode": "a", "encoding": encoding}
    if append:
        kwargs["append_dim"] = "simulation"
    ds.to_zarr(**kwargs)


def _write_root_attrs(
    store: Path,
    cfg: DictConfig,
    groups: list[dict],
    param_names: tuple[str, ...],
) -> None:
    root = zarr.open_group(str(store), mode="a")
    root.attrs.update(
        {
            "simulator": "malaria",
            "config": json.dumps(OmegaConf.to_container(cfg, resolve=True)),
            "groups": json.dumps(groups),
            "parameter_names": json.dumps(list(param_names)),
        }
    )


if __name__ == "__main__":
    main()
