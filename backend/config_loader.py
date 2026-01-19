import toml
import os
from dataclasses import dataclass
from typing import List

@dataclass
class GlobalConfig:
    basedir: str
    observer_coords: List[float]
    output_dir: str
    boxsize: float
    final_snapshot: int
    hdf5_subdir: str
    hdf5_filename_pattern: str

@dataclass
class Mode1Config:
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float
    # HDBSCAN parameters
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str
    cluster_selection_epsilon: float  # 0.0 = off, >0 = distance threshold for merging
    alpha: float  # HDBSCAN alpha parameter for mutual reachability distance
    # Stability thresholds
    existence_prob_stable: float
    existence_prob_tentative: float
    # Save all input halos to HDF5
    save_input_catalog: bool

@dataclass
class Mode2Config:
    target_snapshot: int
    min_cluster_size: int
    mass_tolerance_dex: float
    min_match_rate: float
    distance_tolerance_rel: float

@dataclass
class Mode3Config:
    basedir: str
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float
    num_samplings: int
    # HDBSCAN parameters (same as mode1)
    min_cluster_size: int
    min_samples: int
    cluster_selection_epsilon: float  # 0.0 = off, >0 = distance threshold for merging
    alpha: float  # HDBSCAN alpha parameter for mutual reachability distance
    # Random seed for reproducible observer positions across parameter sweeps
    random_seed: int
    # Save all input halos to HDF5
    save_input_catalog: bool

@dataclass
class Mode4Config:
    basedir: str
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float
    target_snapshot: int
    observer_coords: List[float]

@dataclass
class Config:
    global_config: GlobalConfig
    mode1: Mode1Config
    mode2: Mode2Config
    mode3: Mode3Config
    mode4: Mode4Config

def load_config(config_path: str = "config.toml") -> Config:
    with open(config_path, 'r') as f:
        data = toml.load(f)

    global_config = GlobalConfig(
        basedir=str(data['global']['basedir']),
        observer_coords=[float(x) for x in data['global']['observer_coords']],
        output_dir=str(data['global']['output_dir']),
        boxsize=float(data['global']['boxsize']),
        final_snapshot=int(data['global'].get('final_snapshot', 77)),
        hdf5_subdir=str(data['global'].get('hdf5_subdir', 'soap/SOAP_uncompressed/HBTplus')),
        hdf5_filename_pattern=str(data['global'].get('hdf5_filename_pattern', 'halo_properties_{snap_num:04d}.hdf5'))
    )

    mode2_config = Mode2Config(
        target_snapshot=int(data['mode2']['target_snapshot']),
        min_cluster_size=int(data['mode2']['min_cluster_size']),
        mass_tolerance_dex=float(data['mode2'].get('mass_tolerance_dex', 0.1)),
        min_match_rate=float(data['mode2'].get('min_match_rate', 0.8)),
        distance_tolerance_rel=float(data['mode2'].get('distance_tolerance_rel', 0.2))
    )

    # Compute default min_cluster_size for mode3 based on number of virtual realizations
    n_mode3_sims = int(data['mode3']['mcmc_end']) - int(data['mode3']['mcmc_start']) + 1
    n_mode3_samplings = int(data['mode3']['num_samplings'])
    n_mode3_realizations = n_mode3_sims * n_mode3_samplings
    default_mode3_min_cluster_size = max(10, round(0.15 * n_mode3_realizations))

    mode3_config = Mode3Config(
        basedir=str(data['mode3']['basedir']),
        mcmc_start=int(data['mode3']['mcmc_start']),
        mcmc_end=int(data['mode3']['mcmc_end']),
        m200_mass_cut=float(data['mode3']['m200_mass_cut']),
        radius_cut=float(data['mode3']['radius_cut']),
        num_samplings=int(data['mode3']['num_samplings']),
        min_cluster_size=int(data['mode3'].get('min_cluster_size', default_mode3_min_cluster_size)),
        min_samples=int(data['mode3'].get('min_samples', default_mode3_min_cluster_size)),
        cluster_selection_epsilon=float(data['mode3'].get('cluster_selection_epsilon', 0.0)),
        alpha=float(data['mode3'].get('alpha', 1.0)),
        random_seed=int(data['mode3'].get('random_seed', 42)),
        save_input_catalog=bool(data['mode3'].get('save_input_catalog', False))
    )

    mode4_config = Mode4Config(
        basedir=str(data['mode4']['basedir']),
        mcmc_start=int(data['mode4']['mcmc_start']),
        mcmc_end=int(data['mode4']['mcmc_end']),
        m200_mass_cut=float(data['mode4']['m200_mass_cut']),
        radius_cut=float(data['mode4']['radius_cut']),
        target_snapshot=int(data['mode4']['target_snapshot']),
        observer_coords=[float(x) for x in data['mode4']['observer_coords']]
    )

    # Compute default min_cluster_size based on number of realizations
    n_realizations = int(data['mode1']['mcmc_end']) - int(data['mode1']['mcmc_start']) + 1
    default_min_cluster_size = max(10, round(0.15 * n_realizations))

    mode1_config = Mode1Config(
        mcmc_start=int(data['mode1']['mcmc_start']),
        mcmc_end=int(data['mode1']['mcmc_end']),
        m200_mass_cut=float(data['mode1']['m200_mass_cut']),
        radius_cut=float(data['mode1']['radius_cut']),
        # HDBSCAN parameters
        min_cluster_size=int(data['mode1'].get('min_cluster_size', default_min_cluster_size)),
        min_samples=int(data['mode1'].get('min_samples', default_min_cluster_size)),
        cluster_selection_method=str(data['mode1'].get('cluster_selection_method', 'eom')),
        cluster_selection_epsilon=float(data['mode1'].get('cluster_selection_epsilon', 0.0)),
        alpha=float(data['mode1'].get('alpha', 1.0)),
        # Stability thresholds
        existence_prob_stable=float(data['mode1'].get('existence_prob_stable', 0.5)),
        existence_prob_tentative=float(data['mode1'].get('existence_prob_tentative', 0.2)),
        # Save input catalog
        save_input_catalog=bool(data['mode1'].get('save_input_catalog', False))
    )

    return Config(
        global_config=global_config,
        mode1=mode1_config,
        mode2=mode2_config,
        mode3=mode3_config,
        mode4=mode4_config
    )
