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
class Mode1aConfig:
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float
    # HDBSCAN parameters
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str
    # Whitening parameters
    sigma_logM: float
    knn_k: int
    n_mass_bins: int
    sigma_x_min_percentile: float
    sigma_x_max_percentile: float
    # Stability thresholds
    existence_prob_stable: float
    existence_prob_tentative: float

@dataclass
class Mode1bConfig:
    input_filename: str
    min_association_size: int
    mass_outlier_threshold: float
    use_mass_distance: bool

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
    eps: float
    min_samples: int

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
    mode1a: Mode1aConfig
    mode1b: Mode1bConfig
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

    mode3_config = Mode3Config(
        basedir=str(data['mode3']['basedir']),
        mcmc_start=int(data['mode3']['mcmc_start']),
        mcmc_end=int(data['mode3']['mcmc_end']),
        m200_mass_cut=float(data['mode3']['m200_mass_cut']),
        radius_cut=float(data['mode3']['radius_cut']),
        num_samplings=int(data['mode3']['num_samplings']),
        eps=float(data['mode3']['eps']),
        min_samples=int(data['mode3']['min_samples'])
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
    n_realizations = int(data['mode1a']['mcmc_end']) - int(data['mode1a']['mcmc_start']) + 1
    default_min_cluster_size = max(10, round(0.15 * n_realizations))

    mode1a_config = Mode1aConfig(
        mcmc_start=int(data['mode1a']['mcmc_start']),
        mcmc_end=int(data['mode1a']['mcmc_end']),
        m200_mass_cut=float(data['mode1a']['m200_mass_cut']),
        radius_cut=float(data['mode1a']['radius_cut']),
        # HDBSCAN parameters
        min_cluster_size=int(data['mode1a'].get('min_cluster_size', default_min_cluster_size)),
        min_samples=int(data['mode1a'].get('min_samples', default_min_cluster_size)),
        cluster_selection_method=str(data['mode1a'].get('cluster_selection_method', 'eom')),
        # Whitening parameters
        sigma_logM=float(data['mode1a'].get('sigma_logM', 0.0)),
        knn_k=int(data['mode1a'].get('knn_k', 8)),
        n_mass_bins=int(data['mode1a'].get('n_mass_bins', 10)),
        sigma_x_min_percentile=float(data['mode1a'].get('sigma_x_min_percentile', 5.0)),
        sigma_x_max_percentile=float(data['mode1a'].get('sigma_x_max_percentile', 95.0)),
        # Stability thresholds
        existence_prob_stable=float(data['mode1a'].get('existence_prob_stable', 0.5)),
        existence_prob_tentative=float(data['mode1a'].get('existence_prob_tentative', 0.2))
    )

    mode1b_config = Mode1bConfig(
        input_filename=str(data['mode1b'].get('input_filename', '')),
        min_association_size=int(data['mode1b']['min_association_size']),
        mass_outlier_threshold=float(data['mode1b']['mass_outlier_threshold']),
        use_mass_distance=bool(data['mode1b']['use_mass_distance'])
    )

    return Config(
        global_config=global_config,
        mode1a=mode1a_config,
        mode1b=mode1b_config,
        mode2=mode2_config,
        mode3=mode3_config,
        mode4=mode4_config
    )
