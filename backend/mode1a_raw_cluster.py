import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from backend.config_loader import load_config
from backend.io import ensure_output_dir, save_raw_dbscan_to_hdf5
from backend.common_clustering import load_data_with_radius_filter, combine_haloes

# Temporary dataclass for load_data_with_radius_filter compatibility
@dataclass
class Mode1Wrapper:
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float

def run_mode1a(config_path="config.toml", output_dir="output", eps=None, min_samples=None):
    """
    Mode 1a: Raw DBSCAN clustering (no corrections)

    Performs:
    1. Load data from MCMC samples
    2. Combine halos into single dataset
    3. Run DBSCAN clustering

    Does NOT perform:
    - MCMC uniqueness constraint
    - Mass outlier filtering
    - Re-evaluation with mass distance
    """
    config = load_config(config_path)
    ensure_output_dir(output_dir)

    # Override mode1a eps/min_samples if provided via command line
    if eps is not None:
        config.mode1a.eps = eps
    if min_samples is not None:
        config.mode1a.min_samples = min_samples

    # Determine which parameters to use
    clustering_eps = config.mode1a.eps
    clustering_min_samples = config.mode1a.min_samples

    print("=" * 60)
    print("Mode 1a: Raw DBSCAN Clustering")
    print("=" * 60)

    # Create a wrapper config for compatibility with load_data_with_radius_filter
    config_wrapper = deepcopy(config)
    config_wrapper.mode1 = Mode1Wrapper(
        mcmc_start=config.mode1a.mcmc_start,
        mcmc_end=config.mode1a.mcmc_end,
        m200_mass_cut=config.mode1a.m200_mass_cut,
        radius_cut=config.mode1a.radius_cut
    )

    # Step 1: Load data
    print("\nStep 1: Loading MCMC data...")
    mcmc_data = load_data_with_radius_filter(config_wrapper)

    print(f"\nLoaded data from MCMC samples {config.mode1a.mcmc_start} to {config.mode1a.mcmc_end}")
    for mcmc_id, data in mcmc_data.items():
        print(f"  MCMC {mcmc_id}: {len(data['SO/200_crit/TotalMass'])} haloes")

    # Step 2: Combine halos
    print("\nStep 2: Combining halos from all MCMC samples...")
    combined_data, halo_provenance = combine_haloes(mcmc_data)

    positions = combined_data['SO/200_crit/CentreOfMass']
    m200_masses = combined_data['SO/200_crit/TotalMass']

    print(f"  Total combined halos: {len(positions)}")

    # Step 3: Run DBSCAN
    print(f"\nStep 3: Running DBSCAN (eps={clustering_eps} Mpc, min_samples={clustering_min_samples})...")

    # Optional: Weight positions by mass for clustering
    if config.mode1a.mass_weighted_clustering:
        print(f"  Using mass-weighted clustering (power={config.mode1a.mass_weight_power})")
        log_m200_masses = np.log10(m200_masses)
        log_m200_mass_weights = (log_m200_masses - np.min(log_m200_masses)) + 1  # Ensure positive weights
        m200_mass_weights = log_m200_mass_weights ** config.mode1a.mass_weight_power
        weighted_positions = positions * m200_mass_weights.reshape(-1, 1)
        clustering_input = weighted_positions
    else:
        clustering_input = positions

    clustering = DBSCAN(eps=clustering_eps, min_samples=clustering_min_samples)
    cluster_labels = clustering.fit_predict(clustering_input)

    # Statistics
    n_total = len(cluster_labels)
    n_noise = np.sum(cluster_labels == -1)
    n_clustered = n_total - n_noise
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))

    print(f"\n  Total halos: {n_total}")
    print(f"  Clustered halos: {n_clustered} ({100*n_clustered/n_total:.1f}%)")
    print(f"  Noise halos: {n_noise} ({100*n_noise/n_total:.1f}%)")
    print(f"  Number of clusters: {n_clusters}")

    # Show cluster size distribution
    if n_clusters > 0:
        cluster_sizes = []
        for cluster_id in np.unique(cluster_labels):
            if cluster_id != -1:
                cluster_sizes.append(np.sum(cluster_labels == cluster_id))
        cluster_sizes = sorted(cluster_sizes, reverse=True)

        print(f"\n  Largest 10 clusters: {cluster_sizes[:10]}")
        print(f"  Smallest cluster: {cluster_sizes[-1]}")
        print(f"  Mean cluster size: {np.mean(cluster_sizes):.1f}")
        print(f"  Median cluster size: {np.median(cluster_sizes):.1f}")

    # Save raw DBSCAN output
    print("\nSaving raw DBSCAN results to HDF5...")
    fname = f"raw_clusters_eps_{str(clustering_eps).replace('.','p')}_min_samples_{clustering_min_samples}.h5"
    save_raw_dbscan_to_hdf5(
        cluster_labels, positions, m200_masses, halo_provenance,
        combined_data, config, output_dir, filename=fname
    )

    print("\n" + "=" * 60)
    print("Mode 1a complete!")
    print(f"Output file: {output_dir}/{fname}")
    print("=" * 60)

if __name__ == '__main__':
    run_mode1a()
