import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from backend.config_loader import load_config
from backend.io import ensure_output_dir, save_clusters_to_hdf5
from backend.common_clustering import load_data_with_radius_filter, combine_haloes, _compute_shape_measures

# Temporary dataclass for compatibility with functions expecting mode1 config
@dataclass
class Mode1Wrapper:
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float
    eps: float
    min_samples: int
    min_association_size: int
    mass_outlier_threshold: float
    use_mass_distance: bool
    mass_weighted_clustering: bool
    mass_weight_power: float

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

    # Create a wrapper config for compatibility with functions expecting mode1 config
    config_wrapper = deepcopy(config)
    config_wrapper.mode1 = Mode1Wrapper(
        mcmc_start=config.mode1a.mcmc_start,
        mcmc_end=config.mode1a.mcmc_end,
        m200_mass_cut=config.mode1a.m200_mass_cut,
        radius_cut=config.mode1a.radius_cut,
        eps=clustering_eps,
        min_samples=clustering_min_samples,
        min_association_size=config.mode1b.min_association_size,  # Use mode1b value for output filtering
        mass_outlier_threshold=0.0,  # Not used in mode1a
        use_mass_distance=False,  # Not used in mode1a
        mass_weighted_clustering=config.mode1a.mass_weighted_clustering,
        mass_weight_power=config.mode1a.mass_weight_power
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

    # Build stable_haloes structure from raw DBSCAN results
    print("\nComputing cluster statistics...")
    stable_haloes = []

    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue

        cluster_mask = cluster_labels == cluster_id
        cluster_positions = positions[cluster_mask]
        cluster_m200_masses = m200_masses[cluster_mask]
        cluster_provenance = [halo_provenance[i] for i in range(len(halo_provenance)) if cluster_mask[i]]

        cluster_size = len(cluster_positions)
        mean_position = np.mean(cluster_positions, axis=0)
        mean_m200_mass = np.mean(cluster_m200_masses)
        position_std = np.std(cluster_positions, axis=0)
        m200_mass_std = np.std(cluster_m200_masses)
        log10_m200_mass_std = np.std(np.log10(cluster_m200_masses))

        # Calculate mean subhalo mass and M500
        cluster_subhalo_masses = combined_data['BoundSubhalo/TotalMass'][cluster_mask]
        cluster_m500 = combined_data['SO/500_crit/TotalMass'][cluster_mask]

        # Handle NaN values
        valid_subhalo_masses = cluster_subhalo_masses[~np.isnan(cluster_subhalo_masses)]
        valid_m500 = cluster_m500[~np.isnan(cluster_m500)]

        mean_subhalo_mass = np.mean(valid_subhalo_masses) if len(valid_subhalo_masses) > 0 else np.nan
        mean_m500 = np.mean(valid_m500) if len(valid_m500) > 0 else np.nan
        subhalo_mass_std = np.std(valid_subhalo_masses) if len(valid_subhalo_masses) > 0 else np.nan
        m500_std = np.std(valid_m500) if len(valid_m500) > 0 else np.nan
        log10_m500_std = np.std(np.log10(valid_m500)) if len(valid_m500) > 0 else np.nan

        # Compute shape measures
        shape_measures = _compute_shape_measures(cluster_positions)

        # Extract all member data for this cluster
        member_data = {}
        for key, data in combined_data.items():
            member_data[key] = data[cluster_mask]

        stable_haloes.append({
            'cluster_id': cluster_id,
            'cluster_size': cluster_size,
            'mean_position': mean_position,
            'mean_m200_mass': mean_m200_mass,
            'mean_subhalo_mass': mean_subhalo_mass,
            'mean_m500': mean_m500,
            'position_std': position_std,
            'm200_mass_std': m200_mass_std,
            'subhalo_mass_std': subhalo_mass_std,
            'm500_std': m500_std,
            'members': cluster_provenance,
            'member_data': member_data,
            'log10_m200_mass_std': log10_m200_mass_std,
            'log10_m500_std': log10_m500_std,
            'axis_ratio_ba': shape_measures['axis_ratio_ba'],
            'axis_ratio_ca': shape_measures['axis_ratio_ca'],
            'asphericity': shape_measures['asphericity'],
            'prolateness': shape_measures['prolateness'],
        })

    print(f"  Computed statistics for {len(stable_haloes)} clusters")

    # Sort and display top clusters
    sorted_haloes = sorted(stable_haloes, key=lambda x: x['cluster_size'], reverse=True)

    print("\nTop 10 raw clusters by size (no corrections applied):")
    for i, halo in enumerate(sorted_haloes[:10]):
        print(f"\n  Cluster {i}:")
        print(f"    Cluster size: {halo['cluster_size']}")
        print(f"    M200 mass mean: {halo['mean_m200_mass']:.2e} ± {halo['m200_mass_std']:.2e}")
        if not np.isnan(halo['mean_subhalo_mass']):
            print(f"    Subhalo mass mean: {halo['mean_subhalo_mass']:.2e} ± {halo['subhalo_mass_std']:.2e}")
        if not np.isnan(halo['mean_m500']):
            print(f"    M500 mass mean: {halo['mean_m500']:.2e} ± {halo['m500_std']:.2e}")
        print(f"    Log10 M200 std: {halo['log10_m200_mass_std']:.3f}")
        print(f"    Position mean: [{halo['mean_position'][0]:.1f}, {halo['mean_position'][1]:.1f}, {halo['mean_position'][2]:.1f}]")
        print(f"    Position std: [{halo['position_std'][0]:.1f}, {halo['position_std'][1]:.1f}, {halo['position_std'][2]:.1f}]")

    # Save raw DBSCAN clusters (same format as mode1b output, but without corrections)
    print("\nSaving raw DBSCAN clusters to HDF5...")
    fname = f"raw_clusters_eps_{str(clustering_eps).replace('.','p')}_min_samples_{clustering_min_samples}.h5"
    save_clusters_to_hdf5(
        stable_haloes, positions, m200_masses, halo_provenance, cluster_labels,
        config_wrapper, output_dir, filename=fname
    )

    print("\n" + "=" * 60)
    print("Mode 1a complete!")
    print(f"Output file: {output_dir}/{fname}")
    print("=" * 60)

if __name__ == '__main__':
    run_mode1a()
