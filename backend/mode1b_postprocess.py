import numpy as np
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import List
from backend.config_loader import load_config
from backend.io import ensure_output_dir, load_clusters_from_hdf5, save_clusters_to_hdf5
from backend.common_clustering import (
    enforce_mcmc_constraint_with_mass_filter,
    _compute_shape_measures
)

# Temporary dataclass for save_clusters_to_hdf5 compatibility
@dataclass
class Mode1ConfigWrapper:
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

def run_mode1b(config_path="config.toml", output_dir="output", input_filename=None,
               mass_outlier_threshold=None, use_mass_distance=None):
    """
    Mode 1b: Post-processing corrections on raw DBSCAN output

    Performs:
    4. MCMC uniqueness constraint (position-based)
    5. Mass outlier filtering
    6. Optional re-evaluation with mass distance

    Requires Mode 1a output as input.
    """
    config = load_config(config_path)
    ensure_output_dir(output_dir)

    # Determine input filename
    if input_filename is None:
        input_filename = config.mode1b.input_filename
    if not input_filename:
        # Try to auto-detect based on mode1a config
        eps_str = str(config.mode1a.eps).replace('.', 'p')
        min_samples = config.mode1a.min_samples
        input_filename = f"raw_clusters_eps_{eps_str}_min_samples_{min_samples}.h5"

    # Determine parameters (command-line overrides config)
    mass_outlier_thresh = mass_outlier_threshold if mass_outlier_threshold is not None else config.mode1b.mass_outlier_threshold
    use_mass_dist = use_mass_distance if use_mass_distance is not None else config.mode1b.use_mass_distance

    print("=" * 60)
    print("Mode 1b: Post-processing Corrections")
    print("=" * 60)

    # Load raw clusters from mode1a (standard HDF5 format)
    print(f"\nLoading raw clusters from: {input_filename}")
    raw_clusters, metadata = load_clusters_from_hdf5(output_dir, filename=input_filename, minimal=False, min_m200_mass=0)

    # Reconstruct arrays from clusters for applying corrections
    # We need to rebuild: positions, m200_masses, halo_provenance, cluster_labels, combined_data
    positions_list = []
    m200_masses_list = []
    halo_provenance = []
    cluster_labels_list = []
    combined_data_keys = None

    # Collect all halos from all clusters
    for cluster in raw_clusters:
        cluster_id = cluster['cluster_id']
        n_members = cluster['cluster_size']
        member_data = cluster['member_data']

        # Add to arrays
        positions_list.append(member_data['SO/200_crit/CentreOfMass'])
        m200_masses_list.append(member_data['SO/200_crit/TotalMass'])
        cluster_labels_list.extend([cluster_id] * n_members)

        # Build provenance
        mcmc_ids_in_cluster = member_data['mcmc/id']
        orig_indices_in_cluster = member_data['halo/original/index']
        for i in range(n_members):
            halo_provenance.append({
                'mcmc_id': int(mcmc_ids_in_cluster[i]),
                'original_index': int(orig_indices_in_cluster[i])
            })

        # Initialize combined_data on first cluster
        if combined_data_keys is None:
            combined_data_keys = list(member_data.keys())

    # Concatenate all arrays
    positions = np.vstack(positions_list)
    m200_masses = np.concatenate(m200_masses_list)
    cluster_labels = np.array(cluster_labels_list)

    # Rebuild combined_data
    combined_data = {}
    for key in combined_data_keys:
        data_list = []
        for cluster in raw_clusters:
            data_list.append(cluster['member_data'][key])
        if len(data_list[0].shape) == 1:
            combined_data[key] = np.concatenate(data_list)
        else:
            combined_data[key] = np.vstack(data_list)

    # Extract MCMC IDs
    mcmc_ids = np.array([p['mcmc_id'] for p in halo_provenance])

    print(f"  Loaded {len(raw_clusters)} raw clusters with {len(positions)} total halos")

    # Statistics before corrections
    n_total = len(cluster_labels)
    n_noise_before = np.sum(cluster_labels == -1)
    n_clustered_before = n_total - n_noise_before
    n_clusters_before = len(np.unique(cluster_labels[cluster_labels != -1]))

    print(f"\nBefore corrections:")
    print(f"  Total halos: {n_total}")
    print(f"  Clustered halos: {n_clustered_before} ({100*n_clustered_before/n_total:.1f}%)")
    print(f"  Noise halos: {n_noise_before} ({100*n_noise_before/n_total:.1f}%)")
    print(f"  Number of clusters: {n_clusters_before}")

    # Steps 4-6: Apply post-processing corrections
    print(f"\nApplying post-processing corrections...")
    print(f"  Mass outlier threshold: {mass_outlier_thresh} dex")
    print(f"  Use mass distance: {use_mass_dist}")

    cluster_labels = enforce_mcmc_constraint_with_mass_filter(
        cluster_labels, positions, m200_masses, mcmc_ids,
        mass_outlier_threshold=mass_outlier_thresh,
        use_mass_distance=use_mass_dist
    )

    # Statistics after corrections
    n_noise_after = np.sum(cluster_labels == -1)
    n_clustered_after = n_total - n_noise_after
    n_clusters_after = len(np.unique(cluster_labels[cluster_labels != -1]))

    print(f"\nAfter corrections:")
    print(f"  Total halos: {n_total}")
    print(f"  Clustered halos: {n_clustered_after} ({100*n_clustered_after/n_total:.1f}%)")
    print(f"  Noise halos: {n_noise_after} ({100*n_noise_after/n_total:.1f}%)")
    print(f"  Number of clusters: {n_clusters_after}")
    print(f"\n  Halos removed: {n_clustered_before - n_clustered_after}")
    print(f"  Clusters removed: {n_clusters_before - n_clusters_after}")

    # Build stable_haloes list from corrected cluster_labels
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

    print("\nTop 10 stable haloes by cluster size:")
    for i, halo in enumerate(sorted_haloes[:10]):
        print(f"\n  Halo {i}:")
        print(f"    Cluster size: {halo['cluster_size']}")
        print(f"    M200 mass mean: {halo['mean_m200_mass']:.2e} ± {halo['m200_mass_std']:.2e}")
        if not np.isnan(halo['mean_subhalo_mass']):
            print(f"    Subhalo mass mean: {halo['mean_subhalo_mass']:.2e} ± {halo['subhalo_mass_std']:.2e}")
        if not np.isnan(halo['mean_m500']):
            print(f"    M500 mass mean: {halo['mean_m500']:.2e} ± {halo['m500_std']:.2e}")
        print(f"    Log10 M200 std: {halo['log10_m200_mass_std']:.3f}")
        print(f"    Position mean: [{halo['mean_position'][0]:.1f}, {halo['mean_position'][1]:.1f}, {halo['mean_position'][2]:.1f}]")
        print(f"    Position std: [{halo['position_std'][0]:.1f}, {halo['position_std'][1]:.1f}, {halo['position_std'][2]:.1f}]")

    # Save final corrected clusters
    print("\nSaving corrected clusters to HDF5...")
    eps_str = str(metadata['eps']).replace('.', 'p')
    min_samples = metadata['min_samples']
    fname = f"clusters_eps_{eps_str}_min_samples_{min_samples}.h5"

    # Create a wrapper config for save_clusters_to_hdf5
    # Combine metadata from mode1a output with mode1b settings
    config_wrapper = deepcopy(config)
    config_wrapper.mode1 = Mode1ConfigWrapper(
        mcmc_start=int(metadata['mcmc_start']),
        mcmc_end=int(metadata['mcmc_end']),
        m200_mass_cut=float(metadata['m200_mass_cut']),
        radius_cut=float(metadata['radius_cut']),
        eps=float(metadata['eps']),
        min_samples=int(metadata['min_samples']),
        min_association_size=config.mode1b.min_association_size,
        mass_outlier_threshold=mass_outlier_thresh,
        use_mass_distance=use_mass_dist,
        mass_weighted_clustering=bool(metadata.get('mass_weighted_clustering', False)),
        mass_weight_power=float(metadata.get('mass_weight_power', 0.5))
    )

    save_clusters_to_hdf5(
        stable_haloes, positions, m200_masses, halo_provenance, cluster_labels,
        config_wrapper, output_dir, filename=fname
    )

    print("\n" + "=" * 60)
    print("Mode 1b complete!")
    print(f"Output file: {output_dir}/{fname}")
    print("=" * 60)

if __name__ == '__main__':
    run_mode1b()
