"""
Mode 3: Random Control Simulation Analysis

Performs the same pure 3D HDBSCAN clustering as Mode 1, but on control simulations
with random observer positions to establish a null hypothesis baseline.

Key features:
- Multiple random observer position samplings per control simulation
- Pure 3D spatial clustering (positions only, no whitening)
- HDBSCAN for density-based clustering with soft cluster membership
- One-per-realization constraint enforcement
- Same output format as Mode 1 for direct comparison
"""

import numpy as np
import os
from typing import Tuple, Dict, List

try:
    import hdbscan
except ImportError:
    raise ImportError("hdbscan package required. Install with: pip install hdbscan")

from pymanticore.swift_analysis import SOAPData
from backend.config_loader import load_config
from backend.io import ensure_output_dir, save_hdbscan_clusters_to_hdf5
from backend.common_clustering import combine_haloes, _compute_shape_measures


def load_random_control_data(config):
    """Load random control simulations with multiple random observer samplings.

    Each control simulation is sampled multiple times with different random
    observer positions, creating "virtual" MCMC samples for clustering.
    """

    # Calculate total virtual mcmc samples
    n_real_sims = config.mode3.mcmc_end - config.mode3.mcmc_start + 1
    total_virtual_mcmc = n_real_sims * config.mode3.num_samplings

    mcmc_data = {}

    # Common reference point (box center)
    box_center = np.array([config.global_config.boxsize/2, config.global_config.boxsize/2, config.global_config.boxsize/2])

    # Extended property list
    to_load = [
        "BoundSubhalo/TotalMass",
        "BoundSubhalo/CentreOfMass",
        "BoundSubhalo/CentreOfMassVelocity",
        "SOAP/ProgenitorIndex",
        "BoundSubhalo/MaximumCircularVelocity",
        "BoundSubhalo/EncloseRadius",
        "SO/200_crit/TotalMass",
        "SO/200_crit/CentreOfMass",
        "SO/200_crit/CentreOfMassVelocity",
        "SO/200_crit/Concentration",
        "SO/200_crit/SORadius",
        "SO/200_crit/MassFractionExternal",
        "SO/200_crit/MassFractionSatellites",
        "SO/500_crit/TotalMass",
        "SO/500_crit/CentreOfMass",
        "SO/500_crit/CentreOfMassVelocity",
        "SO/500_crit/Concentration",
        "SO/500_crit/SORadius",
        "SO/500_crit/MassFractionExternal",
        "SO/500_crit/MassFractionSatellites",
        "SOAP/SubhaloRankByBoundMass"
    ]

    for virtual_mcmc_id in range(total_virtual_mcmc):
        # Map virtual mcmc_id to (simulation_id, sampling_id)
        sim_id = config.mode3.mcmc_start + (virtual_mcmc_id // config.mode3.num_samplings)
        sampling_id = virtual_mcmc_id % config.mode3.num_samplings

        # Generate random observer coordinates
        random_observer = np.random.uniform(0, config.global_config.boxsize, 3)

        # Load the simulation
        final_snap = config.global_config.final_snapshot
        filename_pattern = config.global_config.hdf5_filename_pattern.format(snap_num=final_snap)
        filename = os.path.join(config.mode3.basedir, f"mcmc_{sim_id}", config.global_config.hdf5_subdir, filename_pattern)
        soap_data = SOAPData(filename)
        soap_data.load_groups(properties=to_load, only_centrals=True)
        soap_data.set_observer(random_observer, skip_redshift=True, radius_cut=config.mode3.radius_cut)

        # Don't want to keep redshift
        del soap_data.data["redshift"]

        # Apply M200 mass cut
        m200_masses = soap_data.data['SO/200_crit/TotalMass']
        m200_mass_mask = m200_masses >= config.mode3.m200_mass_cut

        filtered_data = {}
        for key, value in soap_data.data.items():
            if isinstance(value, np.ndarray) and len(value) == len(m200_masses):
                filtered_data[key] = value[m200_mass_mask]
            else:
                filtered_data[key] = value

        # Transform positions back to common reference frame (box center)
        observer_relative_positions = filtered_data['SO/200_crit/CentreOfMass']
        box_coordinates = observer_relative_positions - random_observer
        centered_positions = box_coordinates + box_center
        filtered_data['SO/200_crit/CentreOfMass'] = centered_positions

        # Transform other position-based properties if they exist
        for key in ['BoundSubhalo/CentreOfMass', 'SO/500_crit/CentreOfMass']:
            if key in filtered_data:
                observer_relative = filtered_data[key]
                box_coords = observer_relative + random_observer
                filtered_data[key] = box_coords - box_center

        mcmc_data[virtual_mcmc_id] = filtered_data

        print(f"Loaded virtual MCMC {virtual_mcmc_id} (sim {sim_id}, sample {sampling_id}): "
              f"{len(mcmc_data[virtual_mcmc_id]['SO/200_crit/TotalMass'])} haloes, "
              f"observer=[{random_observer[0]:.1f}, {random_observer[1]:.1f}, {random_observer[2]:.1f}]")

    return mcmc_data


def run_hdbscan(
    positions: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str = 'eom'
) -> Tuple[np.ndarray, np.ndarray, 'hdbscan.HDBSCAN']:
    """Run HDBSCAN clustering on 3D positions.

    Args:
        positions: (N, 3) array of x,y,z coordinates in Mpc
        min_cluster_size: minimum cluster size for HDBSCAN
        min_samples: minimum samples for core point
        cluster_selection_method: 'eom' (excess of mass) or 'leaf'

    Returns:
        labels: (N,) cluster labels (-1 for noise)
        probabilities: (N,) membership probabilities
        clusterer: fitted HDBSCAN object (for stability metrics)
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric='euclidean',
        prediction_data=True
    )

    labels = clusterer.fit_predict(positions)
    probabilities = clusterer.probabilities_

    return labels, probabilities, clusterer


def enforce_one_per_realization(
    labels: np.ndarray,
    probabilities: np.ndarray,
    positions: np.ndarray,
    realization_ids: np.ndarray,
    n_realizations: int
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """Enforce <=1 halo per realization per cluster.

    Selection criteria (in order):
    1. Highest membership probability
    2. Tie-break: closest to cluster center
    """
    labels = labels.copy()
    dropped_mask = np.zeros(len(labels), dtype=bool)
    ambiguity_rates = {}

    unique_clusters = np.unique(labels)

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue

        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_positions = positions[cluster_mask]
        cluster_probs = probabilities[cluster_mask]
        cluster_realization_ids = realization_ids[cluster_mask]

        cluster_center = np.median(cluster_positions, axis=0)

        n_conflicts = 0
        unique_realizations = np.unique(cluster_realization_ids)

        for real_id in unique_realizations:
            real_mask = cluster_realization_ids == real_id
            real_indices = cluster_indices[real_mask]

            if len(real_indices) <= 1:
                continue

            n_conflicts += 1

            real_probs = cluster_probs[real_mask]
            real_positions = cluster_positions[real_mask]

            max_prob = np.max(real_probs)
            max_prob_mask = real_probs == max_prob

            if np.sum(max_prob_mask) == 1:
                keep_idx = np.argmax(real_probs)
            else:
                tied_positions = real_positions[max_prob_mask]
                tied_indices = np.where(max_prob_mask)[0]
                distances = np.linalg.norm(tied_positions - cluster_center, axis=1)
                keep_idx = tied_indices[np.argmin(distances)]

            for i, global_idx in enumerate(real_indices):
                if i != keep_idx:
                    labels[global_idx] = -1
                    dropped_mask[global_idx] = True

        n_realizations_present = len(unique_realizations)
        ambiguity_rates[cluster_id] = n_conflicts / n_realizations_present if n_realizations_present > 0 else 0.0

    return labels, dropped_mask, ambiguity_rates


def get_hdbscan_stability(clusterer: 'hdbscan.HDBSCAN', cluster_id: int) -> float:
    """Extract HDBSCAN stability score for a cluster."""
    try:
        cluster_tree = clusterer.condensed_tree_.to_pandas()
        cluster_data = cluster_tree[cluster_tree['child'] == cluster_id]
        if len(cluster_data) > 0:
            return float(cluster_data['lambda_val'].max())
    except Exception:
        pass
    return np.nan


def summarize_clusters(
    labels: np.ndarray,
    probabilities: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    realization_ids: np.ndarray,
    combined_data: Dict,
    clusterer: 'hdbscan.HDBSCAN',
    ambiguity_rates: Dict[int, float],
    n_realizations: int,
    existence_prob_stable: float,
    existence_prob_tentative: float
) -> List[Dict]:
    """Compute comprehensive cluster summaries."""
    log_masses = np.log10(masses)
    cluster_summaries = []

    unique_clusters = np.unique(labels)
    unique_clusters = unique_clusters[unique_clusters != -1]

    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_positions = positions[cluster_mask]
        cluster_masses = masses[cluster_mask]
        cluster_log_masses = log_masses[cluster_mask]
        cluster_probs = probabilities[cluster_mask]
        cluster_realization_ids = realization_ids[cluster_mask]

        n_members = np.sum(cluster_mask)
        n_realizations_present = len(np.unique(cluster_realization_ids))
        existence_prob = n_realizations_present / n_realizations

        center_xyz = np.median(cluster_positions, axis=0)
        cov_xyz = np.cov(cluster_positions.T) if n_members > 1 else np.eye(3) * np.nan

        center_logM = np.median(cluster_log_masses)
        var_logM = np.var(cluster_log_masses) if n_members > 1 else np.nan
        mean_mass = np.mean(cluster_masses)
        mass_std = np.std(cluster_masses) if n_members > 1 else np.nan
        log_mass_std = np.std(cluster_log_masses) if n_members > 1 else np.nan

        mean_prob = np.mean(cluster_probs)
        min_prob = np.min(cluster_probs)

        hdbscan_stability = get_hdbscan_stability(clusterer, cluster_id)
        ambiguity_rate = ambiguity_rates.get(cluster_id, 0.0)

        if existence_prob >= existence_prob_stable:
            status = 'stable'
        elif existence_prob >= existence_prob_tentative:
            status = 'tentative'
        else:
            status = 'rare'

        shape_measures = _compute_shape_measures(cluster_positions)

        # M500 statistics
        cluster_m500 = combined_data['SO/500_crit/TotalMass'][cluster_mask]
        valid_m500 = cluster_m500[~np.isnan(cluster_m500)]
        mean_m500 = np.mean(valid_m500) if len(valid_m500) > 0 else np.nan
        m500_std = np.std(valid_m500) if len(valid_m500) > 0 else np.nan
        log10_m500_std = np.std(np.log10(valid_m500)) if len(valid_m500) > 0 else np.nan

        # Subhalo mass statistics
        cluster_subhalo_masses = combined_data['BoundSubhalo/TotalMass'][cluster_mask]
        valid_subhalo = cluster_subhalo_masses[~np.isnan(cluster_subhalo_masses)]
        mean_subhalo_mass = np.mean(valid_subhalo) if len(valid_subhalo) > 0 else np.nan
        subhalo_mass_std = np.std(valid_subhalo) if len(valid_subhalo) > 0 else np.nan

        member_data = {}
        for key, data in combined_data.items():
            member_data[key] = data[cluster_mask]

        cluster_summaries.append({
            'original_cluster_id': int(cluster_id),
            'cluster_id': None,
            'existence_prob': existence_prob,
            'n_realizations_present': n_realizations_present,
            'n_members': n_members,
            'center_xyz': center_xyz,
            'cov_xyz': cov_xyz,
            'center_logM': center_logM,
            'var_logM': var_logM,
            'mean_m200_mass': mean_mass,
            'm200_mass_std': mass_std,
            'log10_m200_mass_std': log_mass_std,
            'mean_m500': mean_m500,
            'm500_std': m500_std,
            'log10_m500_std': log10_m500_std,
            'mean_subhalo_mass': mean_subhalo_mass,
            'subhalo_mass_std': subhalo_mass_std,
            'hdbscan_stability': hdbscan_stability,
            'mean_membership_prob': mean_prob,
            'min_membership_prob': min_prob,
            'ambiguity_rate': ambiguity_rate,
            'status': status,
            'axis_ratio_ba': shape_measures['axis_ratio_ba'],
            'axis_ratio_ca': shape_measures['axis_ratio_ca'],
            'asphericity': shape_measures['asphericity'],
            'prolateness': shape_measures['prolateness'],
            'position_std': np.std(cluster_positions, axis=0),
            'member_data': member_data,
            'member_indices': np.where(cluster_mask)[0],
            'member_probs': cluster_probs.copy()
        })

    cluster_summaries.sort(
        key=lambda x: (-x['existence_prob'], -x['hdbscan_stability'] if not np.isnan(x['hdbscan_stability']) else 0)
    )

    for i, summary in enumerate(cluster_summaries):
        summary['cluster_id'] = i

    return cluster_summaries


def run_mode3(config_path="config.toml", output_dir="output",
              min_cluster_size=None, min_samples=None):
    """
    Mode 3: Random Control Simulation Analysis with HDBSCAN

    Performs:
    1. Load control simulations with random observer positions
    2. Run HDBSCAN clustering on 3D positions
    3. Enforce one-per-realization constraint
    4. Compute cluster summaries
    5. Save results to HDF5

    Args:
        config_path: Path to config.toml
        output_dir: Output directory for results
        min_cluster_size: Override for HDBSCAN min_cluster_size
        min_samples: Override for HDBSCAN min_samples
    """
    config = load_config(config_path)
    ensure_output_dir(output_dir)

    # Override parameters if provided
    if min_cluster_size is not None:
        config.mode3.min_cluster_size = min_cluster_size
    if min_samples is not None:
        config.mode3.min_samples = min_samples

    n_real_sims = config.mode3.mcmc_end - config.mode3.mcmc_start + 1
    n_realizations = n_real_sims * config.mode3.num_samplings

    print("=" * 60)
    print("Mode 3: Random Control Simulation Analysis (HDBSCAN)")
    print("=" * 60)
    print(f"\nControl simulations: {config.mode3.mcmc_start} to {config.mode3.mcmc_end} ({n_real_sims} sims)")
    print(f"Samplings per simulation: {config.mode3.num_samplings}")
    print(f"Total virtual realizations: {n_realizations}")
    print(f"M200 mass cut: {config.mode3.m200_mass_cut:.2e} Msol")
    print(f"Radius cut: {config.mode3.radius_cut} Mpc")

    # Step 1: Load data
    print("\nStep 1: Loading random control data...")
    mcmc_data = load_random_control_data(config)

    # Combine halos
    combined_data, halo_provenance = combine_haloes(mcmc_data)

    positions = combined_data['SO/200_crit/CentreOfMass']
    masses = combined_data['SO/200_crit/TotalMass']
    realization_ids = np.array([p['mcmc_id'] for p in halo_provenance], dtype=np.int32)

    n_total_halos = len(positions)
    print(f"  Total halos loaded: {n_total_halos}")

    # Filter non-positive masses
    valid_mass_mask = masses > 0
    if not np.all(valid_mass_mask):
        n_invalid = np.sum(~valid_mass_mask)
        print(f"  Warning: Filtering {n_invalid} halos with non-positive masses")
        positions = positions[valid_mass_mask]
        masses = masses[valid_mass_mask]
        realization_ids = realization_ids[valid_mass_mask]
        for key in combined_data:
            if isinstance(combined_data[key], np.ndarray) and len(combined_data[key]) == n_total_halos:
                combined_data[key] = combined_data[key][valid_mass_mask]
        n_total_halos = len(positions)

    # Step 2: Run HDBSCAN
    print(f"\nStep 2: Running HDBSCAN on 3D positions (min_cluster_size={config.mode3.min_cluster_size}, "
          f"min_samples={config.mode3.min_samples})...")
    labels, probabilities, clusterer = run_hdbscan(
        positions,
        config.mode3.min_cluster_size,
        config.mode3.min_samples,
        'eom'
    )

    n_clusters_raw = len(np.unique(labels[labels != -1]))
    n_noise_raw = np.sum(labels == -1)
    n_clustered_raw = n_total_halos - n_noise_raw

    print(f"  Raw clustering results:")
    print(f"    Clusters found: {n_clusters_raw}")
    print(f"    Clustered halos: {n_clustered_raw} ({100*n_clustered_raw/n_total_halos:.1f}%)")
    print(f"    Noise halos: {n_noise_raw} ({100*n_noise_raw/n_total_halos:.1f}%)")

    # Step 3: Enforce one-per-realization constraint
    print("\nStep 3: Enforcing one-per-realization constraint...")
    labels, dropped_mask, ambiguity_rates = enforce_one_per_realization(
        labels, probabilities, positions, realization_ids, n_realizations
    )

    n_dropped = np.sum(dropped_mask)
    n_clusters_final = len(np.unique(labels[labels != -1]))
    n_noise_final = np.sum(labels == -1)
    n_clustered_final = n_total_halos - n_noise_final

    print(f"  Halos dropped by constraint: {n_dropped}")
    print(f"  Final clustered halos: {n_clustered_final} ({100*n_clustered_final/n_total_halos:.1f}%)")
    print(f"  Final noise halos: {n_noise_final} ({100*n_noise_final/n_total_halos:.1f}%)")

    # Step 4: Summarize clusters
    print("\nStep 4: Computing cluster summaries...")
    cluster_summaries = summarize_clusters(
        labels, probabilities, positions, masses, realization_ids, combined_data,
        clusterer, ambiguity_rates, n_realizations,
        existence_prob_stable=0.5,
        existence_prob_tentative=0.2
    )

    n_stable = sum(1 for c in cluster_summaries if c['status'] == 'stable')
    n_tentative = sum(1 for c in cluster_summaries if c['status'] == 'tentative')
    n_rare = sum(1 for c in cluster_summaries if c['status'] == 'rare')

    print(f"  Total clusters: {len(cluster_summaries)}")
    print(f"  Stable (existence_prob >= 50%): {n_stable}")
    print(f"  Tentative (>= 20%): {n_tentative}")
    print(f"  Rare (< 20%): {n_rare}")

    # Show top clusters
    if len(cluster_summaries) > 0:
        print("\nTop 10 clusters by existence probability:")
        for cluster in cluster_summaries[:10]:
            print(f"\n  Cluster {cluster['cluster_id']}:")
            print(f"    Existence prob: {cluster['existence_prob']:.1%} ({cluster['n_realizations_present']}/{n_realizations})")
            print(f"    Members: {cluster['n_members']}")
            print(f"    M200 mean: {cluster['mean_m200_mass']:.2e} Msol")
            print(f"    Log M200 std: {cluster['log10_m200_mass_std']:.3f} dex")

    # Step 5: Save results
    print("\nStep 5: Saving results to HDF5...")
    filename = f"random_control_clusters_mcs_{config.mode3.min_cluster_size}_ms_{config.mode3.min_samples}.h5"

    # Create a mock mode1 config for save_hdbscan_clusters_to_hdf5
    from dataclasses import dataclass

    @dataclass
    class Mode1Mock:
        mcmc_start: int = 0
        mcmc_end: int = n_realizations - 1
        m200_mass_cut: float = config.mode3.m200_mass_cut
        radius_cut: float = config.mode3.radius_cut
        min_cluster_size: int = config.mode3.min_cluster_size
        min_samples: int = config.mode3.min_samples
        cluster_selection_method: str = 'eom'
        existence_prob_stable: float = 0.5
        existence_prob_tentative: float = 0.2

    config.mode1 = Mode1Mock()

    save_hdbscan_clusters_to_hdf5(
        cluster_summaries=cluster_summaries,
        positions=positions,
        masses=masses,
        realization_ids=realization_ids,
        labels=labels,
        probabilities=probabilities,
        dropped_mask=dropped_mask,
        config=config,
        output_dir=output_dir,
        filename=filename,
        sigma_diagnostics={'method': 'pure_3d'},
        sigma_logM=None
    )

    print("\n" + "=" * 60)
    print("Mode 3 complete!")
    print(f"Output file: {output_dir}/{filename}")
    print("=" * 60)

    return cluster_summaries, labels, probabilities


if __name__ == '__main__':
    run_mode3()
