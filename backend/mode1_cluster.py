"""
Mode 1: HDBSCAN Posterior Clustering Pipeline

Performs pure 3D position-based clustering on combined halo catalogs from MCMC
posterior resimulations to identify stable halo associations.

Key features:
- Pure 3D spatial clustering (positions only, no whitening)
- HDBSCAN for density-based clustering with soft cluster membership
- Explicit noise handling
- One-per-realization constraint enforcement
- Comprehensive stability/uncertainty metrics
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

try:
    import hdbscan
except ImportError:
    raise ImportError("hdbscan package required. Install with: pip install hdbscan")

from backend.config_loader import load_config, Mode1Config
from backend.io import ensure_output_dir, save_hdbscan_clusters_to_hdf5
from backend.common_clustering import load_data_with_radius_filter, combine_haloes, _compute_shape_measures


# Temporary dataclass for compatibility with functions expecting mode1 config
@dataclass
class Mode1Wrapper:
    mcmc_start: int
    mcmc_end: int
    m200_mass_cut: float
    radius_cut: float
    min_cluster_size: int
    min_samples: int


def load_halo_data(config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Load and combine halos from all realizations.

    Returns:
        positions: (N, 3) array of x,y,z coordinates
        masses: (N,) array of M200 masses
        realization_ids: (N,) array of MCMC sample indices
        halo_indices: (N,) array of original indices within each realization
        combined_data: dict of all halo properties
    """
    from copy import deepcopy

    # Create wrapper config for compatibility
    config_wrapper = deepcopy(config)
    config_wrapper.mode1 = Mode1Wrapper(
        mcmc_start=config.mode1.mcmc_start,
        mcmc_end=config.mode1.mcmc_end,
        m200_mass_cut=config.mode1.m200_mass_cut,
        radius_cut=config.mode1.radius_cut,
        min_cluster_size=config.mode1.min_cluster_size,
        min_samples=config.mode1.min_samples
    )

    mcmc_data = load_data_with_radius_filter(config_wrapper)
    combined_data, halo_provenance = combine_haloes(mcmc_data)

    positions = combined_data['SO/200_crit/CentreOfMass']
    masses = combined_data['SO/200_crit/TotalMass']
    realization_ids = np.array([p['mcmc_id'] for p in halo_provenance], dtype=np.int32)
    halo_indices = np.array([p['original_index'] for p in halo_provenance], dtype=np.int32)

    return positions, masses, realization_ids, halo_indices, combined_data


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

    Args:
        labels: (N,) cluster labels
        probabilities: (N,) membership probabilities
        positions: (N, 3) halo positions
        realization_ids: (N,) realization IDs
        n_realizations: total number of realizations

    Returns:
        labels: updated labels (rejected halos become -1)
        dropped_mask: (N,) bool array of dropped halos
        ambiguity_rates: {cluster_id: fraction of realizations with conflicts}
    """
    labels = labels.copy()
    dropped_mask = np.zeros(len(labels), dtype=bool)
    ambiguity_rates = {}

    unique_clusters = np.unique(labels)

    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise
            continue

        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_positions = positions[cluster_mask]
        cluster_probs = probabilities[cluster_mask]
        cluster_realization_ids = realization_ids[cluster_mask]

        # Compute cluster center
        cluster_center = np.median(cluster_positions, axis=0)

        # Track conflicts
        n_conflicts = 0
        unique_realizations = np.unique(cluster_realization_ids)

        for real_id in unique_realizations:
            real_mask = cluster_realization_ids == real_id
            real_indices = cluster_indices[real_mask]

            if len(real_indices) <= 1:
                continue

            # Multiple halos from same realization - need to select one
            n_conflicts += 1

            real_probs = cluster_probs[real_mask]
            real_positions = cluster_positions[real_mask]

            # Primary: highest probability
            max_prob = np.max(real_probs)
            max_prob_mask = real_probs == max_prob

            if np.sum(max_prob_mask) == 1:
                # Clear winner by probability
                keep_idx = np.argmax(real_probs)
            else:
                # Tie-break: closest to center among max-probability halos
                tied_positions = real_positions[max_prob_mask]
                tied_indices = np.where(max_prob_mask)[0]
                distances = np.linalg.norm(tied_positions - cluster_center, axis=1)
                keep_idx = tied_indices[np.argmin(distances)]

            # Mark non-selected as noise
            for i, global_idx in enumerate(real_indices):
                if i != keep_idx:
                    labels[global_idx] = -1
                    dropped_mask[global_idx] = True

        # Compute ambiguity rate
        n_realizations_present = len(unique_realizations)
        ambiguity_rates[cluster_id] = n_conflicts / n_realizations_present if n_realizations_present > 0 else 0.0

    return labels, dropped_mask, ambiguity_rates


def get_hdbscan_stability(clusterer: 'hdbscan.HDBSCAN', cluster_id: int) -> float:
    """Extract HDBSCAN stability score for a cluster.

    The stability score measures how persistent a cluster is across
    different density thresholds in the condensed tree.
    """
    try:
        # Get cluster persistence from condensed tree
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
    """Compute comprehensive cluster summaries.

    Per-cluster metrics:
    - cluster_id (deterministic: sorted by existence_prob desc, stability desc)
    - existence_prob = n_realizations_present / N_realizations
    - n_realizations_present
    - n_members_total
    - center_x, center_y, center_z (median position)
    - cov_xyz (3x3 position covariance)
    - center_logM (median log mass)
    - var_logM
    - hdbscan_stability (from condensed tree)
    - mean_membership_prob, min_membership_prob
    - ambiguity_rate
    - status ('stable', 'tentative', 'rare')
    """
    log_masses = np.log10(masses)
    cluster_summaries = []

    unique_clusters = np.unique(labels)
    unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise

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

        # Position statistics
        center_xyz = np.median(cluster_positions, axis=0)
        cov_xyz = np.cov(cluster_positions.T) if n_members > 1 else np.eye(3) * np.nan

        # Mass statistics
        center_logM = np.median(cluster_log_masses)
        var_logM = np.var(cluster_log_masses) if n_members > 1 else np.nan
        mean_mass = np.mean(cluster_masses)
        mass_std = np.std(cluster_masses) if n_members > 1 else np.nan
        log_mass_std = np.std(cluster_log_masses) if n_members > 1 else np.nan

        # Membership probability statistics
        mean_prob = np.mean(cluster_probs)
        min_prob = np.min(cluster_probs)

        # HDBSCAN stability
        hdbscan_stability = get_hdbscan_stability(clusterer, cluster_id)

        # Ambiguity rate
        ambiguity_rate = ambiguity_rates.get(cluster_id, 0.0)

        # Status classification
        if existence_prob >= existence_prob_stable:
            status = 'stable'
        elif existence_prob >= existence_prob_tentative:
            status = 'tentative'
        else:
            status = 'rare'

        # Shape measures
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

        # Extract member data
        member_data = {}
        for key, data in combined_data.items():
            member_data[key] = data[cluster_mask]

        cluster_summaries.append({
            'original_cluster_id': int(cluster_id),  # Original HDBSCAN ID
            'cluster_id': None,  # Will be assigned after sorting
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

    # Sort by existence_prob (desc), then hdbscan_stability (desc)
    cluster_summaries.sort(
        key=lambda x: (-x['existence_prob'], -x['hdbscan_stability'] if not np.isnan(x['hdbscan_stability']) else 0)
    )

    # Assign deterministic cluster IDs
    for i, summary in enumerate(cluster_summaries):
        summary['cluster_id'] = i

    return cluster_summaries


def run_mode1(config_path="config.toml", output_dir="output",
              min_cluster_size=None, min_samples=None):
    """
    Mode 1: Pure 3D HDBSCAN Posterior Clustering

    Performs:
    1. Load data from MCMC samples
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
        config.mode1.min_cluster_size = min_cluster_size
    if min_samples is not None:
        config.mode1.min_samples = min_samples

    n_realizations = config.mode1.mcmc_end - config.mode1.mcmc_start + 1

    print("=" * 60)
    print("Mode 1: Pure 3D HDBSCAN Posterior Clustering")
    print("=" * 60)

    # Step 1: Load data
    print("\nStep 1: Loading MCMC data...")
    positions, masses, realization_ids, halo_indices, combined_data = load_halo_data(config)

    n_total_halos = len(positions)
    print(f"  Total halos loaded: {n_total_halos}")
    print(f"  Realizations: {config.mode1.mcmc_start} to {config.mode1.mcmc_end} ({n_realizations} total)")

    # Filter non-positive masses
    valid_mass_mask = masses > 0
    if not np.all(valid_mass_mask):
        n_invalid = np.sum(~valid_mass_mask)
        print(f"  Warning: Filtering {n_invalid} halos with non-positive masses")
        positions = positions[valid_mass_mask]
        masses = masses[valid_mass_mask]
        realization_ids = realization_ids[valid_mass_mask]
        halo_indices = halo_indices[valid_mass_mask]
        for key in combined_data:
            if isinstance(combined_data[key], np.ndarray) and len(combined_data[key]) == n_total_halos:
                combined_data[key] = combined_data[key][valid_mass_mask]
        n_total_halos = len(positions)

    # Step 2: Run HDBSCAN on 3D positions
    print(f"\nStep 2: Running HDBSCAN on 3D positions (min_cluster_size={config.mode1.min_cluster_size}, "
          f"min_samples={config.mode1.min_samples}, method='{config.mode1.cluster_selection_method}')...")
    labels, probabilities, clusterer = run_hdbscan(
        positions,
        config.mode1.min_cluster_size,
        config.mode1.min_samples,
        config.mode1.cluster_selection_method
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
    if ambiguity_rates:
        mean_ambiguity = np.mean(list(ambiguity_rates.values()))
        print(f"  Mean ambiguity rate: {mean_ambiguity:.1%}")

    # Step 4: Summarize clusters
    print("\nStep 4: Computing cluster summaries...")
    cluster_summaries = summarize_clusters(
        labels, probabilities, positions, masses, realization_ids, combined_data,
        clusterer, ambiguity_rates, n_realizations,
        config.mode1.existence_prob_stable,
        config.mode1.existence_prob_tentative
    )

    n_stable = sum(1 for c in cluster_summaries if c['status'] == 'stable')
    n_tentative = sum(1 for c in cluster_summaries if c['status'] == 'tentative')
    n_rare = sum(1 for c in cluster_summaries if c['status'] == 'rare')

    print(f"  Total clusters: {len(cluster_summaries)}")
    print(f"  Stable (existence_prob >= {config.mode1.existence_prob_stable:.0%}): {n_stable}")
    print(f"  Tentative (>= {config.mode1.existence_prob_tentative:.0%}): {n_tentative}")
    print(f"  Rare (< {config.mode1.existence_prob_tentative:.0%}): {n_rare}")

    # Show top clusters
    print("\nTop 10 clusters by existence probability:")
    for i, cluster in enumerate(cluster_summaries[:10]):
        print(f"\n  Cluster {cluster['cluster_id']}:")
        print(f"    Existence prob: {cluster['existence_prob']:.1%} ({cluster['n_realizations_present']}/{n_realizations})")
        print(f"    Status: {cluster['status']}")
        print(f"    Members: {cluster['n_members']}")
        print(f"    M200 mean: {cluster['mean_m200_mass']:.2e} Msol")
        print(f"    Log M200 std: {cluster['log10_m200_mass_std']:.3f} dex")
        print(f"    Position std: [{cluster['position_std'][0]:.2f}, {cluster['position_std'][1]:.2f}, {cluster['position_std'][2]:.2f}] Mpc")
        print(f"    Center: [{cluster['center_xyz'][0]:.1f}, {cluster['center_xyz'][1]:.1f}, {cluster['center_xyz'][2]:.1f}]")
        print(f"    Mean membership prob: {cluster['mean_membership_prob']:.3f}")
        print(f"    Ambiguity rate: {cluster['ambiguity_rate']:.1%}")

    # Step 5: Save results
    print("\nStep 5: Saving results to HDF5...")
    filename = (f"hdbscan_clusters_mcs_{config.mode1.min_cluster_size}_"
                f"ms_{config.mode1.min_samples}.h5")

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
    print("Mode 1 complete!")
    print(f"Output file: {output_dir}/{filename}")
    print("=" * 60)

    return cluster_summaries, labels, probabilities


def run_synthetic_test():
    """Run synthetic test to verify pipeline recovery."""
    print("=" * 60)
    print("Running Synthetic Test")
    print("=" * 60)

    np.random.seed(42)

    # Generate synthetic data
    n_realizations = 80
    n_true_halos = 5
    clutter_per_realization = 10

    # True halo properties
    true_positions = np.array([
        [100, 100, 100],
        [200, 200, 200],
        [300, 150, 250],
        [400, 300, 100],
        [250, 350, 300]
    ], dtype=float)

    true_masses = np.array([1e15, 5e14, 2e14, 8e14, 3e14])

    # Generate realizations
    all_positions = []
    all_masses = []
    all_realization_ids = []

    for real_id in range(n_realizations):
        # Add true halos with scatter
        for i in range(n_true_halos):
            # Position scatter (3 Mpc for all - testing pure 3D clustering)
            pos_scatter = 3.0

            pos = true_positions[i] + np.random.normal(0, pos_scatter, 3)
            mass = true_masses[i] * 10**(np.random.normal(0, 0.1))  # 0.1 dex scatter

            all_positions.append(pos)
            all_masses.append(mass)
            all_realization_ids.append(real_id)

        # Add clutter
        for _ in range(clutter_per_realization):
            pos = np.random.uniform(50, 450, 3)
            mass = 10**(np.random.uniform(14.0, 15.5))
            all_positions.append(pos)
            all_masses.append(mass)
            all_realization_ids.append(real_id)

    positions = np.array(all_positions)
    masses = np.array(all_masses)
    realization_ids = np.array(all_realization_ids)

    print(f"\nSynthetic data:")
    print(f"  Total halos: {len(positions)}")
    print(f"  Realizations: {n_realizations}")
    print(f"  True halos per realization: {n_true_halos}")
    print(f"  Clutter per realization: {clutter_per_realization}")

    # Run pipeline
    print("\nRunning HDBSCAN pipeline on 3D positions...")

    # Run HDBSCAN directly on positions
    min_cluster_size = max(10, round(0.15 * n_realizations))
    labels, probabilities, clusterer = run_hdbscan(positions, min_cluster_size, min_cluster_size, 'eom')

    # Enforce constraint
    labels, dropped_mask, ambiguity_rates = enforce_one_per_realization(
        labels, probabilities, positions, realization_ids, n_realizations
    )

    # Check recovery
    print("\nRecovery analysis:")
    unique_clusters = np.unique(labels[labels != -1])

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        cluster_reals = np.unique(realization_ids[mask])
        existence_prob = len(cluster_reals) / n_realizations

        cluster_positions = positions[mask]
        center = np.median(cluster_positions, axis=0)

        # Find closest true halo
        distances_to_true = [np.linalg.norm(center - tp) for tp in true_positions]
        closest_true = np.argmin(distances_to_true)
        closest_distance = distances_to_true[closest_true]

        print(f"\n  Cluster {cluster_id}:")
        print(f"    Existence prob: {existence_prob:.1%}")
        print(f"    Center: {center}")
        print(f"    Closest true halo: {closest_true} (distance: {closest_distance:.1f})")
        print(f"    True position: {true_positions[closest_true]}")

    # Count high-existence-prob clusters
    n_recovered = 0
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        cluster_reals = np.unique(realization_ids[mask])
        if len(cluster_reals) / n_realizations >= 0.5:
            n_recovered += 1

    print(f"\nSummary:")
    print(f"  True halos: {n_true_halos}")
    print(f"  Recovered with existence_prob >= 50%: {n_recovered}")
    print(f"  Recovery rate: {n_recovered/n_true_halos:.1%}")

    return n_recovered == n_true_halos


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-test", action="store_true",
                        help="Run synthetic test instead of real data")
    parser.add_argument("--config", default="config.toml",
                        help="Path to config file")
    parser.add_argument("--output", default="output",
                        help="Output directory")
    parser.add_argument("--min-cluster-size", type=int, default=None,
                        help="Override HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="Override HDBSCAN min_samples")
    args = parser.parse_args()

    if args.synthetic_test:
        success = run_synthetic_test()
        exit(0 if success else 1)
    else:
        run_mode1(
            config_path=args.config,
            output_dir=args.output,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples
        )
