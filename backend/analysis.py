import numpy as np
from scipy.spatial import ConvexHull
from .io import load_halo_traces_index, load_specific_halo_traces, load_single_cluster_traces
from .math_utils import _mean_matter_density_Msun_per_Mpc3, _lagrangian_radius_from_mass, _dimensionless_covariance_metrics
from .trace_processing import _get_initial_final_positions, _apply_matched_final_affine
from typing import Dict, List, Any

__all__ = [
    'analyze_volume_ratios_batch',
    'find_control_matches_batch',
    'find_control_matches_and_recenter_single',
    'calculate_lagrangian_volume'
]

def analyze_volume_ratios_batch(clusters,
                                config,
                                trace_filename,
                                min_cluster_size=40,
                                mass_tolerance_dex=0.1,
                                target_snapshot=10,
                                use_matched_final: bool = False):
    """
    Enhanced batch analysis: returns mass, legacy volume ratio (convex hull),
    dimensionless scatter ratio (s_ctrl/s_data), and information gain (bits).
    """
    # Filter clusters by size
    large_clusters = [cluster for cluster in clusters if cluster['cluster_size'] >= min_cluster_size]
    
    print(f"Loading control index...")
    control_filename = f"control_halo_traces_mass_{config.mode4.m200_mass_cut:.1e}_radius_{config.mode4.radius_cut}.h5"
    control_index = load_halo_traces_index(config.global_config.output_dir, filename=control_filename)
    if control_index is None:
        print(f"Could not load control index from {control_filename}")
        return (np.array([]), np.array([]), np.array([]), np.array([]), [], np.array([]))
    
    print(f"Loading constrained traces for {len(large_clusters)} clusters...")
    constrained_traces_dict = {}
    for cluster in large_clusters:
        cluster_id = cluster['cluster_id']
        traces = load_single_cluster_traces(cluster_id, config.global_config.output_dir, filename=trace_filename)
        if traces:
            constrained_traces_dict[cluster_id] = traces
    
    print(f"Processing control matches...")
    cluster_results = find_control_matches_batch(
        constrained_traces_dict, control_index, control_filename, config,
        mass_tolerance_dex=mass_tolerance_dex, matched_final=use_matched_final
    )
    
    print(f"Computing localization metrics (covariance + convex hull)...")
    # Outputs
    cluster_masses = []
    cluster_ids = []
    legacy_ratio = []          # convex-hull control/constrained
    s_ratio_list = []          # dimensionless s_ctrl/s_data
    info_bits_list = []        # info gain in bits
    distances = []             # optional: observer distance
    
    # constants
    rho_m = _mean_matter_density_Msun_per_Mpc3()
    observer = np.array(config.global_config.observer_coords, dtype=float)
    cluster_lookup = {c['cluster_id']: c for c in large_clusters}
    
    for cluster_id, results in cluster_results.items():
        constrained_traces = results['constrained_traces']
        control_traces = results['control_traces']
        if len(control_traces) < 4 or len(constrained_traces) < 4:
            continue
        
        # --- Legacy convex hull volumes ---
        constrained_volume = calculate_lagrangian_volume(constrained_traces, target_snapshot)
        control_volume = calculate_lagrangian_volume(control_traces, target_snapshot)
        
        # --- Covariance-based metrics ---
        Xinit_data, Xfin_data = _get_initial_final_positions(constrained_traces, init_snap=target_snapshot, final_snap=77)
        Xinit_ctrl, Xfin_ctrl = _get_initial_final_positions(control_traces, init_snap=target_snapshot, final_snap=77)
        if Xinit_data.shape[0] < 2 or Xinit_ctrl.shape[0] < 2:
            continue
        
        # Matched-final affine control (if requested)
        if use_matched_final:
            Xinit_ctrl = _apply_matched_final_affine(Xinit_ctrl, Xfin_ctrl, Xfin_data)
            # Note: after affine, Xinit_ctrl is already centered in the ctrl-final frame
        
        # Lagrangian radius from the association's mean M200
        M200 = float(cluster_lookup[cluster_id]['mean_m200_mass'])
        R_L = _lagrangian_radius_from_mass(M200, rho_m)
        
        metrics = _dimensionless_covariance_metrics(Xinit_data, Xinit_ctrl, R_L)
        
        # push results if legacy available
        ok_legacy = (constrained_volume is not None and control_volume is not None and constrained_volume > 0.0)
        if ok_legacy:
            legacy_ratio.append(control_volume / constrained_volume)
            cluster_masses.append(M200)
            cluster_ids.append(cluster_id)
            s_ratio_list.append(metrics["s_ratio"])
            info_bits_list.append(metrics["info_bits"])
            # observer distance of mean position
            mean_pos = np.array(cluster_lookup[cluster_id]['mean_position'])
            distances.append(float(np.linalg.norm(mean_pos - observer)))
    
    return (np.array(cluster_masses),
            np.array(legacy_ratio),
            np.array(s_ratio_list),
            np.array(info_bits_list),
            cluster_ids,
            np.array(distances))


def calculate_lagrangian_volume(traces, target_snapshot=10):
    """
    Calculate convex hull volume of initial positions for a set of halo traces.

    Parameters:
    -----------
    traces : list
        List of trace dictionaries
    target_snapshot : int
        Snapshot to use for volume calculation (default: 10)

    Returns:
    --------
    float : Volume in Mpc^3, or None if insufficient points
    """
    initial_positions = []

    for trace in traces:
        snapshots = trace['snapshots']
        positions = trace['BoundSubhalo/CentreOfMass']

        # Find the earliest available snapshot >= target_snapshot
        valid_snaps = snapshots[snapshots >= target_snapshot]
        if len(valid_snaps) > 0:
            earliest_snap = np.min(valid_snaps)
            snap_idx = np.where(snapshots == earliest_snap)[0][0]
            initial_positions.append(positions[snap_idx])

    if len(initial_positions) < 4:  # Need at least 4 points for 3D convex hull
        return None

    try:
        hull = ConvexHull(np.array(initial_positions))
        return hull.volume
    except Exception:
        return None

def find_control_matches_batch(
    constrained_traces_dict: Dict[int, List[Dict[str, Any]]],
    control_index: Dict[str, Any],
    control_filename: str,
    config: Any,
    mass_tolerance_dex: float = 0.1,
    matched_final: bool = False,
    recenter: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Batch process control matches for multiple clusters using a cached control index.

    Parameters
    ----------
    constrained_traces_dict : dict
        {cluster_id: [list of constrained traces]}
    control_index : dict
        Preloaded index for control haloes (masses, positions, keys).
        Must contain: 'final_m200_masses', 'final_positions', 'halo_keys'
    control_filename : str
        Filename of the control halo trace HDF5.
    config : object
        Configuration object.
    mass_tolerance_dex : float
        Allowed log10 mass difference for matching control haloes.
    matched_final : bool
        If True, this function also computes and returns affine parameters
        (A, ctrl_final_mean, data_final_mean) based on the final positions.
    recenter : bool
        If True, shift control haloes so their final centroid matches the constrained cluster centroid.
        If False, keep controls in their original coordinates.

    Returns
    -------
    cluster_results : dict
        {
          cluster_id: {
            'constrained_traces': [...],
            'control_traces': [...],
            'matched_final': bool,
            # NEW when matched_final=True and enough points:
            'affine_params': {
                'A': (3,3) ndarray,
                'ctrl_final_mean': (3,) ndarray,
                'data_final_mean': (3,) ndarray
            }  # present only if computed successfully
          }, ...
        }
    """
    cluster_results: Dict[int, Dict[str, Any]] = {}

    # Extract control arrays once
    control_masses = control_index['final_m200_masses']
    control_positions = control_index['final_positions']
    control_keys = control_index['halo_keys']

    # Collect all needed control keys across all clusters
    all_needed_control_keys = set()
    cluster_match_info: Dict[int, Dict[str, Any]] = {}

    for cluster_id, constrained_traces in constrained_traces_dict.items():
        # Get final masses and positions for constrained haloes
        constrained_final_masses: List[float] = []
        constrained_final_positions: List[np.ndarray] = []

        for trace in constrained_traces:
            final_idx = np.where(trace['snapshots'] == 77)[0]
            if len(final_idx) == 0:
                continue
            fi = final_idx[0]

            # mass
            if 'BoundSubhalo/TotalMass' in trace:
                final_mass = trace['BoundSubhalo/TotalMass'][fi]
            elif 'SO/200_crit/TotalMass' in trace:
                final_mass = trace['SO/200_crit/TotalMass'][fi]
            else:
                continue

            # position
            if 'BoundSubhalo/CentreOfMass' in trace:
                final_pos = trace['BoundSubhalo/CentreOfMass'][fi]
            elif 'SO/200_crit/CentreOfMass' in trace:
                final_pos = trace['SO/200_crit/CentreOfMass'][fi]
            else:
                continue

            constrained_final_masses.append(final_mass)
            constrained_final_positions.append(final_pos)

        if len(constrained_final_masses) == 0:
            continue

        constrained_final_positions = np.array(constrained_final_positions, dtype=float)
        constrained_mean_position = np.mean(constrained_final_positions, axis=0)

        # Vectorized control mass matching
        log_control_masses = np.log10(control_masses)
        matched_control_keys: List[str] = []
        used_indices = set()

        for m in constrained_final_masses:
            log_m = np.log10(m)
            mass_diffs = np.abs(log_control_masses - log_m)
            valid = np.where(mass_diffs <= mass_tolerance_dex)[0]
            # avoid duplicate picks
            valid = [idx for idx in valid if idx not in used_indices]
            if len(valid) == 0:
                continue
            best_local = int(valid[np.argmin(mass_diffs[valid])])
            matched_control_keys.append(control_keys[best_local])
            used_indices.add(best_local)
            all_needed_control_keys.add(control_keys[best_local])

        cluster_match_info[cluster_id] = {
            'constrained_traces': constrained_traces,
            'matched_control_keys': matched_control_keys,
            'constrained_mean_position': constrained_mean_position,
        }

    # Batch load control traces
    all_control_traces = load_specific_halo_traces(
        list(all_needed_control_keys),
        config.global_config.output_dir,
        filename=control_filename,
    )

    for cluster_id, match_info in cluster_match_info.items():
        matched_control_keys = match_info['matched_control_keys']
        constrained_mean_position = match_info['constrained_mean_position']
        constrained_traces = match_info['constrained_traces']

        recentered_control_traces: List[Dict[str, Any]] = []
        for control_key in matched_control_keys:
            if control_key not in all_control_traces:
                continue

            rec = all_control_traces[control_key].copy()

            if recenter:
                # Get control final position from index (same ordering as control_keys)
                try:
                    control_idx = control_keys.index(control_key)
                except ValueError:
                    # Shouldn't happen; skip if it does
                    continue
                control_final_pos = control_positions[control_idx]
                offset = constrained_mean_position - control_final_pos

                # Apply offset to all snapshots
                for poskey in ['BoundSubhalo/CentreOfMass', 'SO/200_crit/CentreOfMass']:
                    if poskey in rec:
                        rec[poskey] = rec[poskey] + offset

            recentered_control_traces.append(rec)

        # Prepare return payload for this cluster
        payload: Dict[str, Any] = {
            'constrained_traces': constrained_traces,
            'control_traces': recentered_control_traces,
            'matched_final': matched_final,
        }

        # If requested, compute and attach affine params (if enough points)
        if matched_final and len(recentered_control_traces) >= 2 and len(constrained_traces) >= 2:
            # Extract final positions for both sets to compute A
            # We rely on the same helper used elsewhere to respect your trace schema.
            Xinit_data, Xfin_data = _get_initial_final_positions(constrained_traces, init_snap=10, final_snap=77)
            Xinit_ctrl, Xfin_ctrl = _get_initial_final_positions(recentered_control_traces, init_snap=10, final_snap=77)

            try:
                if Xfin_ctrl.shape[0] >= 2 and Xfin_data.shape[0] >= 2:
                    A, mu_ctrl, mu_data = _compute_affine_from_final_clouds(Xfin_ctrl, Xfin_data)
                    payload['affine_params'] = {
                        'A': A,
                        'ctrl_final_mean': mu_ctrl,
                        'data_final_mean': mu_data,
                    }
            except Exception:
                # Be robust: if affine fails (e.g., degenerate covariance), just omit it.
                pass

        cluster_results[cluster_id] = payload

    return cluster_results


def find_control_matches_and_recenter_single(
    cluster_id: int,
    config: Any,
    trace_filename: str,
    mass_tolerance_dex: float = 0.1,
    use_matched_final: bool = False,
    recenter_controls: bool = True,
):
    """
    Convenience wrapper: load traces for one constrained cluster, match
    mass-matched controls, optionally recenter (and optionally matched-final),
    and return both lists.

    Parameters
    ----------
    cluster_id : int
        ID of the constrained cluster to load.
    config : object
        Configuration object.
    trace_filename : str
        Filename of the constrained trace HDF5.
    mass_tolerance_dex : float
        Allowed log10 mass difference for matching control haloes.
    use_matched_final : bool
        If True, compute affine params from final clouds as part of the batch.
    recenter_controls : bool
        If False, return controls in their *true* original coordinates without recentering.

    Returns
    -------
    constrained, controls, affine_params
        - constrained: list of constrained traces (or None)
        - controls: list of matched control traces (or None)
        - affine_params: dict with keys {'A', 'ctrl_final_mean', 'data_final_mean'} if computed,
                         otherwise None.
    """
    # Load control index
    control_filename = f"control_halo_traces_mass_{config.mode4.m200_mass_cut:.1e}_radius_{config.mode4.radius_cut}.h5"
    control_index = load_halo_traces_index(config.global_config.output_dir, filename=control_filename)
    if control_index is None:
        print(f"[warn] Could not load control index from {control_filename}")
        return None, None, None

    # Load constrained traces for this cluster
    constrained_traces = load_single_cluster_traces(cluster_id, config.global_config.output_dir, filename=trace_filename)
    if not constrained_traces:
        print(f"[warn] No constrained traces for cluster {cluster_id}")
        return None, None, None

    constrained_traces_dict = {cluster_id: constrained_traces}
    results = find_control_matches_batch(
        constrained_traces_dict,
        control_index,
        control_filename,
        config,
        mass_tolerance_dex=mass_tolerance_dex,
        matched_final=use_matched_final if recenter_controls else False,
        recenter=recenter_controls,
    )

    if cluster_id not in results:
        print(f"[warn] No control matches returned for cluster {cluster_id}")
        return None, None, None

    affine = results[cluster_id].get('affine_params', None)
    return results[cluster_id]['constrained_traces'], results[cluster_id]['control_traces'], affine

