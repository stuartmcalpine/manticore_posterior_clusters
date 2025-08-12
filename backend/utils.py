import h5py
import numpy as np
import os
from typing import Dict, List, Any
from collections import defaultdict

__all__ = [
    'ensure_output_dir',
    'save_clusters_to_hdf5',
    'load_clusters_from_hdf5',
    'save_halo_traces_to_hdf5',
    'load_halo_traces_index',
    'load_specific_halo_traces',
    'load_halo_traces_from_hdf5',
    'get_cluster_trace_info',
    'load_single_cluster_traces',
    'find_clusters_in_window',
    'load_single_cluster_members',
    'calculate_lagrangian_volume',
    'find_control_matches_batch',
    'analyze_volume_ratios_batch',
    'find_control_matches_and_recenter_single',
    '_get_initial_final_positions',
    '_apply_matched_final_affine',
    '_mean_matter_density_Msun_per_Mpc3',
    '_lagrangian_radius_from_mass',
    '_dimensionless_covariance_metrics'
]

# ---------- Linear algebra helpers ----------

def _safe_covariance(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Return 3x3 covariance of points (N,3) with small Tikhonov regularization.
    """
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 2:
        raise ValueError("Need at least 2 points of shape (N,3) for covariance")
    C = np.cov(X.T, bias=False)
    # Regularize to keep positive-definite
    return C + eps * np.eye(3)

def _sym_eig_sqrt_inv(S: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Symmetric eigendecomposition -> matrix sqrt or inverse sqrt.
    """
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)  # avoid negatives
    if invert:
        W = np.diag(1.0 / np.sqrt(w))
    else:
        W = np.diag(np.sqrt(w))
    return (V @ W) @ V.T

def _matrix_sqrt(S: np.ndarray) -> np.ndarray:
    return _sym_eig_sqrt_inv(S, invert=False)

def _matrix_invsqrt(S: np.ndarray) -> np.ndarray:
    return _sym_eig_sqrt_inv(S, invert=True)

# ---------- Cosmology / Lagrangian radius ----------

def _mean_matter_density_Msun_per_Mpc3() -> float:
    """
    Mean comoving matter density rho_m [Msun / Mpc^3] for fixed Omega_m, h.
    """
    Omega_m = 0.306
    h = 0.681
    # Critical density rho_c0 ≈ 2.77536627e11 * h^2 Msun / Mpc^3
    rho_c0 = 2.77536627e11 * (h ** 2)
    return Omega_m * rho_c0


def _lagrangian_radius_from_mass(M200: float, rho_m_Msun_per_Mpc3: float) -> float:
    """
    Lagrangian radius R_L = (3 M / (4π rho_m))^(1/3) in Mpc.
    Use M200 (Msun) for M; rho_m is comoving mean density (Msun/Mpc^3).
    """
    return ((3.0 * M200) / (4.0 * np.pi * rho_m_Msun_per_Mpc3)) ** (1.0 / 3.0)

# ---------- Trace extraction ----------

def _extract_positions_at_or_after_snapshot(trace: dict, key: str, target_snapshot: int) -> np.ndarray | None:
    """
    Return 3D position from 'key' at earliest snapshot >= target_snapshot.
    """
    snapshots = trace['snapshots']
    positions = trace.get(key, None)
    if positions is None:
        return None
    valid = snapshots[snapshots >= target_snapshot]
    if len(valid) == 0:
        return None
    earliest = int(np.min(valid))
    idx = np.where(snapshots == earliest)[0][0]
    return positions[idx]

def _get_initial_final_positions(traces: list, init_snap: int, final_snap: int = 77) -> tuple[np.ndarray, np.ndarray]:
    """
    Return arrays (N,3) of initial and final positions for provided traces.
    Uses BoundSubhalo/CentreOfMass if present, else SO/200_crit/CentreOfMass.
    For initial: earliest snapshot >= init_snap; for final: exact 'final_snap'.
    """
    init_pts, fin_pts = [], []

    for tr in traces:
        # choose pos key
        if 'BoundSubhalo/CentreOfMass' in tr:
            poskey = 'BoundSubhalo/CentreOfMass'
        elif 'SO/200_crit/CentreOfMass' in tr:
            poskey = 'SO/200_crit/CentreOfMass'
        else:
            continue

        # initial
        p_init = _extract_positions_at_or_after_snapshot(tr, poskey, init_snap)
        # final
        snaps = tr['snapshots']
        fin_idx = np.where(snaps == final_snap)[0]
        if len(fin_idx) == 0 or p_init is None:
            continue
        p_fin = tr[poskey][fin_idx[0]]

        init_pts.append(p_init)
        fin_pts.append(p_fin)

    if len(init_pts) < 2 or len(fin_pts) < 2:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.asarray(init_pts), np.asarray(fin_pts)

# ---------- Matched-final affine mapping ----------

def _apply_matched_final_affine(X_init_ctrl: np.ndarray,
                                X_fin_ctrl: np.ndarray,
                                X_fin_data: np.ndarray) -> np.ndarray:
    """
    Build A = S_data * S_ctrl^{-1} from final covariances and apply to control initial residuals.
    Returns transformed control initial residuals in the control-final mean frame.
    """
    # center
    Xf_ctrl = X_fin_ctrl - X_fin_ctrl.mean(axis=0, keepdims=True)
    Xf_data = X_fin_data - X_fin_data.mean(axis=0, keepdims=True)
    Sf_ctrl = _safe_covariance(Xf_ctrl)
    Sf_data = _safe_covariance(Xf_data)

    A = _matrix_sqrt(Sf_data) @ _matrix_invsqrt(Sf_ctrl)

    Xi_ctrl = X_init_ctrl - X_init_ctrl.mean(axis=0, keepdims=True)
    Xi_ctrl_hat = (A @ Xi_ctrl.T).T
    return Xi_ctrl_hat

# ---------- Covariance-based metrics ----------

def _dimensionless_covariance_metrics(X_init_data: np.ndarray,
                                      X_init_ctrl: np.ndarray,
                                      R_L: float) -> dict:
    """
    Compute dimensionless scatter and information gain (nats & bits)
    from covariance matrices normalized by R_L^2.
    """
    # center
    Xi_d = X_init_data - X_init_data.mean(axis=0, keepdims=True)
    Xi_c = X_init_ctrl - X_init_ctrl.mean(axis=0, keepdims=True)

    Sd = _safe_covariance(Xi_d) / (R_L ** 2)
    Sc = _safe_covariance(Xi_c) / (R_L ** 2)

    # dimensionless scatter (RMS)
    s_data = float(np.sqrt(np.trace(Sd)))
    s_ctrl = float(np.sqrt(np.trace(Sc)))
    s_ratio = s_ctrl / s_data if s_data > 0 else np.nan

    # information gain (nats & bits): 0.5 * ln(det(Sc)/det(Sd))
    det_Sd = float(np.linalg.det(Sd))
    det_Sc = float(np.linalg.det(Sc))
    det_Sd = max(det_Sd, 1e-36)
    det_Sc = max(det_Sc, 1e-36)
    info_nats = 0.5 * np.log(det_Sc / det_Sd)
    info_bits = info_nats / np.log(2.0)

    # legacy 3D "volume" via covariance determinant^1/2 (dimensionless)
    vol_data_dimless = float(np.sqrt(det_Sd))
    vol_ctrl_dimless = float(np.sqrt(det_Sc))
    vol_ratio_dimless = vol_ctrl_dimless / vol_data_dimless if vol_data_dimless > 0 else np.nan

    return {
        "s_data": s_data, "s_ctrl": s_ctrl, "s_ratio": s_ratio,
        "info_nats": info_nats, "info_bits": info_bits,
        "vol_ratio_dimless": vol_ratio_dimless
    }


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


def find_control_matches_batch(constrained_traces_dict,
                               control_index,
                               control_filename,
                               config,
                               mass_tolerance_dex=0.1,
                               matched_final: bool = False,
                               recenter: bool = True):
    """
    Batch process control matches for multiple clusters using cached control index.

    Parameters
    ----------
    constrained_traces_dict : dict
        {cluster_id: [list of constrained traces]}
    control_index : dict
        Preloaded index for control haloes (masses, positions, keys).
    control_filename : str
        Filename of the control halo trace HDF5.
    config : object
        Configuration object.
    mass_tolerance_dex : float
        Allowed log10 mass difference for matching control haloes.
    matched_final : bool
        If True, downstream analysis will apply matched-final affine transformation.
    recenter : bool
        If True, shift control haloes so their final centroid matches the constrained cluster centroid.
        If False, keep controls in their original coordinates.
    """
    cluster_results = {}
    
    # Extract control arrays once
    control_masses = control_index['final_m200_masses']
    control_positions = control_index['final_positions']
    control_keys = control_index['halo_keys']
    
    # Collect all needed control keys across all clusters
    all_needed_control_keys = set()
    cluster_match_info = {}
    
    for cluster_id, constrained_traces in constrained_traces_dict.items():
        # Get final masses and positions for constrained haloes
        constrained_final_masses = []
        constrained_final_positions = []
        
        for trace in constrained_traces:
            final_idx = np.where(trace['snapshots'] == 77)[0]
            if len(final_idx) > 0:
                if 'BoundSubhalo/TotalMass' in trace:
                    final_mass = trace['BoundSubhalo/TotalMass'][final_idx[0]]
                elif 'SO/200_crit/TotalMass' in trace:
                    final_mass = trace['SO/200_crit/TotalMass'][final_idx[0]]
                else:
                    continue
                if 'BoundSubhalo/CentreOfMass' in trace:
                    final_pos = trace['BoundSubhalo/CentreOfMass'][final_idx[0]]
                elif 'SO/200_crit/CentreOfMass' in trace:
                    final_pos = trace['SO/200_crit/CentreOfMass'][final_idx[0]]
                else:
                    continue

                constrained_final_masses.append(final_mass)
                constrained_final_positions.append(final_pos)
        if len(constrained_final_masses) == 0:
            continue
            
        constrained_final_positions = np.array(constrained_final_positions)
        constrained_mean_position = np.mean(constrained_final_positions, axis=0)
        
        # Vectorized control mass matching
        log_control_masses = np.log10(control_masses)
        matched_control_keys = []
        used_indices = set()
        
        for m in constrained_final_masses:
            log_m = np.log10(m)
            mass_diffs = np.abs(log_control_masses - log_m)
            valid_indices = np.where(mass_diffs <= mass_tolerance_dex)[0]
            valid_indices = [idx for idx in valid_indices if idx not in used_indices]
            if len(valid_indices) == 0:
                continue
            best_idx = int(valid_indices[np.argmin(mass_diffs[valid_indices])])
            matched_control_keys.append(control_keys[best_idx])
            used_indices.add(best_idx)
            all_needed_control_keys.add(control_keys[best_idx])
        
        cluster_match_info[cluster_id] = {
            'constrained_traces': constrained_traces,
            'matched_control_keys': matched_control_keys,
            'constrained_mean_position': constrained_mean_position
        }
    
    # Batch load control traces
    all_control_traces = load_specific_halo_traces(list(all_needed_control_keys),
                                                   config.global_config.output_dir,
                                                   filename=control_filename)
    
    for cluster_id, match_info in cluster_match_info.items():
        matched_control_keys = match_info['matched_control_keys']
        constrained_mean_position = match_info['constrained_mean_position']
        
        recentered_control_traces = []
        for control_key in matched_control_keys:
            if control_key in all_control_traces:
                rec = all_control_traces[control_key].copy()

                if recenter:
                    # Get control final position from index
                    control_idx = control_keys.index(control_key)
                    control_final_pos = control_positions[control_idx]
                    offset = constrained_mean_position - control_final_pos

                    # Apply offset to all snapshots
                    for poskey in ['BoundSubhalo/CentreOfMass', 'SO/200_crit/CentreOfMass']:
                        if poskey in rec:
                            rec[poskey] = rec[poskey] + offset

                recentered_control_traces.append(rec)

        cluster_results[cluster_id] = {
            'constrained_traces': match_info['constrained_traces'],
            'control_traces': recentered_control_traces,
            'matched_final': matched_final
        }
    
    return cluster_results


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
    from scipy.spatial import ConvexHull

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

def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def save_clusters_to_hdf5(stable_haloes, positions, m200_masses, halo_provenance, cluster_labels, config, output_dir, filename="clusters.h5"):
    filepath = os.path.join(output_dir, filename)
    
    with h5py.File(filepath, 'w') as f:
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['mcmc_start'] = config.mode1.mcmc_start
        meta_grp.attrs['mcmc_end'] = config.mode1.mcmc_end
        meta_grp.attrs['eps'] = config.mode1.eps
        meta_grp.attrs['min_samples'] = config.mode1.min_samples
        meta_grp.attrs['m200_mass_cut'] = config.mode1.m200_mass_cut
        meta_grp.attrs['radius_cut'] = config.mode1.radius_cut
        meta_grp.attrs['basedir'] = config.global_config.basedir
        meta_grp.attrs['observer_coords'] = config.global_config.observer_coords
        
        all_data_grp = f.create_group('all_haloes')
        all_data_grp.create_dataset('positions', data=positions)
        all_data_grp.create_dataset('m200_masses', data=m200_masses)
        all_data_grp.create_dataset('cluster_labels', data=cluster_labels)
        
        prov_grp = all_data_grp.create_group('provenance')
        mcmc_ids = np.array([p['mcmc_id'] for p in halo_provenance])
        orig_indices = np.array([p['original_index'] for p in halo_provenance])
        prov_grp.create_dataset('mcmc_ids', data=mcmc_ids)
        prov_grp.create_dataset('original_indices', data=orig_indices)
        
        clusters_grp = f.create_group('clusters')
        for i, cluster in enumerate(stable_haloes):
            cluster_grp = clusters_grp.create_group(f'cluster_{cluster["cluster_id"]}')
            cluster_grp.attrs['cluster_id'] = cluster['cluster_id']
            cluster_grp.attrs['cluster_size'] = cluster['cluster_size']
            cluster_grp.attrs['mean_position'] = cluster['mean_position']
            cluster_grp.attrs['mean_m200_mass'] = cluster['mean_m200_mass']
            cluster_grp.attrs['mean_subhalo_mass'] = cluster.get('mean_subhalo_mass', np.nan)
            cluster_grp.attrs['mean_m500'] = cluster.get('mean_m500', np.nan)
            cluster_grp.attrs['position_std'] = cluster['position_std']
            cluster_grp.attrs['m200_mass_std'] = cluster['m200_mass_std']
            cluster_grp.attrs['subhalo_mass_std'] = cluster.get('subhalo_mass_std', np.nan)
            cluster_grp.attrs['m500_std'] = cluster.get('m500_std', np.nan)
            cluster_grp.attrs['log10_m200_mass_std'] = cluster['log10_m200_mass_std']
            cluster_grp.attrs['log10_m500_std'] = cluster['log10_m500_std']

            member_grp = cluster_grp.create_group('members')
            
            # Save all member data properties
            for key, data in cluster['member_data'].items():
                member_grp.create_dataset(key.replace('/', '_'), data=data)
            
            mcmc_ids = np.array([m['mcmc_id'] for m in cluster['members']])
            orig_indices = np.array([m['original_index'] for m in cluster['members']])
            member_grp.create_dataset('mcmc_ids', data=mcmc_ids)
            member_grp.create_dataset('original_indices', data=orig_indices)

def load_clusters_from_hdf5(output_dir, filename="clusters.h5", minimal=True,
        min_m200_mass=1e14):
    filepath = os.path.join(output_dir, filename)
    
    clusters = []
    metadata = {}
    
    with h5py.File(filepath, 'r') as f:
        meta_grp = f['metadata']
        for key in meta_grp.attrs.keys():
            metadata[key] = meta_grp.attrs[key]
        
        clusters_grp = f['clusters']
        for cluster_name in clusters_grp.keys():
            cluster_grp = clusters_grp[cluster_name]
            
            # Check M200 mass filter first before loading any member data
            mean_m200_mass = float(cluster_grp.attrs['mean_m200_mass'])
            if mean_m200_mass < min_m200_mass:
                continue
            
            member_grp = cluster_grp['members']
            mcmc_ids = member_grp['mcmc_ids'][:]
            orig_indices = member_grp['original_indices'][:]
            
            members = []
            for i in range(len(mcmc_ids)):
                members.append({
                    'mcmc_id': int(mcmc_ids[i]),
                    'original_index': int(orig_indices[i])
                })
            
            # Load all member data properties
            member_data = {}
            for key in member_grp.keys():
                if key not in ['mcmc_ids', 'original_indices']:
                    # Convert back to original property name format
                    original_key = key.replace('_', '/')

                    if minimal and original_key not in ["SO/200_crit/CentreOfMass", "SO/200_crit/TotalMass"]:
                        continue
                    member_data[original_key] = member_grp[key][:]
            
            cluster = {
                'cluster_id': int(cluster_grp.attrs['cluster_id']),
                'cluster_size': int(cluster_grp.attrs['cluster_size']),
                'mean_position': cluster_grp.attrs['mean_position'],
                'mean_m200_mass': mean_m200_mass,
                'mean_subhalo_mass': float(cluster_grp.attrs.get('mean_subhalo_mass', np.nan)),
                'mean_m500': float(cluster_grp.attrs.get('mean_m500', np.nan)),
                'position_std': cluster_grp.attrs['position_std'],
                'm200_mass_std': float(cluster_grp.attrs['m200_mass_std']),
                'log10_m200_mass_std': float(cluster_grp.attrs['log10_m200_mass_std']),
                'log10_m500_std': float(cluster_grp.attrs['log10_m500_std']),
                'subhalo_mass_std': float(cluster_grp.attrs.get('subhalo_mass_std', np.nan)),
                'm500_std': float(cluster_grp.attrs.get('m500_std', np.nan)),
                'members': members,
                'member_data': member_data
            }
            
            clusters.append(cluster)
    
    return clusters, metadata

def save_halo_traces_to_hdf5(halo_traces, config, cluster_metadata, output_dir, filename="halo_traces.h5"):
    filepath = os.path.join(output_dir, filename)
    
    with h5py.File(filepath, 'w') as f:
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['target_snapshot'] = config.mode2.target_snapshot
        meta_grp.attrs['min_cluster_size'] = config.mode2.min_cluster_size
        for key, value in cluster_metadata.items():
            if isinstance(value, (int, float, str)):
                meta_grp.attrs[key] = value
            elif isinstance(value, (list, np.ndarray)):
                meta_grp.attrs[key] = np.array(value)
        
        # Create index for fast lookups
        index_grp = f.create_group('index')
        
        final_m200_masses = []
        final_positions = []
        cluster_ids = []
        halo_keys = []
        
        for halo_key, trace_data in halo_traces.items():
            # Find final snapshot (77) data
            final_idx = np.where(trace_data['snapshots'] == 77)[0]
            if len(final_idx) > 0:
                if 'BoundSubhalo/TotalMass' in trace_data:
                    final_mass = trace_data['BoundSubhalo/TotalMass'][final_idx[0]]
                elif 'SO/200_crit/TotalMass' in trace_data:
                    final_mass = trace_data['SO/200_crit/TotalMass'][final_idx[0]]
                else:
                    continue
                
                if 'BoundSubhalo/CentreOfMass' in trace_data:
                    final_pos = trace_data['BoundSubhalo/CentreOfMass'][final_idx[0]]
                elif 'SO/200_crit/CentreOfMass' in trace_data:
                    final_pos = trace_data['SO/200_crit/CentreOfMass'][final_idx[0]]
                else:
                    continue
                
                final_m200_masses.append(final_mass)
                final_positions.append(final_pos)
                cluster_ids.append(trace_data['cluster_id'])
                halo_keys.append(halo_key)
        
        # Save index arrays
        if len(final_m200_masses) > 0:
            index_grp.create_dataset('final_m200_masses', data=np.array(final_m200_masses))
            index_grp.create_dataset('final_positions', data=np.array(final_positions))
            index_grp.create_dataset('cluster_ids', data=np.array(cluster_ids))
            
            # String arrays need special handling in HDF5
            dt = h5py.special_dtype(vlen=str)
            index_grp.create_dataset('halo_keys', data=np.array(halo_keys, dtype=object), dtype=dt)
        
        traces_grp = f.create_group('halo_traces')
        for halo_key, trace_data in halo_traces.items():
            trace_grp = traces_grp.create_group(halo_key)
            trace_grp.attrs['mcmc_id'] = trace_data['mcmc_id']
            trace_grp.attrs['original_index'] = trace_data['original_index']
            trace_grp.attrs['cluster_id'] = trace_data['cluster_id']
            trace_grp.create_dataset('snapshots', data=trace_data['snapshots'])
            
            # Save all traced properties
            for key, data in trace_data.items():
                if key not in ['mcmc_id', 'original_index', 'cluster_id', 'snapshots']:
                    dataset_name = key.replace('/', '_')
                    trace_grp.create_dataset(dataset_name, data=data)

def load_halo_traces_index(output_dir, filename="halo_traces.h5"):
    """Load only the index for fast halo matching"""
    filepath = os.path.join(output_dir, filename)
    
    try:
        with h5py.File(filepath, 'r') as f:
            if 'index' not in f:
                return None
            
            index_grp = f['index']
            index_data = {
                'final_m200_masses': index_grp['final_m200_masses'][:],
                'final_positions': index_grp['final_positions'][:],
                'cluster_ids': index_grp['cluster_ids'][:],
                'halo_keys': [key.decode() if isinstance(key, bytes) else key for key in index_grp['halo_keys'][:]]
            }
            
            return index_data
    except Exception as e:
        print(f"Error loading index from {filepath}: {e}")
        return None

def load_specific_halo_traces(halo_keys, output_dir, filename="halo_traces.h5"):
    """Load traces for specific halo keys only"""
    filepath = os.path.join(output_dir, filename)
    
    halo_traces = {}
    
    try:
        with h5py.File(filepath, 'r') as f:
            traces_grp = f['halo_traces']
            
            for halo_key in halo_keys:
                if halo_key in traces_grp:
                    trace_grp = traces_grp[halo_key]
                    halo_traces[halo_key] = {
                        'mcmc_id': int(trace_grp.attrs['mcmc_id']),
                        'original_index': int(trace_grp.attrs['original_index']),
                        'cluster_id': int(trace_grp.attrs['cluster_id']),
                        'snapshots': trace_grp['snapshots'][:]
                    }
                    
                    # Load all traced properties
                    for key in trace_grp.keys():
                        if key != 'snapshots':
                            # Convert back to original property name format
                            original_key = key.replace('_', '/')
                            halo_traces[halo_key][original_key] = trace_grp[key][:]
        
        return halo_traces
    except Exception as e:
        print(f"Error loading specific traces from {filepath}: {e}")
        return {}

def load_halo_traces_from_hdf5(output_dir, filename="halo_traces.h5"):
    filepath = os.path.join(output_dir, filename)
    
    halo_traces = {}
    metadata = {}
    
    with h5py.File(filepath, 'r') as f:
        meta_grp = f['metadata']
        for key in meta_grp.attrs.keys():
            metadata[key] = meta_grp.attrs[key]
        
        traces_grp = f['halo_traces']
        for halo_key in traces_grp.keys():
            trace_grp = traces_grp[halo_key]
            halo_traces[halo_key] = {
                'mcmc_id': int(trace_grp.attrs['mcmc_id']),
                'original_index': int(trace_grp.attrs['original_index']),
                'cluster_id': int(trace_grp.attrs['cluster_id']),
                'snapshots': trace_grp['snapshots'][:]
            }
            
            # Load all traced properties
            for key in trace_grp.keys():
                if key != 'snapshots':
                    # Convert back to original property name format
                    original_key = key.replace('_', '/')
                    halo_traces[halo_key][original_key] = trace_grp[key][:]
    
    return halo_traces, metadata

def get_cluster_trace_info(output_dir, filename="halo_traces.h5"):
    """Get cluster trace information without loading full data"""
    filepath = os.path.join(output_dir, filename)
    
    try:
        with h5py.File(filepath, 'r') as f:
            cluster_trace_counts = defaultdict(int)
            traces_grp = f['halo_traces']
            
            for halo_key in traces_grp.keys():
                trace_grp = traces_grp[halo_key]
                cluster_id = int(trace_grp.attrs['cluster_id'])
                cluster_trace_counts[cluster_id] += 1
        
        return dict(cluster_trace_counts)
    except FileNotFoundError:
        return {}

def load_single_cluster_traces(cluster_id, output_dir, filename="halo_traces.h5"):
    """Load traces for a single cluster"""
    filepath = os.path.join(output_dir, filename)
    
    try:
        with h5py.File(filepath, 'r') as f:
            cluster_traces = []
            traces_grp = f['halo_traces']
            
            for halo_key in traces_grp.keys():
                trace_grp = traces_grp[halo_key]
                if int(trace_grp.attrs['cluster_id']) == cluster_id:
                    trace_data = {
                        'mcmc_id': int(trace_grp.attrs['mcmc_id']),
                        'original_index': int(trace_grp.attrs['original_index']),
                        'cluster_id': int(trace_grp.attrs['cluster_id']),
                        'snapshots': trace_grp['snapshots'][:]
                    }
                    
                    # Load all traced properties
                    for key in trace_grp.keys():
                        if key != 'snapshots':
                            # Convert back to original property name format
                            original_key = key.replace('_', '/')
                            trace_data[original_key] = trace_grp[key][:]
                    
                    cluster_traces.append(trace_data)
        
        return cluster_traces if cluster_traces else None
    except FileNotFoundError:
        return None

def find_clusters_in_window(output_dir, filename, center_position, window_size):
   """
   Find clusters whose centroids are within a given window without loading full data.

   Parameters:
   -----------
   output_dir : str
       Directory containing the HDF5 file
   filename : str
       HDF5 filename
   center_position : array-like
       [x, y, z] center position
   window_size : float
       Radius of window in Mpc

   Returns:
   --------
   list : Cluster IDs within the window
   """
   filepath = os.path.join(output_dir, filename)
   window_cluster_ids = []

   try:
       with h5py.File(filepath, 'r') as f:
           clusters_grp = f['clusters']

           for cluster_name in clusters_grp.keys():
               cluster_grp = clusters_grp[cluster_name]
               centroid = cluster_grp.attrs['mean_position']

               # Check if centroid is within window
               distance = np.linalg.norm(centroid - center_position)
               if distance <= window_size:
                   cluster_id = int(cluster_grp.attrs['cluster_id'])
                   window_cluster_ids.append(cluster_id)

   except FileNotFoundError:
       return []
   except Exception as e:
       print(f"Error reading {filepath}: {e}")
       return []

   return window_cluster_ids

def load_single_cluster_members(output_dir, filename, cluster_id):
   """
   Load member data for a single cluster by cluster ID.

   Parameters:
   -----------
   output_dir : str
       Directory containing the HDF5 file
   filename : str
       HDF5 filename
   cluster_id : int
       Cluster ID to load

   Returns:
   --------
   dict : Cluster data including member_data, or None if not found
   """
   filepath = os.path.join(output_dir, filename)

   try:
       with h5py.File(filepath, 'r') as f:
           clusters_grp = f['clusters']
           cluster_grp_name = f'cluster_{cluster_id}'

           if cluster_grp_name not in clusters_grp:
               return None

           cluster_grp = clusters_grp[cluster_grp_name]
           member_grp = cluster_grp['members']

           # Load cluster metadata
           cluster_data = {
               'cluster_id': int(cluster_grp.attrs['cluster_id']),
               'cluster_size': int(cluster_grp.attrs['cluster_size']),
               'mean_position': cluster_grp.attrs['mean_position'],
               'mean_m200_mass': float(cluster_grp.attrs['mean_m200_mass']),
               'mean_subhalo_mass': float(cluster_grp.attrs.get('mean_subhalo_mass', np.nan)),
               'mean_m500': float(cluster_grp.attrs.get('mean_m500', np.nan)),
               'position_std': cluster_grp.attrs['position_std'],
               'm200_mass_std': float(cluster_grp.attrs['m200_mass_std']),
               'subhalo_mass_std': float(cluster_grp.attrs.get('subhalo_mass_std', np.nan)),
               'm500_std': float(cluster_grp.attrs.get('m500_std', np.nan)),
               'log10_m200_std': float(cluster_grp.attrs.get('log10_m200_std', np.nan)),
               'log10_m500_std': float(cluster_grp.attrs.get('log10_m500_std', np.nan)),
           }

           # Load member data
           member_data = {}
           for key in member_grp.keys():
               if key not in ['mcmc_ids', 'original_indices']:
                   original_key = key.replace('_', '/')
                   member_data[original_key] = member_grp[key][:]

           cluster_data['member_data'] = member_data

           return cluster_data

   except FileNotFoundError:
       return None
   except Exception as e:
       print(f"Error loading cluster {cluster_id} from {filepath}: {e}")
       return None

def find_control_matches_and_recenter_single(cluster_id,
                                             config,
                                             trace_filename,
                                             mass_tolerance_dex=0.1,
                                             use_matched_final=False,
                                             recenter_controls=True):
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
        If True, apply matched-final affine transformation (only if recenter_controls=True).
    recenter_controls : bool
        If False, return controls in their *true* original coordinates without recentering.
    """
    # Load control index
    control_filename = f"control_halo_traces_mass_{config.mode4.m200_mass_cut:.1e}_radius_{config.mode4.radius_cut}.h5"
    control_index = load_halo_traces_index(config.global_config.output_dir, filename=control_filename)
    if control_index is None:
        print(f"[warn] Could not load control index from {control_filename}")
        return None, None

    # Load constrained traces for this cluster
    constrained_traces = load_single_cluster_traces(cluster_id, config.global_config.output_dir, filename=trace_filename)
    if not constrained_traces:
        print(f"[warn] No constrained traces for cluster {cluster_id}")
        return None, None

    # Use batch API on a dict with a single entry
    constrained_traces_dict = {cluster_id: constrained_traces}
    results = find_control_matches_batch(constrained_traces_dict,
                                         control_index,
                                         control_filename,
                                         config,
                                         mass_tolerance_dex=mass_tolerance_dex,
                                         matched_final=use_matched_final if recenter_controls else False,
                                         recenter=recenter_controls)

    if cluster_id not in results:
        print(f"[warn] No control matches returned for cluster {cluster_id}")
        return None, None

    return results[cluster_id]['constrained_traces'], results[cluster_id]['control_traces']

