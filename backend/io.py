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
    'save_raw_dbscan_to_hdf5',
    'load_raw_dbscan_from_hdf5',
    'save_hdbscan_clusters_to_hdf5',
    'load_hdbscan_clusters_from_hdf5'
]

def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def save_clusters_to_hdf5(stable_haloes, positions, m200_masses, halo_provenance, cluster_labels, config, output_dir, filename="clusters.h5"):
    filepath = os.path.join(output_dir, filename)
    
    # Apply filtering criteria
    m200_cut = config.mode1.m200_mass_cut
    size_cut = config.mode1.min_association_size
    
    filtered_haloes = [
        cluster for cluster in stable_haloes 
        if cluster['mean_m200_mass'] >= m200_cut and cluster['cluster_size'] >= size_cut
    ]
    
    print(f"\nFiltering associations:")
    print(f"  Total associations found: {len(stable_haloes)}")
    print(f"  M200 cut: {m200_cut:.2e} Msol")
    print(f"  Size cut: {size_cut} members")
    print(f"  Associations passing cuts: {len(filtered_haloes)}")
    
    if len(filtered_haloes) == 0:
        print("Warning: No associations pass the filtering criteria!")
        # Still create the file with empty groups
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
            f.create_group('summary')
            f.create_group('posterior')
        return
    
    observer = np.array(config.global_config.observer_coords)
    
    with h5py.File(filepath, 'w') as f:
        # Metadata group
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['mcmc_start'] = config.mode1.mcmc_start
        meta_grp.attrs['mcmc_end'] = config.mode1.mcmc_end
        meta_grp.attrs['eps'] = config.mode1.eps
        meta_grp.attrs['min_samples'] = config.mode1.min_samples
        meta_grp.attrs['m200_mass_cut'] = config.mode1.m200_mass_cut
        meta_grp.attrs['radius_cut'] = config.mode1.radius_cut
        meta_grp.attrs['basedir'] = config.global_config.basedir
        meta_grp.attrs['observer_coords'] = config.global_config.observer_coords
        
        # Summary group
        summary_grp = f.create_group('summary')
        n_clusters = len(filtered_haloes)
        
        # Initialize summary data structures
        summary_data = defaultdict(list)
        cluster_sizes = []
        association_ids = []
        
        for cluster in filtered_haloes:
            association_ids.append(cluster['cluster_id'])
            cluster_sizes.append(cluster['cluster_size'])
            
            member_data = cluster['member_data']
            
            # Process all scalar properties
            for key, data in member_data.items():
                # Skip identifiers and non-numeric data
                if key in ['mcmc/id', 'halo/original/index']:
                    continue
                
                # Handle scalar properties (masses, concentrations, ra, dec, dist, etc.)
                if isinstance(data, np.ndarray) and data.ndim == 1:
                    # Filter out NaN values for statistics
                    valid_data = data[~np.isnan(data)]
                    dataset_name = key.replace('/', '_')
                    if len(valid_data) > 0:
                        summary_data[dataset_name].append([
                            np.median(valid_data),
                            np.percentile(valid_data, 25),
                            np.percentile(valid_data, 75)
                        ])
                    else:
                        summary_data[dataset_name].append([np.nan, np.nan, np.nan])
                
                # Handle 3D vector properties (positions, velocities) - compute magnitudes
                elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 3:
                    magnitudes = np.linalg.norm(data, axis=1)
                    valid_mags = magnitudes[~np.isnan(magnitudes)]
                    dataset_name = key.replace('/', '_') + '_magnitude'
                    if len(valid_mags) > 0:
                        summary_data[dataset_name].append([
                            np.median(valid_mags),
                            np.percentile(valid_mags, 25),
                            np.percentile(valid_mags, 75)
                        ])
                    else:
                        summary_data[dataset_name].append([np.nan, np.nan, np.nan])
        
        # Save summary datasets
        summary_grp.create_dataset('association_id', data=np.array(association_ids))
        summary_grp.create_dataset('cluster_size', data=np.array(cluster_sizes))
        
        for key, values in summary_data.items():
            summary_grp.create_dataset(key, data=np.array(values))
        
        # Posterior group (formerly clusters)
        posterior_grp = f.create_group('posterior')
        
        for cluster in filtered_haloes:
            assoc_grp = posterior_grp.create_group(f'association_{cluster["cluster_id"]}')
            assoc_grp.attrs['association_id'] = cluster['cluster_id']
            assoc_grp.attrs['cluster_size'] = cluster['cluster_size']
            assoc_grp.attrs['mean_position'] = cluster['mean_position']
            assoc_grp.attrs['mean_m200_mass'] = cluster['mean_m200_mass']
            assoc_grp.attrs['mean_subhalo_mass'] = cluster.get('mean_subhalo_mass', np.nan)
            assoc_grp.attrs['mean_m500'] = cluster.get('mean_m500', np.nan)
            assoc_grp.attrs['position_std'] = cluster['position_std']
            assoc_grp.attrs['m200_mass_std'] = cluster['m200_mass_std']
            assoc_grp.attrs['subhalo_mass_std'] = cluster.get('subhalo_mass_std', np.nan)
            assoc_grp.attrs['m500_std'] = cluster.get('m500_std', np.nan)
            assoc_grp.attrs['log10_m200_mass_std'] = cluster['log10_m200_mass_std']
            assoc_grp.attrs['log10_m500_std'] = cluster['log10_m500_std']
            assoc_grp.attrs['axis_ratio_ba'] = cluster.get('axis_ratio_ba', np.nan)
            assoc_grp.attrs['axis_ratio_ca'] = cluster.get('axis_ratio_ca', np.nan)
            assoc_grp.attrs['asphericity'] = cluster.get('asphericity', np.nan)
            assoc_grp.attrs['prolateness'] = cluster.get('prolateness', np.nan)
            
            # Save all member data properties directly in the association group
            for key, data in cluster['member_data'].items():
                assoc_grp.create_dataset(key.replace('/', '_'), data=data)

def _load_clusters_from_flat_format_for_mode2(f, min_m200_mass, minimal):
    """Load clusters from new v2 flat HDBSCAN format and reconstruct legacy structure for Mode 2.

    The new format stores data in flat arrays with CSR-style indexing for members.
    This function reconstructs the nested structure expected by Mode 2's extract_target_haloes().
    """
    clusters = []
    summary = f['summary']

    # Check if summary has any clusters
    if 'cluster_id' not in summary:
        return []

    # Read summary arrays
    cluster_ids = summary['cluster_id'][:]
    n_members_arr = summary['n_members'][:]
    center_xyz = summary['center_xyz'][:]
    mean_m200_masses = summary['mean_m200_mass'][:]

    # Read optional summary fields with defaults
    position_stds = summary['position_std'][:] if 'position_std' in summary else np.full((len(cluster_ids), 3), np.nan)
    m200_mass_stds = summary['m200_mass_std'][:] if 'm200_mass_std' in summary else np.full(len(cluster_ids), np.nan)
    log10_m200_mass_stds = summary['log10_m200_mass_std'][:] if 'log10_m200_mass_std' in summary else np.full(len(cluster_ids), np.nan)
    log10_m500_stds = summary['log10_m500_std'][:] if 'log10_m500_std' in summary else np.full(len(cluster_ids), np.nan)
    mean_subhalo_masses = summary['mean_subhalo_mass'][:] if 'mean_subhalo_mass' in summary else np.full(len(cluster_ids), np.nan)
    subhalo_mass_stds = summary['subhalo_mass_std'][:] if 'subhalo_mass_std' in summary else np.full(len(cluster_ids), np.nan)
    mean_m500s = summary['mean_m500'][:] if 'mean_m500' in summary else np.full(len(cluster_ids), np.nan)
    m500_stds = summary['m500_std'][:] if 'm500_std' in summary else np.full(len(cluster_ids), np.nan)
    axis_ratio_bas = summary['axis_ratio_ba'][:] if 'axis_ratio_ba' in summary else np.full(len(cluster_ids), np.nan)
    axis_ratio_cas = summary['axis_ratio_ca'][:] if 'axis_ratio_ca' in summary else np.full(len(cluster_ids), np.nan)
    asphericities = summary['asphericity'][:] if 'asphericity' in summary else np.full(len(cluster_ids), np.nan)
    prolatenesses = summary['prolateness'][:] if 'prolateness' in summary else np.full(len(cluster_ids), np.nan)

    # Read member index (CSR-style)
    member_index = f['member_index']
    offsets = member_index['offsets'][:]

    # Read all member data arrays
    member_data_grp = f['member_data']
    member_arrays = {}
    for key in member_data_grp.keys():
        member_arrays[key] = member_data_grp[key][:]

    # Build clusters
    for i in range(len(cluster_ids)):
        # Apply mass filter
        if mean_m200_masses[i] < min_m200_mass:
            continue

        start = offsets[i]
        end = offsets[i + 1]

        # Reconstruct member_data dict with slash-separated keys
        member_data = {}
        for underscore_key, data in member_arrays.items():
            # Convert underscore to slash for property names
            slash_key = underscore_key.replace('_', '/')

            # Handle special key mappings for Mode 2 compatibility
            if underscore_key == 'realization_ids':
                slash_key = 'mcmc/id'
            elif underscore_key == 'mcmc_id':
                slash_key = 'mcmc/id'
            elif underscore_key == 'halo_original_index':
                slash_key = 'halo/original/index'

            # Apply minimal filter
            if minimal and slash_key not in ["SO/200/crit/CentreOfMass", "SO/200/crit/TotalMass",
                                              "mcmc/id", "halo/original/index", "SOAP/ProgenitorIndex"]:
                continue

            member_data[slash_key] = data[start:end]

        # Reconstruct members list for backwards compatibility with Mode 2
        members = []
        mcmc_ids = member_data.get('mcmc/id', None)
        orig_indices = member_data.get('halo/original/index', None)

        if mcmc_ids is not None and orig_indices is not None:
            for j in range(len(mcmc_ids)):
                members.append({
                    'mcmc_id': int(mcmc_ids[j]),
                    'original_index': int(orig_indices[j])
                })

        # Helper to convert -1 sentinel back to NaN
        def to_nan(val):
            if isinstance(val, (int, float, np.integer, np.floating)) and val < 0:
                return np.nan
            return float(val)

        cluster = {
            'cluster_id': int(cluster_ids[i]),
            'cluster_size': int(n_members_arr[i]),  # Alias for n_members for Mode 2 compatibility
            'mean_position': center_xyz[i],
            'mean_m200_mass': float(mean_m200_masses[i]),
            'mean_subhalo_mass': to_nan(mean_subhalo_masses[i]),
            'mean_m500': to_nan(mean_m500s[i]),
            'position_std': position_stds[i],
            'm200_mass_std': to_nan(m200_mass_stds[i]),
            'subhalo_mass_std': to_nan(subhalo_mass_stds[i]),
            'm500_std': to_nan(m500_stds[i]),
            'log10_m200_mass_std': to_nan(log10_m200_mass_stds[i]),
            'log10_m500_std': to_nan(log10_m500_stds[i]),
            'axis_ratio_ba': to_nan(axis_ratio_bas[i]),
            'axis_ratio_ca': to_nan(axis_ratio_cas[i]),
            'asphericity': to_nan(asphericities[i]),
            'prolateness': to_nan(prolatenesses[i]),
            'members': members,
            'member_data': member_data
        }

        clusters.append(cluster)

    return clusters


def load_clusters_from_hdf5(output_dir, filename="clusters.h5", minimal=True,
        min_m200_mass=1e14):
    filepath = os.path.join(output_dir, filename)

    clusters = []
    metadata = {}

    with h5py.File(filepath, 'r') as f:
        meta_grp = f['metadata']
        for key in meta_grp.attrs.keys():
            metadata[key] = meta_grp.attrs[key]

        # Check for new flat format (v2)
        format_version = metadata.get('hdf5_format_version', 1)

        if format_version >= 2 and 'member_index' in f:
            # New flat format - reconstruct legacy structure for Mode 2
            clusters = _load_clusters_from_flat_format_for_mode2(f, min_m200_mass, minimal)
            return clusters, metadata

        # Handle both old 'clusters' and new 'posterior' group names (legacy format)
        if 'posterior' in f:
            posterior_grp = f['posterior']
            group_prefix = 'association_'
        elif 'clusters' in f:
            posterior_grp = f['clusters']
            group_prefix = 'cluster_'
        else:
            return [], metadata
        
        for assoc_name in posterior_grp.keys():
            assoc_grp = posterior_grp[assoc_name]
            
            # Check M200 mass filter first before loading any member data
            mean_m200_mass = float(assoc_grp.attrs['mean_m200_mass'])
            if mean_m200_mass < min_m200_mass:
                continue
            
            # Load all member data properties
            member_data = {}

            # Check if member data is in a 'members' subgroup (new format) or directly in assoc_grp (old format)
            if 'members' in assoc_grp:
                # New format: member data is nested in 'members' group
                members_grp = assoc_grp['members']

                def extract_datasets(group, prefix=''):
                    """Recursively extract datasets from nested groups."""
                    for key in group.keys():
                        full_key = f"{prefix}/{key}" if prefix else key
                        item = group[key]

                        if isinstance(item, h5py.Dataset):
                            # This is a dataset, extract it
                            if minimal and full_key not in ["SO/200/crit/CentreOfMass", "SO/200/crit/TotalMass",
                                                            "mcmc/id", "halo/original/index"]:
                                continue
                            member_data[full_key] = item[:]
                        elif isinstance(item, h5py.Group):
                            # This is a group, recurse into it
                            extract_datasets(item, full_key)

                extract_datasets(members_grp)
            else:
                # Old format: datasets directly under association group
                for key in assoc_grp.keys():
                    # Skip merger_tree group if present
                    if key == 'merger_tree':
                        continue

                    # Convert back to original property name format
                    original_key = key.replace('_', '/')

                    if minimal and original_key not in ["SO/200/crit/CentreOfMass", "SO/200/crit/TotalMass",
                                                         "mcmc/id", "halo/original/index"]:
                        continue
                    member_data[original_key] = assoc_grp[key][:]
            
            # Reconstruct members list from member_data for backwards compatibility
            members = []
            if 'mcmc/id' in member_data and 'halo/original/index' in member_data:
                mcmc_ids = member_data['mcmc/id']
                orig_indices = member_data['halo/original/index']
                for i in range(len(mcmc_ids)):
                    members.append({
                        'mcmc_id': int(mcmc_ids[i]),
                        'original_index': int(orig_indices[i])
                    })
            
            # Handle both old and new attribute names
            cluster_id_key = 'association_id' if 'association_id' in assoc_grp.attrs else 'cluster_id'
            
            cluster = {
                'cluster_id': int(assoc_grp.attrs[cluster_id_key]),
                'cluster_size': int(assoc_grp.attrs['cluster_size']),
                'mean_position': assoc_grp.attrs['mean_position'],
                'mean_m200_mass': mean_m200_mass,
                'mean_subhalo_mass': float(assoc_grp.attrs.get('mean_subhalo_mass', np.nan)),
                'mean_m500': float(assoc_grp.attrs.get('mean_m500', np.nan)),
                'position_std': assoc_grp.attrs['position_std'],
                'm200_mass_std': float(assoc_grp.attrs['m200_mass_std']),
                'log10_m200_mass_std': float(assoc_grp.attrs['log10_m200_mass_std']),
                'log10_m500_std': float(assoc_grp.attrs['log10_m500_std']),
                'subhalo_mass_std': float(assoc_grp.attrs.get('subhalo_mass_std', np.nan)),
                'm500_std': float(assoc_grp.attrs.get('m500_std', np.nan)),
                'axis_ratio_ba': float(assoc_grp.attrs.get('axis_ratio_ba', np.nan)),
                'axis_ratio_ca': float(assoc_grp.attrs.get('axis_ratio_ca', np.nan)),
                'asphericity': float(assoc_grp.attrs.get('asphericity', np.nan)),
                'prolateness': float(assoc_grp.attrs.get('prolateness', np.nan)),
                'members': members,
                'member_data': member_data
            }
            
            clusters.append(cluster)
    
    return clusters, metadata

def save_halo_traces_to_hdf5(halo_traces, config, cluster_metadata, output_dir, filename="halo_traces.h5"):
    filepath = os.path.join(output_dir, filename)
    
    def _extract_positions_at_or_after_snapshot(trace_data, key, target_snapshot):
        snapshots = trace_data['snapshots']
        positions = trace_data.get(key, None)
        if positions is None:
            return None
        valid = snapshots[snapshots >= target_snapshot]
        if len(valid) == 0:
            return None
        earliest = int(np.min(valid))
        idx = np.where(snapshots == earliest)[0][0]
        return positions[idx]
    
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
        initial_positions_10 = []
        distance_traveled_10 = []
        cluster_ids = []
        halo_keys = []
        
        final_snapshot = config.global_config.final_snapshot
        for halo_key, trace_data in halo_traces.items():
            # Find final snapshot data
            final_idx = np.where(trace_data['snapshots'] == final_snapshot)[0]
            if len(final_idx) > 0:
                fi = final_idx[0]
                
                # Get final mass and position
                if 'BoundSubhalo/TotalMass' in trace_data:
                    final_mass = trace_data['BoundSubhalo/TotalMass'][fi]
                    poskey = 'BoundSubhalo/CentreOfMass'
                elif 'SO/200_crit/TotalMass' in trace_data:
                    final_mass = trace_data['SO/200_crit/TotalMass'][fi]
                    poskey = 'SO/200_crit/CentreOfMass'
                else:
                    continue
                
                if poskey not in trace_data:
                    continue
                    
                final_pos = trace_data[poskey][fi]
                
                # Get initial position at snapshot 10
                initial_pos = _extract_positions_at_or_after_snapshot(trace_data, poskey, 10)
                if initial_pos is None:
                    continue
                
                # Calculate distance traveled
                distance = float(np.linalg.norm(final_pos - initial_pos))
                
                final_m200_masses.append(final_mass)
                final_positions.append(final_pos)
                initial_positions_10.append(initial_pos)
                distance_traveled_10.append(distance)
                cluster_ids.append(trace_data['cluster_id'])
                halo_keys.append(halo_key)
        
        # Save index arrays
        if len(final_m200_masses) > 0:
            index_grp.create_dataset('final_m200_masses', data=np.array(final_m200_masses))
            index_grp.create_dataset('final_positions', data=np.array(final_positions))
            index_grp.create_dataset('initial_positions_10', data=np.array(initial_positions_10))
            index_grp.create_dataset('distance_traveled_10', data=np.array(distance_traveled_10))
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
            
            # Load new fields if they exist
            if 'initial_positions_10' in index_grp:
                index_data['initial_positions_10'] = index_grp['initial_positions_10'][:]
            if 'distance_traveled_10' in index_grp:
                index_data['distance_traveled_10'] = index_grp['distance_traveled_10'][:]
            
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
           # Handle both old 'clusters' and new 'posterior' group names
           if 'posterior' in f:
               posterior_grp = f['posterior']
           elif 'clusters' in f:
               posterior_grp = f['clusters']
           else:
               return []

           for assoc_name in posterior_grp.keys():
               assoc_grp = posterior_grp[assoc_name]
               centroid = assoc_grp.attrs['mean_position']

               # Check if centroid is within window
               distance = np.linalg.norm(centroid - center_position)
               if distance <= window_size:
                   # Handle both old and new attribute names
                   if 'association_id' in assoc_grp.attrs:
                       cluster_id = int(assoc_grp.attrs['association_id'])
                   else:
                       cluster_id = int(assoc_grp.attrs['cluster_id'])
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
           # Handle both old 'clusters' and new 'posterior' group names
           if 'posterior' in f:
               posterior_grp = f['posterior']
               assoc_grp_name = f'association_{cluster_id}'
           elif 'clusters' in f:
               posterior_grp = f['clusters']
               assoc_grp_name = f'cluster_{cluster_id}'
           else:
               return None

           if assoc_grp_name not in posterior_grp:
               return None

           assoc_grp = posterior_grp[assoc_grp_name]

           # Handle both old and new attribute names
           cluster_id_key = 'association_id' if 'association_id' in assoc_grp.attrs else 'cluster_id'

           # Load cluster metadata
           cluster_data = {
               'cluster_id': int(assoc_grp.attrs[cluster_id_key]),
               'cluster_size': int(assoc_grp.attrs['cluster_size']),
               'mean_position': assoc_grp.attrs['mean_position'],
               'mean_m200_mass': float(assoc_grp.attrs['mean_m200_mass']),
               'mean_subhalo_mass': float(assoc_grp.attrs.get('mean_subhalo_mass', np.nan)),
               'mean_m500': float(assoc_grp.attrs.get('mean_m500', np.nan)),
               'position_std': assoc_grp.attrs['position_std'],
               'm200_mass_std': float(assoc_grp.attrs['m200_mass_std']),
               'subhalo_mass_std': float(assoc_grp.attrs.get('subhalo_mass_std', np.nan)),
               'm500_std': float(assoc_grp.attrs.get('m500_std', np.nan)),
               'log10_m200_std': float(assoc_grp.attrs.get('log10_m200_std', np.nan)),
               'log10_m500_std': float(assoc_grp.attrs.get('log10_m500_std', np.nan)),
               'axis_ratio_ba': float(assoc_grp.attrs.get('axis_ratio_ba', np.nan)),
               'axis_ratio_ca': float(assoc_grp.attrs.get('axis_ratio_ca', np.nan)),
               'asphericity': float(assoc_grp.attrs.get('asphericity', np.nan)),
               'prolateness': float(assoc_grp.attrs.get('prolateness', np.nan)),
           }

           # Load member data
           member_data = {}
           for key in assoc_grp.keys():
               original_key = key.replace('_', '/')
               member_data[original_key] = assoc_grp[key][:]

           cluster_data['member_data'] = member_data

           return cluster_data

   except FileNotFoundError:
       return None
   except Exception as e:
       print(f"Error loading cluster {cluster_id} from {filepath}: {e}")
       return None

def save_raw_dbscan_to_hdf5(cluster_labels, positions, m200_masses, halo_provenance,
                             combined_data, config, output_dir, filename="raw_clusters.h5"):
    """
    Save raw DBSCAN output (before MCMC constraint and mass filtering) to HDF5.

    Parameters:
    -----------
    cluster_labels : array
        Raw cluster labels from DBSCAN (-1 for noise)
    positions : array
        Halo positions (N x 3)
    m200_masses : array
        Halo M200 masses (N,)
    halo_provenance : list of dict
        List of {'mcmc_id': int, 'original_index': int} for each halo
    combined_data : dict
        Dictionary of all halo properties
    config : Config
        Configuration object with mode1 settings (or mode3 for control sims)
    output_dir : str
        Output directory path
    filename : str
        Output filename
    """
    filepath = os.path.join(output_dir, filename)

    with h5py.File(filepath, 'w') as f:
        # Metadata group
        meta_grp = f.create_group('metadata')
        # Store config parameters - try mode1 first, fall back to mode3 for control sims
        mode_config = getattr(config, 'mode1', None) or getattr(config, 'mode3', None)
        if mode_config:
            meta_grp.attrs['mcmc_start'] = mode_config.mcmc_start
            meta_grp.attrs['mcmc_end'] = mode_config.mcmc_end
            if hasattr(mode_config, 'eps'):
                meta_grp.attrs['eps'] = mode_config.eps
            meta_grp.attrs['min_samples'] = mode_config.min_samples
            meta_grp.attrs['m200_mass_cut'] = mode_config.m200_mass_cut
            meta_grp.attrs['radius_cut'] = mode_config.radius_cut
        meta_grp.attrs['basedir'] = config.global_config.basedir
        meta_grp.attrs['observer_coords'] = config.global_config.observer_coords
        meta_grp.attrs['boxsize'] = config.global_config.boxsize
        meta_grp.attrs['final_snapshot'] = config.global_config.final_snapshot

        # Raw DBSCAN results
        dbscan_grp = f.create_group('dbscan')
        dbscan_grp.create_dataset('cluster_labels', data=cluster_labels)
        dbscan_grp.create_dataset('positions', data=positions)
        dbscan_grp.create_dataset('m200_masses', data=m200_masses)

        # Halo provenance
        mcmc_ids = np.array([p['mcmc_id'] for p in halo_provenance])
        original_indices = np.array([p['original_index'] for p in halo_provenance])
        dbscan_grp.create_dataset('mcmc_ids', data=mcmc_ids)
        dbscan_grp.create_dataset('original_indices', data=original_indices)

        # Combined data (all halo properties)
        combined_grp = f.create_group('combined_data')
        for key, data in combined_data.items():
            dataset_name = key.replace('/', '_')
            combined_grp.create_dataset(dataset_name, data=data)

        # Summary statistics
        n_total = len(cluster_labels)
        n_noise = np.sum(cluster_labels == -1)
        n_clustered = n_total - n_noise
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))

        meta_grp.attrs['n_total_halos'] = n_total
        meta_grp.attrs['n_noise'] = n_noise
        meta_grp.attrs['n_clustered'] = n_clustered
        meta_grp.attrs['n_clusters'] = n_clusters

    print(f"\nRaw DBSCAN results saved to {filepath}")
    print(f"  Total halos: {n_total}")
    print(f"  Clustered halos: {n_clustered}")
    print(f"  Noise halos: {n_noise}")
    print(f"  Number of clusters: {n_clusters}")

def load_raw_dbscan_from_hdf5(output_dir, filename="raw_clusters.h5"):
    """
    Load raw DBSCAN output from HDF5.

    Returns:
    --------
    tuple : (cluster_labels, positions, m200_masses, halo_provenance, combined_data, metadata)
    """
    filepath = os.path.join(output_dir, filename)

    with h5py.File(filepath, 'r') as f:
        # Load metadata
        metadata = {}
        meta_grp = f['metadata']
        for key in meta_grp.attrs.keys():
            metadata[key] = meta_grp.attrs[key]

        # Load DBSCAN results
        dbscan_grp = f['dbscan']
        cluster_labels = dbscan_grp['cluster_labels'][:]
        positions = dbscan_grp['positions'][:]
        m200_masses = dbscan_grp['m200_masses'][:]
        mcmc_ids = dbscan_grp['mcmc_ids'][:]
        original_indices = dbscan_grp['original_indices'][:]

        # Reconstruct halo_provenance list
        halo_provenance = []
        for i in range(len(mcmc_ids)):
            halo_provenance.append({
                'mcmc_id': int(mcmc_ids[i]),
                'original_index': int(original_indices[i])
            })

        # Load combined data
        combined_data = {}
        combined_grp = f['combined_data']
        for key in combined_grp.keys():
            # Convert back to original property name format
            original_key = key.replace('_', '/')
            combined_data[original_key] = combined_grp[key][:]

    print(f"\nLoaded raw DBSCAN results from {filepath}")
    print(f"  Total halos: {metadata.get('n_total_halos', len(cluster_labels))}")
    print(f"  Clustered halos: {metadata.get('n_clustered', 'N/A')}")
    print(f"  Noise halos: {metadata.get('n_noise', 'N/A')}")
    print(f"  Number of clusters: {metadata.get('n_clusters', 'N/A')}")

    return cluster_labels, positions, m200_masses, halo_provenance, combined_data, metadata


def save_hdbscan_clusters_to_hdf5(
    cluster_summaries: List[Dict],
    positions: np.ndarray,
    masses: np.ndarray,
    realization_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    dropped_mask: np.ndarray,
    config,
    output_dir: str,
    filename: str,
    sigma_diagnostics: Dict = None,
    sigma_logM: float = None,
    save_input_catalog: bool = False,
    halo_indices: np.ndarray = None
) -> None:
    """Save HDBSCAN clustering results to HDF5.

    FLAT STRUCTURE (optimized for bulk reads):

    /metadata
        @mcmc_start, @mcmc_end, @m200_mass_cut, @radius_cut
        @min_cluster_size, @min_samples, @cluster_selection_method
        @n_realizations, @n_total_halos, @n_clustered, @n_noise, @n_clusters
        @sigma_logM, @existence_prob_stable, @existence_prob_tentative

    /whitening (diagnostics for reproducibility)
        @method, @sigma_logM
        bin_centers, bin_sigmas (if binned interpolation used)

    /summary (ALL cluster attributes as contiguous arrays - fast bulk read)
        cluster_id (C,)
        original_cluster_id (C,)
        existence_prob (C,)
        n_realizations_present (C,)
        n_members (C,)
        center_xyz (C, 3)
        center_logM (C,)
        var_logM (C,)
        mean_m200_mass (C,)
        m200_mass_std (C,)
        log10_m200_mass_std (C,)
        mean_m500 (C,)
        m500_std (C,)
        log10_m500_std (C,)
        mean_subhalo_mass (C,)
        subhalo_mass_std (C,)
        hdbscan_stability (C,)
        mean_membership_prob (C,)
        min_membership_prob (C,)
        ambiguity_rate (C,)
        status (C,) - string array
        axis_ratio_ba (C,)
        axis_ratio_ca (C,)
        asphericity (C,)
        prolateness (C,)
        position_std (C, 3)
        cov_xyz (C, 3, 3)

    /member_index (CSR-style indexing for O(1) member lookup)
        offsets (C+1,) - start index for each cluster in member arrays
        halo_indices (M,) - concatenated global halo indices for all clusters

    /member_data (all per-member properties as contiguous arrays)
        cluster_id (M,) - which cluster each member belongs to
        positions (M, 3)
        masses (M,)
        realization_ids (M,)
        membership_probs (M,)
        ...other properties from combined_data...

    /assignments (all halos for quick halo->cluster lookup)
        realization_id (N,)
        x, y, z (N,)
        mass (N,)
        logM (N,)
        label (N,) - cluster ID or -1
        membership_prob (N,)
        dropped_by_enforcement (N,) - bool

    /input_catalog (optional, if save_input_catalog=True)
        positions (N, 3)
        mcmc_ids (N,)
        cluster_ids (N,)
        halo_indices (N,)

    To get members for cluster i:
        start = offsets[i]
        end = offsets[i+1]
        member_halo_indices = halo_indices[start:end]
        member_positions = member_data['positions'][start:end]
    """
    filepath = os.path.join(output_dir, filename)
    n_realizations = config.mode1.mcmc_end - config.mode1.mcmc_start + 1

    n_total_halos = len(labels)
    n_noise = np.sum(labels == -1)
    n_clustered = n_total_halos - n_noise
    n_clusters = len(cluster_summaries)

    with h5py.File(filepath, 'w') as f:
        # Metadata group
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['mcmc_start'] = config.mode1.mcmc_start
        meta_grp.attrs['mcmc_end'] = config.mode1.mcmc_end
        meta_grp.attrs['m200_mass_cut'] = config.mode1.m200_mass_cut
        meta_grp.attrs['radius_cut'] = config.mode1.radius_cut
        meta_grp.attrs['min_cluster_size'] = config.mode1.min_cluster_size
        meta_grp.attrs['min_samples'] = config.mode1.min_samples
        meta_grp.attrs['cluster_selection_method'] = config.mode1.cluster_selection_method
        meta_grp.attrs['alpha'] = config.mode1.alpha
        meta_grp.attrs['n_realizations'] = n_realizations
        meta_grp.attrs['n_total_halos'] = n_total_halos
        meta_grp.attrs['n_clustered'] = n_clustered
        meta_grp.attrs['n_noise'] = n_noise
        meta_grp.attrs['n_clusters'] = n_clusters
        meta_grp.attrs['existence_prob_stable'] = config.mode1.existence_prob_stable
        meta_grp.attrs['existence_prob_tentative'] = config.mode1.existence_prob_tentative
        meta_grp.attrs['basedir'] = config.global_config.basedir
        meta_grp.attrs['observer_coords'] = config.global_config.observer_coords
        meta_grp.attrs['hdf5_format_version'] = 2  # Mark as new flat format

        # Whitening diagnostics
        whitening_grp = f.create_group('whitening')
        if sigma_logM is not None:
            whitening_grp.attrs['sigma_logM'] = sigma_logM
        if sigma_diagnostics is not None:
            whitening_grp.attrs['method'] = sigma_diagnostics.get('method', 'unknown')
            if 'bin_centers' in sigma_diagnostics:
                whitening_grp.create_dataset('bin_centers', data=sigma_diagnostics['bin_centers'])
            if 'bin_sigmas' in sigma_diagnostics:
                whitening_grp.create_dataset('bin_sigmas', data=sigma_diagnostics['bin_sigmas'])
            if 'bin_counts' in sigma_diagnostics:
                whitening_grp.create_dataset('bin_counts', data=sigma_diagnostics['bin_counts'])
            if 'sigma_floor' in sigma_diagnostics:
                whitening_grp.attrs['sigma_floor'] = sigma_diagnostics['sigma_floor']
            if 'sigma_ceiling' in sigma_diagnostics:
                whitening_grp.attrs['sigma_ceiling'] = sigma_diagnostics['sigma_ceiling']
            if 'k' in sigma_diagnostics:
                whitening_grp.attrs['knn_k'] = sigma_diagnostics['k']

        # Summary group - ALL cluster attributes as contiguous arrays
        summary_grp = f.create_group('summary')
        if n_clusters > 0:
            # Helper to safely get values, replacing NaN with -1 for storage
            def safe_array(key, default=-1.0):
                values = []
                for c in cluster_summaries:
                    val = c.get(key, default)
                    if isinstance(val, float) and np.isnan(val):
                        val = default
                    values.append(val)
                return np.array(values)

            # Core identifiers
            summary_grp.create_dataset('cluster_id', data=np.array([c['cluster_id'] for c in cluster_summaries]))
            summary_grp.create_dataset('original_cluster_id', data=np.array([c.get('original_cluster_id', c['cluster_id']) for c in cluster_summaries]))

            # Existence and membership
            summary_grp.create_dataset('existence_prob', data=np.array([c['existence_prob'] for c in cluster_summaries]))
            summary_grp.create_dataset('n_realizations_present', data=np.array([c['n_realizations_present'] for c in cluster_summaries]))
            summary_grp.create_dataset('n_members', data=np.array([c['n_members'] for c in cluster_summaries]))

            # Position statistics
            summary_grp.create_dataset('center_xyz', data=np.array([c['center_xyz'] for c in cluster_summaries]))
            summary_grp.create_dataset('position_std', data=np.array([c['position_std'] for c in cluster_summaries]))

            # Covariance matrices
            cov_list = []
            for c in cluster_summaries:
                cov = c.get('cov_xyz', np.eye(3) * np.nan)
                if not isinstance(cov, np.ndarray):
                    cov = np.eye(3) * np.nan
                cov_list.append(cov)
            summary_grp.create_dataset('cov_xyz', data=np.array(cov_list))

            # Mass statistics
            summary_grp.create_dataset('center_logM', data=safe_array('center_logM'))
            summary_grp.create_dataset('var_logM', data=safe_array('var_logM'))
            summary_grp.create_dataset('mean_m200_mass', data=np.array([c['mean_m200_mass'] for c in cluster_summaries]))
            summary_grp.create_dataset('m200_mass_std', data=safe_array('m200_mass_std'))
            summary_grp.create_dataset('log10_m200_mass_std', data=safe_array('log10_m200_mass_std'))
            summary_grp.create_dataset('mean_m500', data=safe_array('mean_m500'))
            summary_grp.create_dataset('m500_std', data=safe_array('m500_std'))
            summary_grp.create_dataset('log10_m500_std', data=safe_array('log10_m500_std'))
            summary_grp.create_dataset('mean_subhalo_mass', data=safe_array('mean_subhalo_mass'))
            summary_grp.create_dataset('subhalo_mass_std', data=safe_array('subhalo_mass_std'))

            # Clustering quality metrics
            summary_grp.create_dataset('hdbscan_stability', data=safe_array('hdbscan_stability'))
            summary_grp.create_dataset('mean_membership_prob', data=np.array([c['mean_membership_prob'] for c in cluster_summaries]))
            summary_grp.create_dataset('min_membership_prob', data=np.array([c['min_membership_prob'] for c in cluster_summaries]))
            summary_grp.create_dataset('ambiguity_rate', data=np.array([c['ambiguity_rate'] for c in cluster_summaries]))

            # Shape measures
            summary_grp.create_dataset('axis_ratio_ba', data=safe_array('axis_ratio_ba'))
            summary_grp.create_dataset('axis_ratio_ca', data=safe_array('axis_ratio_ca'))
            summary_grp.create_dataset('asphericity', data=safe_array('asphericity'))
            summary_grp.create_dataset('prolateness', data=safe_array('prolateness'))

            # Status as string array
            dt = h5py.special_dtype(vlen=str)
            summary_grp.create_dataset('status', data=np.array([c['status'] for c in cluster_summaries], dtype=object), dtype=dt)

        # Member index group - CSR-style for O(1) member lookup
        member_index_grp = f.create_group('member_index')
        if n_clusters > 0:
            # Build CSR-style offset array and concatenated member indices
            offsets = [0]
            all_halo_indices = []
            all_member_probs = []

            for cluster in cluster_summaries:
                member_indices = cluster['member_indices']
                all_halo_indices.extend(member_indices)
                all_member_probs.extend(cluster['member_probs'])
                offsets.append(len(all_halo_indices))

            member_index_grp.create_dataset('offsets', data=np.array(offsets, dtype=np.int64))
            member_index_grp.create_dataset('halo_indices', data=np.array(all_halo_indices, dtype=np.int64))

        # Member data group - all per-member properties as contiguous arrays
        member_data_grp = f.create_group('member_data')
        if n_clusters > 0:
            # Concatenate all member data across clusters
            all_cluster_ids = []
            all_positions = []
            all_masses = []
            all_realization_ids = []
            all_membership_probs = []

            # Collect all unique keys from member_data across clusters
            member_data_keys = set()
            for cluster in cluster_summaries:
                if 'member_data' in cluster:
                    member_data_keys.update(cluster['member_data'].keys())

            # Initialize collectors for additional member data
            additional_data = {key: [] for key in member_data_keys}

            for cluster in cluster_summaries:
                member_indices = cluster['member_indices']
                n_members = len(member_indices)

                all_cluster_ids.extend([cluster['cluster_id']] * n_members)
                all_positions.append(positions[member_indices])
                all_masses.append(masses[member_indices])
                all_realization_ids.append(realization_ids[member_indices])
                all_membership_probs.append(cluster['member_probs'])

                # Collect additional member data
                if 'member_data' in cluster:
                    for key in member_data_keys:
                        if key in cluster['member_data']:
                            additional_data[key].append(cluster['member_data'][key])
                        else:
                            # Fill with NaN for missing data
                            shape = cluster['member_data'].get(list(cluster['member_data'].keys())[0], np.array([])).shape
                            if len(shape) > 1:
                                fill_shape = (n_members,) + shape[1:]
                            else:
                                fill_shape = (n_members,)
                            additional_data[key].append(np.full(fill_shape, np.nan))

            # Save core member arrays
            member_data_grp.create_dataset('cluster_id', data=np.array(all_cluster_ids, dtype=np.int32))
            member_data_grp.create_dataset('positions', data=np.vstack(all_positions))
            member_data_grp.create_dataset('masses', data=np.concatenate(all_masses))
            member_data_grp.create_dataset('realization_ids', data=np.concatenate(all_realization_ids))
            member_data_grp.create_dataset('membership_probs', data=np.concatenate(all_membership_probs))

            # Save additional member data
            for key, data_list in additional_data.items():
                if len(data_list) > 0:
                    dataset_name = key.replace('/', '_')
                    if dataset_name not in ['positions', 'masses', 'realization_ids', 'membership_probs', 'cluster_id']:
                        try:
                            if len(data_list[0].shape) > 1:
                                combined = np.vstack(data_list)
                            else:
                                combined = np.concatenate(data_list)
                            member_data_grp.create_dataset(dataset_name, data=combined)
                        except Exception:
                            pass  # Skip problematic data

        # Assignments group (all halos for quick halo->cluster lookup)
        assignments_grp = f.create_group('assignments')
        assignments_grp.create_dataset('realization_id', data=realization_ids)
        assignments_grp.create_dataset('x', data=positions[:, 0])
        assignments_grp.create_dataset('y', data=positions[:, 1])
        assignments_grp.create_dataset('z', data=positions[:, 2])
        assignments_grp.create_dataset('mass', data=masses)
        assignments_grp.create_dataset('logM', data=np.log10(masses))
        assignments_grp.create_dataset('label', data=labels)
        assignments_grp.create_dataset('membership_prob', data=probabilities)
        assignments_grp.create_dataset('dropped_by_enforcement', data=dropped_mask)

        # Input catalog group (optional - all input halos including unclustered)
        if save_input_catalog:
            input_grp = f.create_group('input_catalog')
            input_grp.create_dataset('positions', data=positions)
            input_grp.create_dataset('mcmc_ids', data=realization_ids)
            input_grp.create_dataset('cluster_ids', data=labels)
            if halo_indices is not None:
                input_grp.create_dataset('halo_indices', data=halo_indices)

    print(f"\nHDBSCAN results saved to {filepath}")
    print(f"  Total clusters: {n_clusters}")
    print(f"  Total halos: {n_total_halos}")
    print(f"  Clustered: {n_clustered} ({100*n_clustered/n_total_halos:.1f}%)")
    print(f"  Noise: {n_noise} ({100*n_noise/n_total_halos:.1f}%)")


def load_hdbscan_clusters_from_hdf5(
    output_dir: str,
    filename: str,
    min_existence_prob: float = 0.0,
    load_members: bool = True
) -> tuple:
    """Load HDBSCAN clustering results from HDF5.

    Supports both new flat format (v2) and legacy nested format (v1).

    Args:
        output_dir: Directory containing the HDF5 file
        filename: HDF5 filename
        min_existence_prob: Minimum existence probability for loading clusters
        load_members: Whether to load member data (set False for summary only)

    Returns:
        cluster_summaries: List of cluster dictionaries
        assignments: Dictionary with all halo assignments
        metadata: Dictionary with run metadata
    """
    filepath = os.path.join(output_dir, filename)

    cluster_summaries = []
    assignments = {}
    metadata = {}

    with h5py.File(filepath, 'r') as f:
        # Load metadata
        meta_grp = f['metadata']
        for key in meta_grp.attrs.keys():
            metadata[key] = meta_grp.attrs[key]

        # Check format version
        format_version = metadata.get('hdf5_format_version', 1)

        # Load whitening diagnostics if present
        if 'whitening' in f:
            whitening_grp = f['whitening']
            metadata['whitening'] = {}
            for key in whitening_grp.attrs.keys():
                metadata['whitening'][key] = whitening_grp.attrs[key]
            for key in whitening_grp.keys():
                metadata['whitening'][key] = whitening_grp[key][:]

        # Load assignments
        if 'assignments' in f:
            assignments_grp = f['assignments']
            for key in assignments_grp.keys():
                assignments[key] = assignments_grp[key][:]

        # Load clusters based on format version
        if format_version >= 2 and 'member_index' in f:
            # New flat format (v2) - bulk read from arrays
            cluster_summaries = _load_clusters_flat_format(
                f, min_existence_prob, load_members
            )
        elif 'clusters' in f:
            # Legacy nested format (v1)
            cluster_summaries = _load_clusters_legacy_format(
                f, min_existence_prob, load_members
            )

    # Sort by cluster_id
    cluster_summaries.sort(key=lambda x: x['cluster_id'])

    print(f"\nLoaded HDBSCAN results from {filepath}")
    print(f"  Format version: {format_version}")
    print(f"  Clusters loaded: {len(cluster_summaries)}")
    print(f"  Total halos: {metadata.get('n_total_halos', 'N/A')}")

    return cluster_summaries, assignments, metadata


def _load_clusters_flat_format(f, min_existence_prob, load_members):
    """Load clusters from new flat format (v2)."""
    cluster_summaries = []
    summary = f['summary']

    # Bulk read all summary arrays
    cluster_ids = summary['cluster_id'][:]
    original_cluster_ids = summary['original_cluster_id'][:] if 'original_cluster_id' in summary else cluster_ids
    existence_probs = summary['existence_prob'][:]
    n_realizations_present = summary['n_realizations_present'][:]
    n_members = summary['n_members'][:]
    center_xyzs = summary['center_xyz'][:]
    position_stds = summary['position_std'][:]
    cov_xyzs = summary['cov_xyz'][:] if 'cov_xyz' in summary else np.full((len(cluster_ids), 3, 3), np.nan)
    center_logMs = summary['center_logM'][:] if 'center_logM' in summary else np.full(len(cluster_ids), np.nan)
    var_logMs = summary['var_logM'][:] if 'var_logM' in summary else np.full(len(cluster_ids), np.nan)
    mean_m200_masses = summary['mean_m200_mass'][:]
    m200_mass_stds = summary['m200_mass_std'][:] if 'm200_mass_std' in summary else np.full(len(cluster_ids), np.nan)
    log10_m200_mass_stds = summary['log10_m200_mass_std'][:] if 'log10_m200_mass_std' in summary else np.full(len(cluster_ids), np.nan)
    mean_m500s = summary['mean_m500'][:] if 'mean_m500' in summary else np.full(len(cluster_ids), np.nan)
    m500_stds = summary['m500_std'][:] if 'm500_std' in summary else np.full(len(cluster_ids), np.nan)
    log10_m500_stds = summary['log10_m500_std'][:] if 'log10_m500_std' in summary else np.full(len(cluster_ids), np.nan)
    mean_subhalo_masses = summary['mean_subhalo_mass'][:] if 'mean_subhalo_mass' in summary else np.full(len(cluster_ids), np.nan)
    subhalo_mass_stds = summary['subhalo_mass_std'][:] if 'subhalo_mass_std' in summary else np.full(len(cluster_ids), np.nan)
    hdbscan_stabilities = summary['hdbscan_stability'][:] if 'hdbscan_stability' in summary else np.full(len(cluster_ids), np.nan)
    mean_membership_probs = summary['mean_membership_prob'][:]
    min_membership_probs = summary['min_membership_prob'][:]
    ambiguity_rates = summary['ambiguity_rate'][:]
    statuses = summary['status'][:] if 'status' in summary else np.full(len(cluster_ids), 'unknown')
    axis_ratio_bas = summary['axis_ratio_ba'][:] if 'axis_ratio_ba' in summary else np.full(len(cluster_ids), np.nan)
    axis_ratio_cas = summary['axis_ratio_ca'][:] if 'axis_ratio_ca' in summary else np.full(len(cluster_ids), np.nan)
    asphericities = summary['asphericity'][:] if 'asphericity' in summary else np.full(len(cluster_ids), np.nan)
    prolatenesses = summary['prolateness'][:] if 'prolateness' in summary else np.full(len(cluster_ids), np.nan)

    # Load member index for member data access
    offsets = None
    halo_indices = None
    member_data_arrays = {}

    if load_members and 'member_index' in f:
        member_index = f['member_index']
        offsets = member_index['offsets'][:]
        halo_indices = member_index['halo_indices'][:]

        # Load all member data arrays
        if 'member_data' in f:
            member_data_grp = f['member_data']
            for key in member_data_grp.keys():
                member_data_arrays[key] = member_data_grp[key][:]

    # Helper to convert -1 sentinel back to NaN
    def to_nan(val):
        if isinstance(val, (int, float)) and val < 0:
            return np.nan
        return val

    # Build cluster summaries
    for i in range(len(cluster_ids)):
        if existence_probs[i] < min_existence_prob:
            continue

        cluster = {
            'cluster_id': int(cluster_ids[i]),
            'original_cluster_id': int(original_cluster_ids[i]),
            'existence_prob': float(existence_probs[i]),
            'n_realizations_present': int(n_realizations_present[i]),
            'n_members': int(n_members[i]),
            'center_xyz': center_xyzs[i],
            'position_std': position_stds[i],
            'cov_xyz': cov_xyzs[i],
            'center_logM': to_nan(float(center_logMs[i])),
            'var_logM': to_nan(float(var_logMs[i])),
            'mean_m200_mass': float(mean_m200_masses[i]),
            'm200_mass_std': to_nan(float(m200_mass_stds[i])),
            'log10_m200_mass_std': to_nan(float(log10_m200_mass_stds[i])),
            'mean_m500': to_nan(float(mean_m500s[i])),
            'm500_std': to_nan(float(m500_stds[i])),
            'log10_m500_std': to_nan(float(log10_m500_stds[i])),
            'mean_subhalo_mass': to_nan(float(mean_subhalo_masses[i])),
            'subhalo_mass_std': to_nan(float(subhalo_mass_stds[i])),
            'hdbscan_stability': to_nan(float(hdbscan_stabilities[i])),
            'mean_membership_prob': float(mean_membership_probs[i]),
            'min_membership_prob': float(min_membership_probs[i]),
            'ambiguity_rate': float(ambiguity_rates[i]),
            'status': str(statuses[i]) if isinstance(statuses[i], (str, bytes)) else statuses[i].decode() if hasattr(statuses[i], 'decode') else str(statuses[i]),
            'axis_ratio_ba': to_nan(float(axis_ratio_bas[i])),
            'axis_ratio_ca': to_nan(float(axis_ratio_cas[i])),
            'asphericity': to_nan(float(asphericities[i])),
            'prolateness': to_nan(float(prolatenesses[i])),
        }

        # Load member data using CSR-style indexing
        if load_members and offsets is not None:
            start = offsets[i]
            end = offsets[i + 1]

            member_data = {}
            for key, data in member_data_arrays.items():
                original_key = key.replace('_', '/')
                member_data[original_key] = data[start:end]

            cluster['member_data'] = member_data
            cluster['member_indices'] = halo_indices[start:end]

        cluster_summaries.append(cluster)

    return cluster_summaries


def _load_clusters_legacy_format(f, min_existence_prob, load_members):
    """Load clusters from legacy nested format (v1)."""
    cluster_summaries = []
    clusters_grp = f['clusters']

    for cluster_name in clusters_grp.keys():
        cluster_grp = clusters_grp[cluster_name]

        # Filter by existence probability
        existence_prob = float(cluster_grp.attrs['existence_prob'])
        if existence_prob < min_existence_prob:
            continue

        cluster = {
            'cluster_id': int(cluster_grp.attrs['cluster_id']),
            'original_cluster_id': int(cluster_grp.attrs.get('original_cluster_id', cluster_grp.attrs['cluster_id'])),
            'existence_prob': existence_prob,
            'n_realizations_present': int(cluster_grp.attrs['n_realizations_present']),
            'n_members': int(cluster_grp.attrs['n_members']),
            'center_xyz': cluster_grp.attrs['center_xyz'],
            'center_logM': float(cluster_grp.attrs['center_logM']),
            'var_logM': float(cluster_grp.attrs['var_logM']) if cluster_grp.attrs['var_logM'] >= 0 else np.nan,
            'mean_m200_mass': float(cluster_grp.attrs['mean_m200_mass']),
            'm200_mass_std': float(cluster_grp.attrs['m200_mass_std']) if cluster_grp.attrs['m200_mass_std'] >= 0 else np.nan,
            'log10_m200_mass_std': float(cluster_grp.attrs['log10_m200_mass_std']) if cluster_grp.attrs['log10_m200_mass_std'] >= 0 else np.nan,
            'mean_m500': float(cluster_grp.attrs['mean_m500']) if cluster_grp.attrs['mean_m500'] >= 0 else np.nan,
            'm500_std': float(cluster_grp.attrs['m500_std']) if cluster_grp.attrs['m500_std'] >= 0 else np.nan,
            'log10_m500_std': float(cluster_grp.attrs['log10_m500_std']) if cluster_grp.attrs['log10_m500_std'] >= 0 else np.nan,
            'mean_subhalo_mass': float(cluster_grp.attrs['mean_subhalo_mass']) if cluster_grp.attrs['mean_subhalo_mass'] >= 0 else np.nan,
            'subhalo_mass_std': float(cluster_grp.attrs['subhalo_mass_std']) if cluster_grp.attrs['subhalo_mass_std'] >= 0 else np.nan,
            'hdbscan_stability': float(cluster_grp.attrs['hdbscan_stability']) if cluster_grp.attrs['hdbscan_stability'] >= 0 else np.nan,
            'mean_membership_prob': float(cluster_grp.attrs['mean_membership_prob']),
            'min_membership_prob': float(cluster_grp.attrs['min_membership_prob']),
            'ambiguity_rate': float(cluster_grp.attrs['ambiguity_rate']),
            'status': str(cluster_grp.attrs['status']),
            'axis_ratio_ba': float(cluster_grp.attrs['axis_ratio_ba']) if cluster_grp.attrs['axis_ratio_ba'] >= 0 else np.nan,
            'axis_ratio_ca': float(cluster_grp.attrs['axis_ratio_ca']) if cluster_grp.attrs['axis_ratio_ca'] >= 0 else np.nan,
            'asphericity': float(cluster_grp.attrs['asphericity']) if cluster_grp.attrs['asphericity'] >= 0 else np.nan,
            'prolateness': float(cluster_grp.attrs['prolateness']) if cluster_grp.attrs['prolateness'] >= 0 else np.nan,
            'position_std': cluster_grp.attrs['position_std'],
        }

        # Load covariance if present
        if 'cov_xyz' in cluster_grp:
            cluster['cov_xyz'] = cluster_grp['cov_xyz'][:]
        else:
            cluster['cov_xyz'] = np.eye(3) * np.nan

        # Load member data if requested
        if load_members and 'members' in cluster_grp:
            members_grp = cluster_grp['members']
            member_data = {}
            for key in members_grp.keys():
                original_key = key.replace('_', '/')
                member_data[original_key] = members_grp[key][:]
            cluster['member_data'] = member_data

        cluster_summaries.append(cluster)

    return cluster_summaries
