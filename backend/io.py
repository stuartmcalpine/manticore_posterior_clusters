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
    'load_raw_dbscan_from_hdf5'
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

def load_clusters_from_hdf5(output_dir, filename="clusters.h5", minimal=True,
        min_m200_mass=1e14):
    filepath = os.path.join(output_dir, filename)
    
    clusters = []
    metadata = {}
    
    with h5py.File(filepath, 'r') as f:
        meta_grp = f['metadata']
        for key in meta_grp.attrs.keys():
            metadata[key] = meta_grp.attrs[key]
        
        # Handle both old 'clusters' and new 'posterior' group names
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
            for key in assoc_grp.keys():
                # Convert back to original property name format
                original_key = key.replace('_', '/')

                if minimal and original_key not in ["SO/200_crit/CentreOfMass", "SO/200_crit/TotalMass", 
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
        Configuration object with mode1a settings
    output_dir : str
        Output directory path
    filename : str
        Output filename
    """
    filepath = os.path.join(output_dir, filename)

    with h5py.File(filepath, 'w') as f:
        # Metadata group
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['mcmc_start'] = config.mode1a.mcmc_start
        meta_grp.attrs['mcmc_end'] = config.mode1a.mcmc_end
        meta_grp.attrs['eps'] = config.mode1a.eps
        meta_grp.attrs['min_samples'] = config.mode1a.min_samples
        meta_grp.attrs['m200_mass_cut'] = config.mode1a.m200_mass_cut
        meta_grp.attrs['radius_cut'] = config.mode1a.radius_cut
        meta_grp.attrs['mass_weighted_clustering'] = config.mode1a.mass_weighted_clustering
        meta_grp.attrs['mass_weight_power'] = config.mode1a.mass_weight_power
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
