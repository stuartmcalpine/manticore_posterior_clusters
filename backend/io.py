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
    'load_single_cluster_members'
]

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
            cluster_grp.attrs['axis_ratio_ba'] = cluster.get('axis_ratio_ba', np.nan)
            cluster_grp.attrs['axis_ratio_ca'] = cluster.get('axis_ratio_ca', np.nan)
            cluster_grp.attrs['asphericity'] = cluster.get('asphericity', np.nan)
            cluster_grp.attrs['prolateness'] = cluster.get('prolateness', np.nan)

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
                'axis_ratio_ba': float(cluster_grp.attrs.get('axis_ratio_ba', np.nan)),
                'axis_ratio_ca': float(cluster_grp.attrs.get('axis_ratio_ca', np.nan)),
                'asphericity': float(cluster_grp.attrs.get('asphericity', np.nan)),
                'prolateness': float(cluster_grp.attrs.get('prolateness', np.nan)),
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
        
        for halo_key, trace_data in halo_traces.items():
            # Find final snapshot (77) data
            final_idx = np.where(trace_data['snapshots'] == 77)[0]
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
               'axis_ratio_ba': float(cluster_grp.attrs.get('axis_ratio_ba', np.nan)),
               'axis_ratio_ca': float(cluster_grp.attrs.get('axis_ratio_ca', np.nan)),
               'asphericity': float(cluster_grp.attrs.get('asphericity', np.nan)),
               'prolateness': float(cluster_grp.attrs.get('prolateness', np.nan)),
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
