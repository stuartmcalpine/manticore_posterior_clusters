import h5py
import numpy as np
import os
from typing import Dict, List, Any
from collections import defaultdict

def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def save_clusters_to_hdf5(stable_haloes, positions, masses, halo_provenance, cluster_labels, config, output_dir, filename="clusters.h5"):
    filepath = os.path.join(output_dir, filename)
    
    with h5py.File(filepath, 'w') as f:
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['mcmc_start'] = config.mode1.mcmc_start
        meta_grp.attrs['mcmc_end'] = config.mode1.mcmc_end
        meta_grp.attrs['eps'] = config.mode1.eps
        meta_grp.attrs['min_samples'] = config.mode1.min_samples
        meta_grp.attrs['mass_cut'] = config.mode1.mass_cut
        meta_grp.attrs['radius_cut'] = config.mode1.radius_cut
        meta_grp.attrs['basedir'] = config.global_config.basedir
        meta_grp.attrs['observer_coords'] = config.global_config.observer_coords
        
        all_data_grp = f.create_group('all_haloes')
        all_data_grp.create_dataset('positions', data=positions)
        all_data_grp.create_dataset('masses', data=masses)
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
            cluster_grp.attrs['mean_mass'] = cluster['mean_mass']
            cluster_grp.attrs['position_std'] = cluster['position_std']
            cluster_grp.attrs['mass_std'] = cluster['mass_std']
            
            member_grp = cluster_grp.create_group('members')
            
            # Save all member data properties
            for key, data in cluster['member_data'].items():
                member_grp.create_dataset(key.replace('/', '_'), data=data)
            
            mcmc_ids = np.array([m['mcmc_id'] for m in cluster['members']])
            orig_indices = np.array([m['original_index'] for m in cluster['members']])
            member_grp.create_dataset('mcmc_ids', data=mcmc_ids)
            member_grp.create_dataset('original_indices', data=orig_indices)

def load_clusters_from_hdf5(output_dir, filename="clusters.h5", minimal=True,
        min_mass=1e14):
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

                    if minimal and original_key not in ["BoundSubhalo/CentreOfMass", "BoundSubhalo/TotalMass"]:
                        continue
                    member_data[original_key] = member_grp[key][:]
            
            cluster = {
                'cluster_id': int(cluster_grp.attrs['cluster_id']),
                'cluster_size': int(cluster_grp.attrs['cluster_size']),
                'mean_position': cluster_grp.attrs['mean_position'],
                'mean_mass': float(cluster_grp.attrs['mean_mass']),
                'position_std': cluster_grp.attrs['position_std'],
                'mass_std': float(cluster_grp.attrs['mass_std']),
                'members': members,
                'member_data': member_data
            }

            if cluster["mean_mass"] >= min_mass:
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
               'mean_mass': float(cluster_grp.attrs['mean_mass']),
               'position_std': cluster_grp.attrs['position_std'],
               'mass_std': float(cluster_grp.attrs['mass_std'])
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
