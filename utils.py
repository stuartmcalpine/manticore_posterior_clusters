# utils.py
import h5py
import numpy as np
import os
from typing import Dict, List, Any

def ensure_output_dir(config):
    os.makedirs(config.global_config.output_dir, exist_ok=True)

def save_clusters_to_hdf5(stable_haloes, positions, masses, halo_provenance, cluster_labels, config, filename="clusters.h5"):
    filepath = os.path.join(config.global_config.output_dir, filename)
    
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

def load_clusters_from_hdf5(config, filename="clusters.h5"):
    filepath = os.path.join(config.global_config.output_dir, filename)
    
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
            clusters.append(cluster)
    
    return clusters, metadata

def save_halo_traces_to_hdf5(halo_traces, config, cluster_metadata, filename="halo_traces.h5"):
    filepath = os.path.join(config.global_config.output_dir, filename)
    
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

def load_halo_traces_from_hdf5(config, filename="halo_traces.h5"):
    filepath = os.path.join(config.global_config.output_dir, filename)
    
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
