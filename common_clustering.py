# common_clustering.py
import numpy as np
import os
from pymanticore.swift_analysis import SOAPData
from sklearn.cluster import DBSCAN

def enforce_mcmc_constraint(cluster_labels, positions, mcmc_ids):
    """Post-process clusters to ensure at most one halo per MCMC per cluster"""
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_mask = cluster_labels == cluster_id
        cluster_positions = positions[cluster_mask]
        cluster_mcmc_ids = mcmc_ids[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        # Find cluster center
        cluster_center = np.mean(cluster_positions, axis=0)
        
        # Group by MCMC ID
        unique_mcmc_ids = np.unique(cluster_mcmc_ids)
        
        for mcmc_id in unique_mcmc_ids:
            mcmc_mask = cluster_mcmc_ids == mcmc_id
            mcmc_indices_in_cluster = cluster_indices[mcmc_mask]
            
            if len(mcmc_indices_in_cluster) > 1:
                # Multiple halos from same MCMC in this cluster
                mcmc_positions = cluster_positions[mcmc_mask]
                
                # Find the one closest to cluster center
                distances_to_center = np.linalg.norm(mcmc_positions - cluster_center, axis=1)
                closest_idx = np.argmin(distances_to_center)
                
                # Keep only the closest one, mark others as noise
                for i, global_idx in enumerate(mcmc_indices_in_cluster):
                    if i != closest_idx:
                        cluster_labels[global_idx] = -1
    
    return cluster_labels

def combine_haloes(mcmc_data):
    combined_positions = []
    combined_masses = []
    combined_progenitor_indices = []
    halo_provenance = []
    
    for mcmc_id, data in mcmc_data.items():
        positions = data['BoundSubhalo/CentreOfMass']
        masses = data['BoundSubhalo/TotalMass']
        progenitor_indices = data['SOAP/ProgenitorIndex']
        
        combined_positions.append(positions)
        combined_masses.append(masses)
        combined_progenitor_indices.append(progenitor_indices)
        
        for i in range(len(masses)):
            halo_provenance.append({'mcmc_id': mcmc_id, 'original_index': i})
    
    combined_positions = np.vstack(combined_positions)
    combined_masses = np.concatenate(combined_masses)
    combined_progenitor_indices = np.concatenate(combined_progenitor_indices)
    
    return combined_positions, combined_masses, combined_progenitor_indices, halo_provenance

def load_data_with_radius_filter(config, radius_inner=None, radius_outer=None):
    mcmc_data = {}
    
    # Use radius_outer for initial loading if provided, otherwise use radius_cut
    initial_radius_cut = radius_outer if radius_outer is not None else config.mode1.radius_cut
    
    for mcmc_id in range(config.mode1.mcmc_start, config.mode1.mcmc_end + 1):
        filename = os.path.join(config.global_config.basedir, f"mcmc_{mcmc_id}/soap/SOAP_uncompressed/HBTplus/halo_properties_0077.hdf5")
        soap_data = SOAPData(filename, mass_cut=config.mode1.mass_cut, radius_cut=initial_radius_cut)
        to_load = ["BoundSubhalo/TotalMass", "SO/200_crit/SORadius", "BoundSubhalo/CentreOfMass", "SOAP/ProgenitorIndex"]
        soap_data.load_groups(properties=to_load, only_centrals=True)
        soap_data.set_observer(config.global_config.observer_coords, skip_redshift=True)
        
        # Apply radius filtering if both inner and outer are specified
        if radius_inner is not None and radius_outer is not None:
            distances = soap_data.data['dist']
            radius_mask = (distances >= radius_inner) & (distances <= radius_outer)
            
            filtered_data = {}
            for key, value in soap_data.data.items():
                if isinstance(value, np.ndarray) and len(value) == len(distances):
                    filtered_data[key] = value[radius_mask]
                else:
                    filtered_data[key] = value
            
            mcmc_data[mcmc_id] = filtered_data
        else:
            mcmc_data[mcmc_id] = soap_data.data.copy()
        
        print(f"Loaded MCMC step {mcmc_id}: {len(mcmc_data[mcmc_id]['BoundSubhalo/TotalMass'])} haloes")
    
    return mcmc_data

def find_stable_haloes(mcmc_data, config):
    positions, masses, progenitor_indices, halo_provenance = combine_haloes(mcmc_data)
    
    mcmc_ids = np.array([p['mcmc_id'] for p in halo_provenance])
    
    # Run standard DBSCAN
    clustering = DBSCAN(eps=config.mode1.eps, min_samples=config.mode1.min_samples)
    cluster_labels = clustering.fit_predict(positions)
    
    # Enforce MCMC constraint
    cluster_labels = enforce_mcmc_constraint(cluster_labels, positions, mcmc_ids)
    
    stable_haloes = []
    
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue
            
        cluster_mask = cluster_labels == cluster_id
        cluster_positions = positions[cluster_mask]
        cluster_masses = masses[cluster_mask]
        cluster_progenitor_indices = progenitor_indices[cluster_mask]
        cluster_provenance = [halo_provenance[i] for i in range(len(halo_provenance)) if cluster_mask[i]]
        
        cluster_size = len(cluster_positions)
        mean_position = np.mean(cluster_positions, axis=0)
        mean_mass = np.mean(cluster_masses)
        position_std = np.std(cluster_positions, axis=0)
        mass_std = np.std(cluster_masses)
        
        stable_haloes.append({
            'cluster_id': cluster_id,
            'cluster_size': cluster_size,
            'mean_position': mean_position,
            'mean_mass': mean_mass,
            'position_std': position_std,
            'mass_std': mass_std,
            'members': cluster_provenance,
            'member_data': {
                'positions': cluster_positions,
                'masses': cluster_masses,
                'progenitor_indices': cluster_progenitor_indices
            }
        })
    
    return stable_haloes, positions, masses, halo_provenance, cluster_labels
