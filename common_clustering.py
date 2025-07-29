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
    combined_data = {}
    halo_provenance = []
    
    # Get all available property keys from first MCMC sample
    first_mcmc = next(iter(mcmc_data.values()))
    property_keys = list(first_mcmc.keys())
    
    # Initialize combined arrays for each property
    for key in property_keys:
        combined_data[key] = []
    
    for mcmc_id, data in mcmc_data.items():
        n_haloes = len(data['BoundSubhalo/TotalMass'])
        
        # Combine all properties
        for key in property_keys:
            if key in data:
                combined_data[key].append(data[key])
            else:
                # Handle missing properties with appropriate fill values
                if 'Mass' in key:
                    fill_shape = (n_haloes,)
                elif 'CentreOfMass' in key:
                    fill_shape = (n_haloes, 3)
                else:
                    fill_shape = (n_haloes,)
                combined_data[key].append(np.full(fill_shape, np.nan))
        
        # Track provenance
        for i in range(n_haloes):
            halo_provenance.append({'mcmc_id': mcmc_id, 'original_index': i})
    
    # Stack/concatenate all properties
    for key in property_keys:
        if combined_data[key]:
            if len(combined_data[key][0].shape) == 1:
                combined_data[key] = np.concatenate(combined_data[key])
            else:
                combined_data[key] = np.vstack(combined_data[key])
    
    return combined_data, halo_provenance

def load_data_with_radius_filter(config, radius_inner=None, radius_outer=None):
    mcmc_data = {}
    
    # Use radius_outer for initial loading if provided, otherwise use radius_cut
    initial_radius_cut = radius_outer if radius_outer is not None else config.mode1.radius_cut
    
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
    
    for mcmc_id in range(config.mode1.mcmc_start, config.mode1.mcmc_end + 1):
        filename = os.path.join(config.global_config.basedir, f"mcmc_{mcmc_id}/soap/SOAP_uncompressed/HBTplus/halo_properties_0077.hdf5")
        soap_data = SOAPData(filename, mass_cut=config.mode1.mass_cut, radius_cut=initial_radius_cut)
        soap_data.load_groups(properties=to_load, only_centrals=True)
        soap_data.set_observer(config.global_config.observer_coords, skip_redshift=True)
       
        # Don't want to keep redshift
        del soap_data.data["redshift"]

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
    combined_data, halo_provenance = combine_haloes(mcmc_data)
    
    positions = combined_data['BoundSubhalo/CentreOfMass']
    masses = combined_data['BoundSubhalo/TotalMass']
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
        cluster_provenance = [halo_provenance[i] for i in range(len(halo_provenance)) if cluster_mask[i]]
        
        cluster_size = len(cluster_positions)
        mean_position = np.mean(cluster_positions, axis=0)
        mean_mass = np.mean(cluster_masses)
        position_std = np.std(cluster_positions, axis=0)
        mass_std = np.std(cluster_masses)
        
        # Extract all member data for this cluster
        member_data = {}
        for key, data in combined_data.items():
            member_data[key] = data[cluster_mask]
        
        stable_haloes.append({
            'cluster_id': cluster_id,
            'cluster_size': cluster_size,
            'mean_position': mean_position,
            'mean_mass': mean_mass,
            'position_std': position_std,
            'mass_std': mass_std,
            'members': cluster_provenance,
            'member_data': member_data
        })
    
    return stable_haloes, positions, masses, halo_provenance, cluster_labels
