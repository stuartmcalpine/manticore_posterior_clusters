import numpy as np
import os
from pymanticore.swift_analysis import SOAPData
from sklearn.cluster import DBSCAN

def load_data(basedir, mcmc_start, mcmc_end, mass_cut=1e15, radius_cut=300, observer_coords=[500,500,500]):
    mcmc_data = {}
    
    for mcmc_id in range(mcmc_start, mcmc_end + 1):
        filename = os.path.join(basedir, f"mcmc_{mcmc_id}/soap/SOAP_uncompressed/HBTplus/halo_properties_0077.hdf5")
        soap_data = SOAPData(filename, mass_cut=mass_cut, radius_cut=radius_cut)
        to_load = ["BoundSubhalo/TotalMass", "SO/200_crit/SORadius", "BoundSubhalo/CentreOfMass", "SOAP/ProgenitorIndex"]
        soap_data.load_groups(properties=to_load, only_centrals=True)
        soap_data.set_observer(observer_coords, skip_redshift=True)
        
        mcmc_data[mcmc_id] = soap_data.data.copy()
        
        print(f"Loaded MCMC step {mcmc_id}: {len(soap_data.data['BoundSubhalo/TotalMass'])} haloes")
    
    return mcmc_data

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

def find_stable_haloes(mcmc_data, eps=7.5, min_samples=5):
    positions, masses, progenitor_indices, halo_provenance = combine_haloes(mcmc_data)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(positions)
    
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
