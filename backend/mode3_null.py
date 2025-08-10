import numpy as np
from backend.config_loader import load_config
from backend.utils import ensure_output_dir, save_clusters_to_hdf5
from backend.common_clustering import find_stable_haloes
import os
from pymanticore.swift_analysis import SOAPData

def load_random_control_data(config):
    """Load random control simulations with multiple random observer samplings"""
    
    # Calculate total virtual mcmc samples
    n_real_sims = config.mode3.mcmc_end - config.mode3.mcmc_start + 1
    total_virtual_mcmc = n_real_sims * config.mode3.num_samplings
    
    mcmc_data = {}
    
    # Common reference point (box center)
    box_center = np.array([config.global_config.boxsize/2, config.global_config.boxsize/2, config.global_config.boxsize/2])
    
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
    
    for virtual_mcmc_id in range(total_virtual_mcmc):
        # Map virtual mcmc_id to (simulation_id, sampling_id)
        sim_id = config.mode3.mcmc_start + (virtual_mcmc_id // config.mode3.num_samplings)
        sampling_id = virtual_mcmc_id % config.mode3.num_samplings
        
        # Generate random observer coordinates
        random_observer = np.random.uniform(0, config.global_config.boxsize, 3)
        
        # Load the simulation
        filename = os.path.join(config.mode3.basedir, f"mcmc_{sim_id}/soap/SOAP_uncompressed/HBTplus/halo_properties_0077.hdf5")
        soap_data = SOAPData(filename, radius_cut=config.mode3.radius_cut)
        soap_data.load_groups(properties=to_load, only_centrals=True)
        soap_data.set_observer(random_observer, skip_redshift=True)
        
        # Don't want to keep redshift
        del soap_data.data["redshift"]
        
        # Apply M200 mass cut
        m200_masses = soap_data.data['SO/200_crit/TotalMass']
        m200_mass_mask = m200_masses >= config.mode3.m200_mass_cut
        
        filtered_data = {}
        for key, value in soap_data.data.items():
            if isinstance(value, np.ndarray) and len(value) == len(m200_masses):
                filtered_data[key] = value[m200_mass_mask]
            else:
                filtered_data[key] = value
        
        # Transform positions back to common reference frame (box center)
        observer_relative_positions = filtered_data['SO/200_crit/CentreOfMass']
        box_coordinates = observer_relative_positions - random_observer
        centered_positions = box_coordinates + box_center
        filtered_data['SO/200_crit/CentreOfMass'] = centered_positions
        
        # Transform other position-based properties if they exist
        for key in ['BoundSubhalo/CentreOfMass', 'SO/500_crit/CentreOfMass']:
            if key in filtered_data:
                observer_relative = filtered_data[key]
                box_coords = observer_relative + random_observer
                filtered_data[key] = box_coords - box_center
        
        mcmc_data[virtual_mcmc_id] = filtered_data
        
        print(f"Loaded virtual MCMC {virtual_mcmc_id} (sim {sim_id}, sample {sampling_id}): {len(mcmc_data[virtual_mcmc_id]['SO/200_crit/TotalMass'])} haloes, observer=[{random_observer[0]:.1f}, {random_observer[1]:.1f}, {random_observer[2]:.1f}]")
    
    return mcmc_data

def run_mode3(config_path="config.toml", output_dir="output"):
    config = load_config(config_path)
    ensure_output_dir(output_dir)
    
    print("Running Mode 3: Random control simulation analysis")
    print(f"Control simulations: {config.mode3.mcmc_start} to {config.mode3.mcmc_end}")
    print(f"Samplings per simulation: {config.mode3.num_samplings}")
    print(f"Total virtual samples: {(config.mode3.mcmc_end - config.mode3.mcmc_start + 1) * config.mode3.num_samplings}")
    print(f"M200 mass cut: {config.mode3.m200_mass_cut:.2e} M☉")
    print(f"Radius cut: {config.mode3.radius_cut} Mpc")
    print(f"Box size: {config.global_config.boxsize} Mpc")
    
    print("Loading random control data with multiple observer samplings...")
    mcmc_data = load_random_control_data(config)
    
    total_haloes = 0
    for virtual_mcmc_id, data in mcmc_data.items():
        n_haloes = len(data['SO/200_crit/TotalMass'])
        total_haloes += n_haloes
    
    print(f"Total haloes across all virtual samples: {total_haloes}")
    
    print("Finding clusters in random control simulations...")
    stable_haloes, positions, m200_masses, halo_provenance, cluster_labels = find_stable_haloes(
        mcmc_data, config, eps=config.mode3.eps, min_samples=config.mode3.min_samples
    )
    
    print(f"Found {len(stable_haloes)} clusters in random control simulations")
    
    if len(stable_haloes) > 0:
        sorted_haloes = sorted(stable_haloes, key=lambda x: x['cluster_size'], reverse=True)
        
        print("\nTop 10 clusters by size:")
        for i, halo in enumerate(sorted_haloes[:10]):
            print(f"  Cluster {i}: size={halo['cluster_size']}, "
                  f"M200={halo['mean_m200_mass']:.2e}, "
                  f"position=[{halo['mean_position'][0]:.1f}, {halo['mean_position'][1]:.1f}, {halo['mean_position'][2]:.1f}]")
        
        # Print summary statistics
        cluster_sizes = [cluster['cluster_size'] for cluster in stable_haloes]
        print(f"\nCluster size statistics:")
        print(f"  Total clusters: {len(cluster_sizes)}")
        print(f"  Mean size: {np.mean(cluster_sizes):.2f}")
        print(f"  Median size: {np.median(cluster_sizes):.2f}")
        print(f"  Max size: {max(cluster_sizes)}")
        print(f"  Size distribution: {np.bincount(cluster_sizes)[1:]}")
    
    print("Saving random control clusters to HDF5...")
    save_clusters_to_hdf5(stable_haloes, positions, m200_masses, halo_provenance, cluster_labels, config, output_dir, filename="random_control_clusters.h5")
    print("Mode 3 complete!")

if __name__ == '__main__':
    run_mode3()
