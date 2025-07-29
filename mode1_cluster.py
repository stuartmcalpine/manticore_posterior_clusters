# mode1_cluster.py (updated to use common functions)
import numpy as np
from config_loader import load_config
from utils import ensure_output_dir, save_clusters_to_hdf5
from common_clustering import load_data_with_radius_filter, find_stable_haloes

def run_mode1():
    config = load_config()
    ensure_output_dir(config)
    
    print("Loading MCMC data...")
    mcmc_data = load_data_with_radius_filter(config)
    
    print(f"Loaded data from MCMC samples {config.mode1.mcmc_start} to {config.mode1.mcmc_end}")
    for mcmc_id, data in mcmc_data.items():
        print(f"MCMC {mcmc_id}: {len(data['BoundSubhalo/TotalMass'])} haloes")
    
    print("Finding stable halo clusters...")
    stable_haloes, positions, masses, halo_provenance, cluster_labels = find_stable_haloes(mcmc_data, config)
    
    print(f"Found {len(stable_haloes)} stable halo clusters")
    
    sorted_haloes = sorted(stable_haloes, key=lambda x: x['cluster_size'], reverse=True)
    
    print("\nTop 10 stable haloes by cluster size:")
    for i, halo in enumerate(sorted_haloes[:10]):
        print(f"\nHalo {i}:")
        print(f"  Cluster size: {halo['cluster_size']}")
        print(f"  Mass mean: {halo['mean_mass']:.2e} ± {halo['mass_std']:.2e}")
        print(f"  Position mean: [{halo['mean_position'][0]:.1f}, {halo['mean_position'][1]:.1f}, {halo['mean_position'][2]:.1f}]")
        print(f"  Position std: [{halo['position_std'][0]:.1f}, {halo['position_std'][1]:.1f}, {halo['position_std'][2]:.1f}]")
    
    print("Saving clusters to HDF5...")
    save_clusters_to_hdf5(stable_haloes, positions, masses, halo_provenance, cluster_labels, config)
    print("Mode 1 complete!")

if __name__ == '__main__':
    run_mode1()
