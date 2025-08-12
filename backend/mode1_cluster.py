import numpy as np
from backend.config_loader import load_config
from backend.io import ensure_output_dir, save_clusters_to_hdf5
from backend.common_clustering import load_data_with_radius_filter, enhanced_find_stable_haloes, analyze_mass_distribution_in_clusters

def run_mode1(config_path="config.toml", output_dir="output", use_mass_filtering=True, 
              mass_outlier_threshold=0.3, use_mass_distance=True):
    config = load_config(config_path)
    ensure_output_dir(output_dir)
    
    print("Loading MCMC data...")
    mcmc_data = load_data_with_radius_filter(config)
    
    print(f"Loaded data from MCMC samples {config.mode1.mcmc_start} to {config.mode1.mcmc_end}")
    for mcmc_id, data in mcmc_data.items():
        print(f"MCMC {mcmc_id}: {len(data['SO/200_crit/TotalMass'])} haloes")
    
    print("Finding stable halo clusters...")
    if use_mass_filtering:
        print(f"Using M200 mass filtering with threshold {mass_outlier_threshold} dex")
        stable_haloes, positions, m200_masses, halo_provenance, cluster_labels = enhanced_find_stable_haloes(
            mcmc_data, config, 
            mass_outlier_threshold=mass_outlier_threshold,
            use_mass_distance=use_mass_distance
        )
    else:
        from backend.common_clustering import find_stable_haloes
        stable_haloes, positions, m200_masses, halo_provenance, cluster_labels = find_stable_haloes(mcmc_data, config)
    
    print(f"Found {len(stable_haloes)} stable halo clusters")
    
    sorted_haloes = sorted(stable_haloes, key=lambda x: x['cluster_size'], reverse=True)
    
    print("\nTop 10 stable haloes by cluster size:")
    for i, halo in enumerate(sorted_haloes[:10]):
        print(f"\nHalo {i}:")
        print(f"  Cluster size: {halo['cluster_size']}")
        print(f"  M200 mass mean: {halo['mean_m200_mass']:.2e} ± {halo['m200_mass_std']:.2e}")
        if 'mean_subhalo_mass' in halo:
            print(f"  Subhalo mass mean: {halo['mean_subhalo_mass']:.2e} ± {halo['subhalo_mass_std']:.2e}")
        if 'mean_m500' in halo:
            print(f"  M500 mass mean: {halo['mean_m500']:.2e} ± {halo['m500_std']:.2e}")
        if 'log_m200_mass_std' in halo:
            print(f"  Log-M200 mean: {halo['mean_log_m200_mass']:.2f} ± {halo['log_m200_mass_std']:.2f}")
            print(f"  Log-M200 range: {halo['log_m200_mass_range']:.2f} dex")
        print(f"  Position mean: [{halo['mean_position'][0]:.1f}, {halo['mean_position'][1]:.1f}, {halo['mean_position'][2]:.1f}]")
        print(f"  Position std: [{halo['position_std'][0]:.1f}, {halo['position_std'][1]:.1f}, {halo['position_std'][2]:.1f}]")
    
    if use_mass_filtering:
        analyze_mass_distribution_in_clusters(sorted_haloes)
    
    print("Saving clusters to HDF5...")
    fname = f"clusters_eps_{str(config.mode1.eps).replace('.','p')}_min_samples_{config.mode1.min_samples}.h5"
    save_clusters_to_hdf5(stable_haloes, positions, m200_masses, halo_provenance, cluster_labels, config, output_dir,
            filename=fname)
    print("Mode 1 complete!")

if __name__ == '__main__':
    run_mode1()
