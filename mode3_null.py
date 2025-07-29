# mode3_null.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from config_loader import load_config
from utils import ensure_output_dir
from common_clustering import load_data_with_radius_filter, find_stable_haloes

def plot_cluster_size_distribution(stable_haloes, config):
    """Plot and save cluster size distribution"""
    if len(stable_haloes) == 0:
        print("No clusters found for distribution plot")
        return
    
    cluster_sizes = [cluster['cluster_size'] for cluster in stable_haloes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(cluster_sizes, bins=range(1, max(cluster_sizes) + 2), alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Cluster Size')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Cluster Size Distribution\n(Total clusters: {len(stable_haloes)})')
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.hist(cluster_sizes, bins=range(1, max(cluster_sizes) + 2), alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Cluster Size')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Cluster Size Distribution (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.global_config.output_dir, 'null_cluster_size_distribution.png'), 
                dpi=config.mode3.figure_dpi, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nCluster size statistics:")
    print(f"  Total clusters: {len(cluster_sizes)}")
    print(f"  Mean size: {np.mean(cluster_sizes):.2f}")
    print(f"  Median size: {np.median(cluster_sizes):.2f}")
    print(f"  Max size: {max(cluster_sizes)}")
    print(f"  Size distribution: {np.bincount(cluster_sizes)[1:]}")

def run_mode3():
    config = load_config()
    ensure_output_dir(config)
    
    print("Running Mode 3: Null clustering analysis")
    print(f"Radius range: {config.mode3.radius_inner} to {config.mode3.radius_outer} Mpc")
    
    print("Loading MCMC data with radius filtering...")
    mcmc_data = load_data_with_radius_filter(config, 
                                           radius_inner=config.mode3.radius_inner,
                                           radius_outer=config.mode3.radius_outer)
    
    print(f"Loaded data from MCMC samples {config.mode1.mcmc_start} to {config.mode1.mcmc_end}")
    total_haloes = 0
    for mcmc_id, data in mcmc_data.items():
        n_haloes = len(data['BoundSubhalo/TotalMass'])
        print(f"MCMC {mcmc_id}: {n_haloes} haloes")
        total_haloes += n_haloes
    
    print(f"Total haloes in radius range: {total_haloes}")
    
    print("Finding clusters in null region...")
    stable_haloes, positions, masses, halo_provenance, cluster_labels = find_stable_haloes(mcmc_data, config)
    
    print(f"Found {len(stable_haloes)} clusters in null region")
    
    if len(stable_haloes) > 0:
        sorted_haloes = sorted(stable_haloes, key=lambda x: x['cluster_size'], reverse=True)
        
        print("\nTop 10 clusters by size:")
        for i, halo in enumerate(sorted_haloes[:10]):
            print(f"  Cluster {i}: size={halo['cluster_size']}, "
                  f"mass={halo['mean_mass']:.2e}, "
                  f"position=[{halo['mean_position'][0]:.1f}, {halo['mean_position'][1]:.1f}, {halo['mean_position'][2]:.1f}]")
    
    print("Creating cluster size distribution plot...")
    plot_cluster_size_distribution(stable_haloes, config)
    
    print("Mode 3 complete!")

if __name__ == '__main__':
    run_mode3()
