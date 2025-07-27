#!/usr/bin/env python3
import argparse
from haloes_io import load_data, find_stable_haloes
from plotting import plot_largest_cluster_diagnostic, plot_all_significant_clusters, plot_cluster_temporal_evolution
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Process MCMC files')
    parser.add_argument('--basedir', help='Base directory containing MCMC files',
            default="/cosma7/data/dp004/rttw52/manticore_data/production_runs/v2b/resimulations/2MPP_MULTIBIN_N256_DES_V2/R1024")
    parser.add_argument('--mcmc_start', type=int, help='Starting MCMC file number',
            default=1)
    parser.add_argument('--mcmc_end', type=int, help='Ending MCMC file number',
            default=11)
    parser.add_argument('--eps', type=float, help='DBSCAN epsilon parameter (Mpc)',
            default=7.5)
    parser.add_argument('--min_samples', type=int, help='DBSCAN minimum samples parameter',
            default=5)
    parser.add_argument('--min_members', type=int, help='Minimum cluster size for significance plot',
            default=10)
    parser.add_argument('--target_snapshot', type=int, help='Target snapshot for temporal evolution',
            default=70)
    parser.add_argument('--observer_coords', nargs=3, type=float, help='Observer coordinates',
            default=[500, 500, 500])
    
    args = parser.parse_args()
    
    mcmc_data = load_data(args.basedir, args.mcmc_start, args.mcmc_end, observer_coords=args.observer_coords)
    
    print(f"Loaded data from MCMC samples {args.mcmc_start} to {args.mcmc_end}")
    for mcmc_id, data in mcmc_data.items():
        print(f"MCMC {mcmc_id}: {len(data['BoundSubhalo/TotalMass'])} haloes")
    
    stable_haloes, positions, masses, halo_provenance, cluster_labels = find_stable_haloes(mcmc_data, args.eps, args.min_samples)
    
    print(f"\nFound {len(stable_haloes)} stable halo clusters")
    
    sorted_haloes = sorted(stable_haloes, key=lambda x: x['cluster_size'], reverse=True)
    
    print("\nTop 10 stable haloes by cluster size:")
    for i, halo in enumerate(sorted_haloes[:10]):
        print(f"\nHalo {i}:")
        print(f"  Cluster size: {halo['cluster_size']}")
        print(f"  Mass mean: {halo['mean_mass']:.2e} ± {halo['mass_std']:.2e}")
        print(f"  Position mean: [{halo['mean_position'][0]:.1f}, {halo['mean_position'][1]:.1f}, {halo['mean_position'][2]:.1f}]")
        print(f"  Position std: [{halo['position_std'][0]:.1f}, {halo['position_std'][1]:.1f}, {halo['position_std'][2]:.1f}]")
    
    #plot_largest_cluster_diagnostic(positions, masses, halo_provenance, cluster_labels, stable_haloes)
    
    #plot_all_significant_clusters(positions, masses, halo_provenance, cluster_labels, stable_haloes, args.min_members)
    
    plot_cluster_temporal_evolution(args.basedir, stable_haloes, args.target_snapshot, args.observer_coords)

if __name__ == '__main__':
    main()
