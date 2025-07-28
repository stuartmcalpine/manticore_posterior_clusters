# mode3_plot.py
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from config_loader import load_config
from utils import load_clusters_from_hdf5, load_halo_traces_from_hdf5

def plot_cluster_temporal_evolution(config, halo_traces, trace_metadata):
    output_dir = config.global_config.output_dir
    
    cluster_traces = defaultdict(list)
    for halo_key, trace_data in halo_traces.items():
        cluster_id = trace_data['cluster_id']
        cluster_traces[cluster_id].append(trace_data)
    
    for cluster_id, traces in cluster_traces.items():
        if len(traces) == 0:
            continue
            
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        
        # Generate colors for each trajectory
        colors = plt.cm.tab20(np.linspace(0, 1, len(traces)))
        if len(traces) > 20:
            colors = plt.cm.gist_ncar(np.linspace(0, 1, len(traces)))
        
        for i, trace_data in enumerate(traces):
            positions = trace_data['positions']
            snapshots = trace_data['snapshots']
            
            # Plot trajectory line with reduced visibility
            ax.plot(positions[:, 0], positions[:, 1], '-', 
                   alpha=0.4, linewidth=0.8, color=colors[i])
            
            # Plot start and end points with higher visibility
            ax.scatter(positions[0, 0], positions[0, 1], 
                      c='red', s=60, marker='s', alpha=0.9, 
                      edgecolors='darkred', linewidth=0.5, zorder=10)
            ax.scatter(positions[-1, 0], positions[-1, 1], 
                      c='blue', s=60, marker='o', alpha=0.9,
                      edgecolors='darkblue', linewidth=0.5, zorder=10)
        
        # Legend with larger, more visible markers
        ax.scatter([], [], c='red', s=100, marker='s', 
                  label=f'Snapshot 77 (start)', edgecolors='darkred', linewidth=0.5)
        ax.scatter([], [], c='blue', s=100, marker='o', 
                  label=f'Snapshot {config.mode2.target_snapshot} (end)', 
                  edgecolors='darkblue', linewidth=0.5)
        ax.plot([], [], 'gray', alpha=0.6, linewidth=0.8, label='Halo trajectories')
        
        ax.set_xlabel('X (Mpc)', fontsize=12)
        ax.set_ylabel('Y (Mpc)', fontsize=12)
        ax.set_title(f'Cluster {cluster_id} Halo Trajectories\nSnapshot {config.mode2.target_snapshot} to 77 (n={len(traces)})', 
                    fontsize=14, pad=20)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Set background color to help with line visibility
        ax.set_facecolor('#f8f8f8')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_temporal_evolution.png'), 
                   dpi=config.mode3.figure_dpi, bbox_inches='tight', facecolor='white')
        plt.close()

def plot_largest_cluster_diagnostic(config):
    clusters, _ = load_clusters_from_hdf5(config)
    
    if len(clusters) == 0:
        print("No clusters found for diagnostic plot")
        return
        
    largest_cluster = max(clusters, key=lambda x: x['cluster_size'])
    largest_cluster_center = largest_cluster['mean_position']
    largest_cluster_id = largest_cluster['cluster_id']
    
    all_positions = []
    all_cluster_ids = []
    
    for cluster in clusters:
        positions = cluster['member_data']['positions']
        cluster_id = cluster['cluster_id']
        all_positions.extend(positions)
        all_cluster_ids.extend([cluster_id] * len(positions))
    
    all_positions = np.array(all_positions)
    all_cluster_ids = np.array(all_cluster_ids)
    
    distances = np.linalg.norm(all_positions - largest_cluster_center, axis=1)
    within_15mpc = distances <= 15.0
    
    nearby_positions = all_positions[within_15mpc]
    nearby_cluster_labels = all_cluster_ids[within_15mpc]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    unique_labels = np.unique(nearby_cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = nearby_cluster_labels == label
        if label == largest_cluster_id:
            ax.scatter(nearby_positions[mask, 0], nearby_positions[mask, 1], 
                      c='red', s=50, alpha=0.8, label=f'Largest Cluster (ID {label})')
        else:
            ax.scatter(nearby_positions[mask, 0], nearby_positions[mask, 1], 
                      c=[colors[i]], s=30, alpha=0.6, label=f'Cluster {label}')
    
    ax.scatter(largest_cluster_center[0], largest_cluster_center[1], 
              c='black', s=200, marker='*', label='Cluster Center')
    
    circle = plt.Circle((largest_cluster_center[0], largest_cluster_center[1]), 
                       7.5, fill=False, color='black', linestyle='--', linewidth=2, label='7.5 Mpc threshold')
    ax.add_patch(circle)
    
    ax.set_xlabel('X (Mpc)')
    ax.set_ylabel('Y (Mpc)')
    ax.set_title(f'Haloes within 15 Mpc of Largest Cluster (X-Y projection)\nCluster Size: {largest_cluster["cluster_size"]}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.global_config.output_dir, 'largest_cluster_diagnostic.png'), 
               dpi=config.mode3.figure_dpi, bbox_inches='tight')
    plt.close()

def plot_all_significant_clusters(config):
    clusters, _ = load_clusters_from_hdf5(config)
    
    significant_clusters = [h for h in clusters if h['cluster_size'] >= config.mode3.min_members_plot]
    
    if len(significant_clusters) == 0:
        print(f"No clusters found with at least {config.mode3.min_members_plot} members")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(significant_clusters)))
    
    for i, cluster in enumerate(significant_clusters):
        positions = cluster['member_data']['positions']
        cluster_size = cluster['cluster_size']
        cluster_id = cluster['cluster_id']
        
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c=[colors[i]], s=40, alpha=0.7, 
                  label=f'Cluster {cluster_id} (n={cluster_size})')
    
    ax.set_xlabel('X (Mpc)')
    ax.set_ylabel('Y (Mpc)')
    ax.set_title(f'All Clusters with ≥{config.mode3.min_members_plot} Members\nTotal Significant Clusters: {len(significant_clusters)}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.global_config.output_dir, 'all_significant_clusters.png'), 
               dpi=config.mode3.figure_dpi, bbox_inches='tight')
    plt.close()
    
    print(f"\nSignificant clusters (≥{config.mode3.min_members_plot} members):")
    for cluster in significant_clusters:
        print(f"  Cluster {cluster['cluster_id']}: {cluster['cluster_size']} members")

def run_mode3():
    config = load_config()
    
    print("Loading data for plotting...")
    
    print("Creating cluster diagnostic plots...")
    plot_largest_cluster_diagnostic(config)
    plot_all_significant_clusters(config)
    
    try:
        halo_traces, trace_metadata = load_halo_traces_from_hdf5(config)
        print(f"Loaded {len(halo_traces)} halo traces")
        
        print("Creating temporal evolution plots...")
        plot_cluster_temporal_evolution(config, halo_traces, trace_metadata)
        
    except FileNotFoundError:
        print("No halo trace data found. Run Mode 2 first to generate temporal evolution plots.")
    
    print("Mode 3 complete! Check output directory for plots.")

if __name__ == '__main__':
    run_mode3()
