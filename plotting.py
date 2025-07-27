import numpy as np
import matplotlib.pyplot as plt
from merger_tree import MergerTreeTracer

def plot_cluster_temporal_evolution(basedir, stable_haloes, target_snapshot=30, observer_coords=[500,500,500]):
    sorted_clusters = sorted(stable_haloes, key=lambda x: x['cluster_size'], reverse=True)[:1]
    
    merger_tracer = MergerTreeTracer(basedir, observer_coords=observer_coords)
    
    for cluster_idx, cluster in enumerate(sorted_clusters):
        traced_results = merger_tracer.trace_cluster_members_batch(
            cluster['members'], 
            cluster['member_data'], 
            target_snapshot
        )
        
        if len(traced_results) == 0:
            print(f"No successful traces found for cluster {cluster_idx}")
            continue
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        for key, result in traced_results.items():
            positions = result['positions']
            snapshots = result['snapshots']
            
            ax.plot(positions[:, 0], positions[:, 1], 'o-', alpha=0.7, linewidth=1, markersize=3)
            
            ax.scatter(positions[0, 0], positions[0, 1], c='red', s=80, marker='s', alpha=0.8)
            ax.scatter(positions[-1, 0], positions[-1, 1], c='blue', s=80, marker='o', alpha=0.8)
        
        ax.scatter([], [], c='red', s=80, marker='s', label=f'Snapshot 77 (start)')
        ax.scatter([], [], c='blue', s=80, marker='o', label=f'Snapshot {target_snapshot} (end)')
        ax.plot([], [], 'gray', alpha=0.7, label='Halo trajectories')
        
        ax.set_xlabel('X (Mpc)')
        ax.set_ylabel('Y (Mpc)')
        ax.set_title(f'Cluster {cluster_idx+1} Full Halo Trajectories\nSnapshot {target_snapshot} to 77 (n={len(traced_results)})')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'cluster_{cluster_idx+1}_temporal_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()

def plot_largest_cluster_diagnostic(positions, masses, halo_provenance, cluster_labels, stable_haloes):
    largest_cluster = max(stable_haloes, key=lambda x: x['cluster_size'])
    largest_cluster_center = largest_cluster['mean_position']
    largest_cluster_id = largest_cluster['cluster_id']
    
    distances = np.linalg.norm(positions - largest_cluster_center, axis=1)
    within_15mpc = distances <= 15.0
    
    nearby_positions = positions[within_15mpc]
    nearby_cluster_labels = cluster_labels[within_15mpc]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    unique_labels = np.unique(nearby_cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = nearby_cluster_labels == label
        if label == largest_cluster_id:
            ax.scatter(nearby_positions[mask, 0], nearby_positions[mask, 1], 
                      c='red', s=50, alpha=0.8, label=f'Largest Cluster (ID {label})')
        elif label == -1:
            ax.scatter(nearby_positions[mask, 0], nearby_positions[mask, 1], 
                      c='gray', s=20, alpha=0.4, marker='x', label='Noise')
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
    plt.savefig('largest_cluster_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_all_significant_clusters(positions, masses, halo_provenance, cluster_labels, stable_haloes, min_members=10):
    significant_clusters = [h for h in stable_haloes if h['cluster_size'] >= min_members]
    
    if len(significant_clusters) == 0:
        print(f"No clusters found with at least {min_members} members")
        return
    
    significant_cluster_ids = [cluster['cluster_id'] for cluster in significant_clusters]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(significant_cluster_ids)))
    
    for i, cluster_id in enumerate(significant_cluster_ids):
        mask = cluster_labels == cluster_id
        cluster_positions = positions[mask]
        cluster_size = np.sum(mask)
        
        ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], 
                  c=[colors[i]], s=40, alpha=0.7, 
                  label=f'Cluster {cluster_id} (n={cluster_size})')
    
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        ax.scatter(positions[noise_mask, 0], positions[noise_mask, 1], 
                  c='gray', s=15, alpha=0.3, marker='x', label='Noise')
    
    ax.set_xlabel('X (Mpc)')
    ax.set_ylabel('Y (Mpc)')
    ax.set_title(f'All Clusters with ≥{min_members} Members\nTotal Significant Clusters: {len(significant_cluster_ids)}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('all_significant_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSignificant clusters (≥{min_members} members):")
    for cluster in significant_clusters:
        print(f"  Cluster {cluster['cluster_id']}: {cluster['cluster_size']} members")
