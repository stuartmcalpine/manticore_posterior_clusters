# mode2_trace.py
from mpi4py import MPI
import numpy as np
import os
from pymanticore.swift_analysis import SOAPData
from collections import defaultdict
from config_loader import load_config
from utils import load_clusters_from_hdf5, save_halo_traces_to_hdf5

class HaloTracer:
    def __init__(self, basedir, observer_coords, rank=0):
        self.basedir = basedir
        self.observer_coords = observer_coords
        self.rank = rank
    
    def _load_snapshot(self, mcmc_id, snap_num):
        filename = os.path.join(self.basedir, f"mcmc_{mcmc_id}/soap/SOAP_uncompressed/HBTplus/halo_properties_{snap_num:04d}.hdf5")
        
        if snap_num == 77:
            soap_data = SOAPData(filename, mass_cut=1e15, radius_cut=300)
            soap_data.load_groups(properties=["BoundSubhalo/CentreOfMass", "BoundSubhalo/TotalMass", "SOAP/ProgenitorIndex", "SOAP/DescendantIndex"], only_centrals=True)
        else:
            soap_data = SOAPData(filename)
            soap_data.load_groups(properties=["BoundSubhalo/CentreOfMass", "BoundSubhalo/TotalMass", "SOAP/ProgenitorIndex", "SOAP/DescendantIndex"], only_centrals=False)
        
        soap_data.set_observer(self.observer_coords, skip_redshift=True)
        
        return {
            'positions': soap_data.data['BoundSubhalo/CentreOfMass'],
            'masses': soap_data.data['BoundSubhalo/TotalMass'],
            'progenitor_indices': soap_data.data['SOAP/ProgenitorIndex'],
            'descendant_indices': soap_data.data['SOAP/DescendantIndex']
        }
    
    def trace_haloes_for_mcmc(self, halo_list, mcmc_id, target_snapshot):
        n_snapshots = 77 - target_snapshot + 1
        n_haloes = len(halo_list)
        
        position_histories = np.full((n_haloes, n_snapshots, 3), np.nan)
        mass_histories = np.full((n_haloes, n_snapshots), np.nan)
        valid_flags = np.zeros((n_haloes, n_snapshots), dtype=bool)
        current_indices = np.array([h['progenitor_index'] for h in halo_list])
        
        # Initialize snapshot 77 data
        for i, halo in enumerate(halo_list):
            if halo['progenitor_index'] >= 0:
                position_histories[i, 0] = halo['position']
                mass_histories[i, 0] = halo['mass']
                valid_flags[i, 0] = True
        
        # Process each snapshot backward
        for snap_idx, snap_num in enumerate(range(76, target_snapshot - 1, -1)):
            print(f"  Rank {self.rank}: Processing MCMC {mcmc_id}, snapshot {snap_num}")
            
            active_mask = current_indices >= 0
            if not np.any(active_mask):
                break
            
            snapshot_data = self._load_snapshot(mcmc_id, snap_num)
            
            pos_data = snapshot_data['positions']
            mass_data = snapshot_data['masses']
            prog_data = snapshot_data['progenitor_indices']
            
            for i in range(n_haloes):
                if not active_mask[i]:
                    continue
                    
                current_idx = current_indices[i]
                
                if current_idx >= len(pos_data):
                    current_indices[i] = -1
                    continue
                
                position_histories[i, snap_idx + 1] = pos_data[current_idx]
                mass_histories[i, snap_idx + 1] = mass_data[current_idx]
                valid_flags[i, snap_idx + 1] = True
                
                if snap_num > target_snapshot:
                    progenitor_idx = prog_data[current_idx]
                    current_indices[i] = progenitor_idx if progenitor_idx >= 0 else -1
                else:
                    current_indices[i] = -1
        
        # Package results
        results = {}
        for i, halo in enumerate(halo_list):
            valid_snaps = valid_flags[i, :]
            if np.sum(valid_snaps) > 1:
                valid_positions = position_histories[i, valid_snaps]
                valid_masses = mass_histories[i, valid_snaps]
                valid_snapshots = np.array([77 - j for j in range(n_snapshots)])[valid_snaps]
                
                halo_key = f"mcmc_{mcmc_id}_halo_{halo['original_index']}"
                results[halo_key] = {
                    'mcmc_id': mcmc_id,
                    'original_index': halo['original_index'],
                    'cluster_id': halo['cluster_id'],
                    'positions': valid_positions,
                    'masses': valid_masses,
                    'snapshots': valid_snapshots
                }
        
        return results

def extract_target_haloes(clusters, min_cluster_size):
    mcmc_haloes = defaultdict(list)
    
    for cluster in clusters:
        if cluster['cluster_size'] < min_cluster_size:
            continue
            
        for i, member in enumerate(cluster['members']):
            mcmc_id = member['mcmc_id']
            original_index = member['original_index']
            
            mcmc_haloes[mcmc_id].append({
                'original_index': original_index,
                'cluster_id': cluster['cluster_id'],
                'position': cluster['member_data']['positions'][i],
                'mass': cluster['member_data']['masses'][i],
                'progenitor_index': cluster['member_data']['progenitor_indices'][i]
            })
    
    return mcmc_haloes

def distribute_mcmc_work(mcmc_haloes, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        mcmc_ids = list(mcmc_haloes.keys())
        work_assignments = defaultdict(dict)
        
        for i, mcmc_id in enumerate(mcmc_ids):
            assigned_rank = i % size
            work_assignments[assigned_rank][mcmc_id] = mcmc_haloes[mcmc_id]
        
        for target_rank in range(size):
            if target_rank == 0:
                my_work = work_assignments[0]
            else:
                comm.send(work_assignments[target_rank], dest=target_rank, tag=0)
        
        return my_work
    else:
        my_work = comm.recv(source=0, tag=0)
        return my_work

def run_mode2():
    #from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    config = load_config()
    
    if rank == 0:
        print("Loading clusters from Mode 1...")
        clusters, cluster_metadata = load_clusters_from_hdf5(config)
        print(f"Loaded {len(clusters)} clusters")
        
        print(f"Extracting haloes from clusters with >= {config.mode2.min_cluster_size} members...")
        mcmc_haloes = extract_target_haloes(clusters, config.mode2.min_cluster_size)
        
        total_haloes = sum(len(haloes) for haloes in mcmc_haloes.values())
        print(f"Found {total_haloes} haloes across {len(mcmc_haloes)} MCMC samples to trace")
        
        if total_haloes == 0:
            print("No haloes to trace. Exiting.")
            return
        
        observer_coords = [float(x) for x in cluster_metadata['observer_coords']]
        basedir = cluster_metadata['basedir'].decode() if isinstance(cluster_metadata['basedir'], bytes) else cluster_metadata['basedir']
        
        metadata_to_broadcast = {
            'observer_coords': observer_coords,
            'basedir': basedir,
            'cluster_metadata': cluster_metadata
        }
    else:
        mcmc_haloes = None
        metadata_to_broadcast = None
    
    # Broadcast metadata to all ranks
    metadata_to_broadcast = comm.bcast(metadata_to_broadcast, root=0)
    
    # Distribute work
    my_mcmc_work = distribute_mcmc_work(mcmc_haloes, comm)
    
    print(f"Rank {rank}: Assigned {len(my_mcmc_work)} MCMC samples to process")
    
    # Each rank processes their assigned MCMC samples
    tracer = HaloTracer(metadata_to_broadcast['basedir'], metadata_to_broadcast['observer_coords'], rank)
    my_halo_traces = {}
    
    for mcmc_id, halo_list in my_mcmc_work.items():
        print(f"Rank {rank}: Tracing {len(halo_list)} haloes for MCMC {mcmc_id}")
        
        try:
            mcmc_traces = tracer.trace_haloes_for_mcmc(halo_list, mcmc_id, config.mode2.target_snapshot)
            my_halo_traces.update(mcmc_traces)
            print(f"Rank {rank}: Successfully traced {len(mcmc_traces)} haloes for MCMC {mcmc_id}")
            
        except Exception as e:
            print(f"Rank {rank}: Error tracing MCMC {mcmc_id}: {e}")
            continue
    
    print(f"Rank {rank}: Completed tracing {len(my_halo_traces)} total haloes")
    
    # Gather all results to root
    all_traces = comm.gather(my_halo_traces, root=0)
    
    if rank == 0:
        combined_halo_traces = {}
        for rank_traces in all_traces:
            combined_halo_traces.update(rank_traces)
        
        print(f"Combined {len(combined_halo_traces)} halo traces from all ranks")
        print("Saving halo traces to HDF5...")
        save_halo_traces_to_hdf5(combined_halo_traces, config, metadata_to_broadcast['cluster_metadata'])
        print("Mode 2 complete!")

if __name__ == '__main__':
    run_mode2()
