from mpi4py import MPI
import numpy as np
import os
from pymanticore.swift_analysis import SOAPData
from collections import defaultdict
from backend.config_loader import load_config
from backend.utils import load_clusters_from_hdf5, save_halo_traces_to_hdf5

class HaloTracer:
    def __init__(self, basedir, observer_coords, rank=0):
        self.basedir = basedir
        self.observer_coords = observer_coords
        self.rank = rank
        
        # Extended property list
        self.to_load = [
            "BoundSubhalo/TotalMass",
            "BoundSubhalo/CentreOfMass", 
            "BoundSubhalo/CentreOfMassVelocity",
            "SOAP/ProgenitorIndex",
            "SOAP/DescendantIndex",
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
    
    def _load_snapshot(self, mcmc_id, snap_num):
        """Load a single snapshot with validation data"""
        filename = os.path.join(self.basedir, f"mcmc_{mcmc_id}/soap/SOAP_uncompressed/HBTplus/halo_properties_{snap_num:04d}.hdf5")
        
        if snap_num >= 77:
            raise ValueError("Should not be loading final snapshot during this phase")
            #soap_data = SOAPData(filename, mass_cut=1e15, radius_cut=300)
            #soap_data.load_groups(properties=self.to_load, only_centrals=True)
        else:
            soap_data = SOAPData(filename)
            soap_data.load_groups(properties=self.to_load, only_centrals=False)
        
        soap_data.set_observer(self.observer_coords, skip_redshift=True)
        
        # Remove redshift
        del soap_data.data["redshift"]

        return soap_data.data.copy()
    
    def _validate_links(self, current_snap, prev_snap, current_indices, mcmc_id):
        """Validate progenitor-descendant links between snapshots"""
        validation_errors = 0
        
        for j, current_idx in enumerate(current_indices):
            if current_idx < 0 or current_idx >= len(prev_snap['BoundSubhalo/CentreOfMass']):
                continue
                
            prog_idx = current_snap['SOAP/ProgenitorIndex'][j] if j < len(current_snap['SOAP/ProgenitorIndex']) else -1
            
            if prog_idx >= 0 and prog_idx < len(prev_snap['SOAP/DescendantIndex']):
                expected_descendant = prev_snap['SOAP/DescendantIndex'][prog_idx]
                if expected_descendant != j:
                    validation_errors += 1
                    print(f"  Validation error: MCMC {mcmc_id}, halo {j} -> prog {prog_idx}, but prog has descendant {expected_descendant}")
        
        if validation_errors > 0:
            print(f"  Found {validation_errors} validation errors for MCMC {mcmc_id}")
        
        return validation_errors
    
    def trace_haloes_for_mcmc(self, halo_list, mcmc_id, target_snapshot):
        n_snapshots = 77 - target_snapshot + 1
        n_haloes = len(halo_list)
        
        # Initialize history arrays for all properties
        property_histories = {}
        valid_flags = np.zeros((n_haloes, n_snapshots), dtype=bool)
        current_indices = np.array([h['progenitor_index'] for h in halo_list])
        
        # Get property keys from the first halo's data
        first_halo_data = {k: v for k, v in halo_list[0].items() if k not in ['mcmc_id', 'original_index', 'cluster_id', 'progenitor_index']}
        property_keys = list(first_halo_data.keys())
        
        # Initialize arrays for each property
        for key in property_keys:
            if isinstance(first_halo_data[key], np.ndarray) and len(first_halo_data[key].shape) > 0:
                if len(first_halo_data[key]) == 3:  # 3D vector
                    property_histories[key] = np.full((n_haloes, n_snapshots, 3), np.nan)
                else:
                    property_histories[key] = np.full((n_haloes, n_snapshots), np.nan)
            else:
                property_histories[key] = np.full((n_haloes, n_snapshots), np.nan)
        
        # Initialize snapshot 77 data
        for i, halo in enumerate(halo_list):
            if halo['progenitor_index'] >= 0:
                for key in property_keys:
                    if key in halo:
                        if isinstance(halo[key], np.ndarray) and len(halo[key]) == 3:
                            property_histories[key][i, 0] = halo[key]
                        else:
                            property_histories[key][i, 0] = halo[key]
                valid_flags[i, 0] = True
        
        # Process each snapshot backward
        for snap_idx, snap_num in enumerate(range(76, target_snapshot - 1, -1)):
            print(f"  Rank {self.rank}: Processing MCMC {mcmc_id}, snapshot {snap_num}")
            
            active_mask = current_indices >= 0
            if not np.any(active_mask):
                break
            
            snapshot_data = self._load_snapshot(mcmc_id, snap_num)
            
            for i in range(n_haloes):
                if not active_mask[i]:
                    continue
                    
                current_idx = current_indices[i]
                
                if current_idx >= len(snapshot_data['BoundSubhalo/CentreOfMass']):
                    current_indices[i] = -1
                    continue
                
                # Store all properties for this halo at this snapshot
                for key in property_keys:
                    if key in snapshot_data:
                        data_value = snapshot_data[key][current_idx]
                        if isinstance(data_value, np.ndarray) and len(data_value) == 3:
                            property_histories[key][i, snap_idx + 1] = data_value
                        else:
                            property_histories[key][i, snap_idx + 1] = data_value
                
                valid_flags[i, snap_idx + 1] = True
                
                if snap_num > target_snapshot:
                    progenitor_idx = snapshot_data['SOAP/ProgenitorIndex'][current_idx]
                    current_indices[i] = progenitor_idx if progenitor_idx >= 0 else -1
                else:
                    current_indices[i] = -1
        
        # Package results
        results = {}
        for i, halo in enumerate(halo_list):
            valid_snaps = valid_flags[i, :]
            if np.sum(valid_snaps) > 1:
                valid_snapshots = np.array([77 - j for j in range(n_snapshots)])[valid_snaps]
                
                halo_key = f"mcmc_{mcmc_id}_halo_{halo['original_index']}"
                results[halo_key] = {
                    'mcmc_id': mcmc_id,
                    'original_index': halo['original_index'],
                    'cluster_id': halo['cluster_id'],
                    'snapshots': valid_snapshots
                }
                
                # Add all property histories
                for key in property_keys:
                    if key in property_histories:
                        results[halo_key][key] = property_histories[key][i, valid_snaps]
        
        return results

def extract_target_haloes(clusters, min_cluster_size):
    mcmc_haloes = defaultdict(list)
    
    for cluster in clusters:
        if cluster['cluster_size'] < min_cluster_size:
            continue
            
        for i, member in enumerate(cluster['members']):
            mcmc_id = member['mcmc_id']
            original_index = member['original_index']
            
            # Build halo data dictionary with all available properties
            halo_data = {
                'mcmc_id': mcmc_id,
                'original_index': original_index,
                'cluster_id': cluster['cluster_id']
            }
            
            # Add all member data properties
            for key, data in cluster['member_data'].items():
                halo_data[key] = data[i]
            
            # Extract progenitor index specifically
            if 'SOAP/ProgenitorIndex' in cluster['member_data']:
                halo_data['progenitor_index'] = cluster['member_data']['SOAP/ProgenitorIndex'][i]
            else:
                halo_data['progenitor_index'] = -1
            
            mcmc_haloes[mcmc_id].append(halo_data)
    
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

def run_mode2(config_path="config.toml", output_dir="output"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    config = load_config(config_path)
    
    if rank == 0:
        print("Loading clusters from Mode 1...")
        # Convert eps to filename format (2.5 -> "2p5")
        eps_str = str(config.mode1.eps).replace('.', 'p')
        filename = f"clusters_eps_{eps_str}_min_samples_{config.mode1.min_samples}.h5"

        clusters, cluster_metadata = load_clusters_from_hdf5(output_dir, filename=filename, minimal=False)
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
        fname = f"halo_traces_eps_{str(config.mode1.eps).replace('.','p')}_min_samples_{config.mode1.min_samples}.h5"
        save_halo_traces_to_hdf5(combined_halo_traces, config, metadata_to_broadcast['cluster_metadata'], output_dir,
                filename=fname)
        print("Mode 2 complete!")

if __name__ == '__main__':
    run_mode2()
