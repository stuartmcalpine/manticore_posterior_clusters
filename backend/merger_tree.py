import numpy as np
import os
from pymanticore.swift_analysis import SOAPData
from collections import defaultdict

class MergerTreeTracer:
    def __init__(self, basedir, observer_coords=[500,500,500]):
        self.basedir = basedir
        self.observer_coords = observer_coords
    
    def _load_snapshot(self, mcmc_id, snap_num):
        """Load a single snapshot with validation data"""
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
    
    def _validate_links(self, current_snap, prev_snap, current_indices, mcmc_id):
        """Validate progenitor-descendant links between snapshots"""
        validation_errors = 0
        
        for j, current_idx in enumerate(current_indices):
            if current_idx < 0 or current_idx >= len(prev_snap['positions']):
                continue
                
            prog_idx = current_snap['progenitor_indices'][j] if j < len(current_snap['progenitor_indices']) else -1
            
            if prog_idx >= 0 and prog_idx < len(prev_snap['descendant_indices']):
                expected_descendant = prev_snap['descendant_indices'][prog_idx]
                if expected_descendant != j:
                    validation_errors += 1
                    print(f"  Validation error: MCMC {mcmc_id}, halo {j} -> prog {prog_idx}, but prog has descendant {expected_descendant}")
        
        if validation_errors > 0:
            print(f"  Found {validation_errors} validation errors for MCMC {mcmc_id}")
        
        return validation_errors
    
    def trace_cluster_members_batch(self, cluster_members, cluster_data, target_snapshot=30):
        """Trace all members using 2D arrays per mcmc_id with validation"""
        
        # Group members by mcmc_id
        mcmc_groups = defaultdict(list)
        for i, member in enumerate(cluster_members):
            mcmc_groups[member['mcmc_id']].append({
                'original_index': member['original_index'],
                'array_index': i,
                'position': cluster_data['positions'][i],
                'mass': cluster_data['masses'][i],
                'progenitor_index': cluster_data['progenitor_indices'][i]
            })
        
        n_snapshots = 77 - target_snapshot + 1
        
        # Initialize arrays per mcmc_id
        position_arrays = {}
        mass_arrays = {}
        valid_arrays = {}
        current_indices = {}
        
        for mcmc_id, members in mcmc_groups.items():
            n_haloes = len(members)
            position_arrays[mcmc_id] = np.full((n_snapshots, n_haloes, 3), np.nan)
            mass_arrays[mcmc_id] = np.full((n_snapshots, n_haloes), np.nan)
            valid_arrays[mcmc_id] = np.zeros((n_snapshots, n_haloes), dtype=bool)
            current_indices[mcmc_id] = np.full(n_haloes, -1, dtype=int)
            
            # Fill snapshot 77 data
            for j, member in enumerate(members):
                if member['progenitor_index'] >= 0:
                    position_arrays[mcmc_id][0, j] = member['position']
                    mass_arrays[mcmc_id][0, j] = member['mass']
                    valid_arrays[mcmc_id][0, j] = True
                    current_indices[mcmc_id][j] = member['progenitor_index']
        
        print(f"Initialized arrays for {len(mcmc_groups)} MCMC samples")
        
        # Keep previous snapshot in memory for validation
        prev_snapshots = {}
        
        # Process each snapshot
        for snap_idx, snap_num in enumerate(range(76, target_snapshot - 1, -1)):
            print(f"Processing snapshot {snap_num} ({snap_idx + 1}/{n_snapshots - 1})")
            
            current_snapshots = {}
            
            for mcmc_id in mcmc_groups.keys():
                active_mask = current_indices[mcmc_id] >= 0
                if not np.any(active_mask):
                    continue
                
                # Load current snapshot
                current_snapshots[mcmc_id] = self._load_snapshot(mcmc_id, snap_num)
                
                # Validate links if we have previous snapshot
                if mcmc_id in prev_snapshots:
                    self._validate_links(prev_snapshots[mcmc_id], current_snapshots[mcmc_id], current_indices[mcmc_id], mcmc_id)
                
                pos_data = current_snapshots[mcmc_id]['positions']
                mass_data = current_snapshots[mcmc_id]['masses']
                prog_data = current_snapshots[mcmc_id]['progenitor_indices']
                
                # Update arrays for active haloes
                for j in range(len(mcmc_groups[mcmc_id])):
                    if not active_mask[j]:
                        continue
                        
                    current_idx = current_indices[mcmc_id][j]
                    
                    if current_idx >= len(pos_data):
                        current_indices[mcmc_id][j] = -1
                        continue
                    
                    # Store position and mass
                    position_arrays[mcmc_id][snap_idx + 1, j] = pos_data[current_idx]
                    mass_arrays[mcmc_id][snap_idx + 1, j] = mass_data[current_idx]
                    valid_arrays[mcmc_id][snap_idx + 1, j] = True
                    
                    # Update current index for next iteration
                    if snap_num > target_snapshot:
                        progenitor_idx = prog_data[current_idx]
                        current_indices[mcmc_id][j] = progenitor_idx if progenitor_idx >= 0 else -1
                    else:
                        current_indices[mcmc_id][j] = -1
            
            # Update previous snapshots for next iteration
            prev_snapshots = current_snapshots
        
        # Collect results
        results = {}
        
        for mcmc_id, members in mcmc_groups.items():
            for j, member in enumerate(members):
                valid_snaps = valid_arrays[mcmc_id][:, j]
                if np.sum(valid_snaps) > 1:
                    valid_positions = position_arrays[mcmc_id][valid_snaps, j]
                    valid_masses = mass_arrays[mcmc_id][valid_snaps, j]
                    valid_snapshots = np.array([77 - i for i in range(n_snapshots)])[valid_snaps]
                    
                    results[f"{mcmc_id}_{member['original_index']}"] = {
                        'mcmc_id': mcmc_id,
                        'original_index': member['original_index'],
                        'positions': valid_positions,
                        'masses': valid_masses,
                        'snapshots': valid_snapshots
                    }
        
        print(f"Successfully traced {len(results)} haloes")
        return results
