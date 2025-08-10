import numpy as np
import os
from pymanticore.swift_analysis import SOAPData
from sklearn.cluster import DBSCAN

def enforce_mcmc_constraint(cluster_labels, positions, mcmc_ids):
    """Post-process clusters to ensure at most one halo per MCMC per cluster"""
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_mask = cluster_labels == cluster_id
        cluster_positions = positions[cluster_mask]
        cluster_mcmc_ids = mcmc_ids[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        # Find cluster center
        cluster_center = np.mean(cluster_positions, axis=0)
        
        # Group by MCMC ID
        unique_mcmc_ids = np.unique(cluster_mcmc_ids)
        
        for mcmc_id in unique_mcmc_ids:
            mcmc_mask = cluster_mcmc_ids == mcmc_id
            mcmc_indices_in_cluster = cluster_indices[mcmc_mask]
            
            if len(mcmc_indices_in_cluster) > 1:
                # Multiple halos from same MCMC in this cluster
                mcmc_positions = cluster_positions[mcmc_mask]
                
                # Find the one closest to cluster center
                distances_to_center = np.linalg.norm(mcmc_positions - cluster_center, axis=1)
                closest_idx = np.argmin(distances_to_center)
                
                # Keep only the closest one, mark others as noise
                for i, global_idx in enumerate(mcmc_indices_in_cluster):
                    if i != closest_idx:
                        cluster_labels[global_idx] = -1
    
    return cluster_labels

def enforce_mcmc_constraint_with_mass_filter(cluster_labels, positions, m200_masses, mcmc_ids, 
                                            mass_outlier_threshold=0.3, use_mass_distance=False):
    """
    Post-process clusters to ensure at most one halo per MCMC per cluster,
    with additional mass-based filtering to exclude outliers.
    
    Parameters:
    -----------
    cluster_labels : array
        Cluster assignments from DBSCAN
    positions : array
        Halo positions
    m200_masses : array
        Halo M200 masses (linear units)
    mcmc_ids : array
        MCMC sample IDs
    mass_outlier_threshold : float
        Threshold in log-mass units for outlier detection (default 0.3 = ~2x mass difference)
    use_mass_distance : bool
        Whether to use combined position+mass distance for selection
    """
    unique_clusters = np.unique(cluster_labels)
    
    # Work in log-mass space
    log_m200_masses = np.log10(m200_masses)
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_mask = cluster_labels == cluster_id
        cluster_positions = positions[cluster_mask]
        cluster_log_m200_masses = log_m200_masses[cluster_mask]
        cluster_mcmc_ids = mcmc_ids[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        # Calculate cluster statistics in log-mass space
        cluster_center = np.mean(cluster_positions, axis=0)
        cluster_mean_log_m200_mass = np.mean(cluster_log_m200_masses)
        cluster_log_m200_mass_std = np.std(cluster_log_m200_masses)
        
        # Identify mass outliers within the cluster (in log space)
        if cluster_log_m200_mass_std > 0:
            log_m200_mass_z_scores = np.abs((cluster_log_m200_masses - cluster_mean_log_m200_mass) / cluster_log_m200_mass_std)
            # Convert threshold to standard deviations if needed
            if mass_outlier_threshold < 1.0:
                # Interpret as direct log-mass difference threshold
                log_m200_mass_deviations = np.abs(cluster_log_m200_masses - cluster_mean_log_m200_mass)
                mass_outliers = log_m200_mass_deviations > mass_outlier_threshold
            else:
                # Interpret as number of standard deviations
                mass_outliers = log_m200_mass_z_scores > mass_outlier_threshold
            
            # Mark mass outliers as noise
            outlier_indices = cluster_indices[mass_outliers]
            for idx in outlier_indices:
                cluster_labels[idx] = -1
            
            # Update masks after removing outliers
            cluster_mask = cluster_labels == cluster_id
            cluster_positions = positions[cluster_mask]
            cluster_log_m200_masses = log_m200_masses[cluster_mask]
            cluster_mcmc_ids = mcmc_ids[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Recalculate cluster center after filtering
            if len(cluster_positions) > 0:
                cluster_center = np.mean(cluster_positions, axis=0)
                cluster_mean_log_m200_mass = np.mean(cluster_log_m200_masses)
        
        # Group by MCMC ID (after mass filtering)
        unique_mcmc_ids = np.unique(cluster_mcmc_ids)
        
        for mcmc_id in unique_mcmc_ids:
            mcmc_mask = cluster_mcmc_ids == mcmc_id
            mcmc_indices_in_cluster = cluster_indices[mcmc_mask]
            
            if len(mcmc_indices_in_cluster) > 1:
                # Multiple halos from same MCMC in this cluster
                mcmc_positions = cluster_positions[mcmc_mask]
                mcmc_log_m200_masses = cluster_log_m200_masses[mcmc_mask]
                
                if use_mass_distance:
                    # Use combined position + log-mass distance
                    pos_distances = np.linalg.norm(mcmc_positions - cluster_center, axis=1)
                    log_m200_mass_distances = np.abs(mcmc_log_m200_masses - cluster_mean_log_m200_mass)
                    
                    # Normalize both distances and combine
                    pos_distances_norm = pos_distances / np.max(pos_distances) if np.max(pos_distances) > 0 else pos_distances
                    log_m200_mass_distances_norm = log_m200_mass_distances / np.max(log_m200_mass_distances) if np.max(log_m200_mass_distances) > 0 else log_m200_mass_distances
                    
                    combined_distances = pos_distances_norm + log_m200_mass_distances_norm
                    closest_idx = np.argmin(combined_distances)
                else:
                    # Use only position distance (original method)
                    distances_to_center = np.linalg.norm(mcmc_positions - cluster_center, axis=1)
                    closest_idx = np.argmin(distances_to_center)
                
                # Keep only the closest one, mark others as noise
                for i, global_idx in enumerate(mcmc_indices_in_cluster):
                    if i != closest_idx:
                        cluster_labels[global_idx] = -1
    
    return cluster_labels

def combine_haloes(mcmc_data):
    combined_data = {}
    halo_provenance = []
    
    # Get all available property keys from first MCMC sample
    first_mcmc = next(iter(mcmc_data.values()))
    property_keys = list(first_mcmc.keys())
    
    # Initialize combined arrays for each property
    for key in property_keys:
        combined_data[key] = []
    
    for mcmc_id, data in mcmc_data.items():
        n_haloes = len(data['SO/200_crit/TotalMass'])
        
        # Combine all properties
        for key in property_keys:
            if key in data:
                combined_data[key].append(data[key])
            else:
                # Handle missing properties with appropriate fill values
                if 'Mass' in key:
                    fill_shape = (n_haloes,)
                elif 'CentreOfMass' in key:
                    fill_shape = (n_haloes, 3)
                else:
                    fill_shape = (n_haloes,)
                combined_data[key].append(np.full(fill_shape, np.nan))
        
        # Track provenance
        for i in range(n_haloes):
            halo_provenance.append({'mcmc_id': mcmc_id, 'original_index': i})
    
    # Stack/concatenate all properties
    for key in property_keys:
        if combined_data[key]:
            if len(combined_data[key][0].shape) == 1:
                combined_data[key] = np.concatenate(combined_data[key])
            else:
                combined_data[key] = np.vstack(combined_data[key])
    
    return combined_data, halo_provenance

def load_data_with_radius_filter(config, radius_inner=None, radius_outer=None):
    mcmc_data = {}
    
    # Use radius_outer for initial loading if provided, otherwise use radius_cut
    initial_radius_cut = radius_outer if radius_outer is not None else config.mode1.radius_cut
    
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
    
    for mcmc_id in range(config.mode1.mcmc_start, config.mode1.mcmc_end + 1):
        filename = os.path.join(config.global_config.basedir, f"mcmc_{mcmc_id}/soap/SOAP_uncompressed/HBTplus/halo_properties_0077.hdf5")
        soap_data = SOAPData(filename, radius_cut=initial_radius_cut)
        soap_data.load_groups(properties=to_load, only_centrals=True)
        soap_data.set_observer(config.global_config.observer_coords, skip_redshift=True)
       
        # Don't want to keep redshift
        del soap_data.data["redshift"]

        # Apply M200 mass cut
        m200_masses = soap_data.data['SO/200_crit/TotalMass']
        m200_mass_mask = m200_masses >= config.mode1.m200_mass_cut
        
        # Apply radius filtering if both inner and outer are specified
        if radius_inner is not None and radius_outer is not None:
            distances = soap_data.data['dist']
            radius_mask = (distances >= radius_inner) & (distances <= radius_outer)
            combined_mask = m200_mass_mask & radius_mask
        else:
            combined_mask = m200_mass_mask
        
        filtered_data = {}
        for key, value in soap_data.data.items():
            if isinstance(value, np.ndarray) and len(value) == len(m200_masses):
                filtered_data[key] = value[combined_mask]
            else:
                filtered_data[key] = value
        
        mcmc_data[mcmc_id] = filtered_data
        
        print(f"Loaded MCMC step {mcmc_id}: {len(mcmc_data[mcmc_id]['SO/200_crit/TotalMass'])} haloes")
    
    return mcmc_data

def find_stable_haloes(mcmc_data, config, eps=None, min_samples=None):
    combined_data, halo_provenance = combine_haloes(mcmc_data)
    
    positions = combined_data['SO/200_crit/CentreOfMass']
    m200_masses = combined_data['SO/200_crit/TotalMass']
    mcmc_ids = np.array([p['mcmc_id'] for p in halo_provenance])
    
    # Use provided eps and min_samples, otherwise default to mode1 config
    clustering_eps = eps if eps is not None else config.mode1.eps
    clustering_min_samples = min_samples if min_samples is not None else config.mode1.min_samples
    
    # Run standard DBSCAN
    clustering = DBSCAN(eps=clustering_eps, min_samples=clustering_min_samples)
    cluster_labels = clustering.fit_predict(positions)
    
    # Enforce MCMC constraint
    cluster_labels = enforce_mcmc_constraint(cluster_labels, positions, mcmc_ids)
    
    stable_haloes = []
    
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue
            
        cluster_mask = cluster_labels == cluster_id
        cluster_positions = positions[cluster_mask]
        cluster_m200_masses = m200_masses[cluster_mask]
        cluster_provenance = [halo_provenance[i] for i in range(len(halo_provenance)) if cluster_mask[i]]
        
        cluster_size = len(cluster_positions)
        mean_position = np.mean(cluster_positions, axis=0)
        mean_m200_mass = np.mean(cluster_m200_masses)
        position_std = np.std(cluster_positions, axis=0)
        m200_mass_std = np.std(cluster_m200_masses)
        log10_m200_mass_std = np.std(np.log10(cluster_m200_masses))

        # Calculate mean subhalo mass and M500
        cluster_subhalo_masses = combined_data['BoundSubhalo/TotalMass'][cluster_mask]
        cluster_m500 = combined_data['SO/500_crit/TotalMass'][cluster_mask]
        
        # Handle NaN values
        valid_subhalo_masses = cluster_subhalo_masses[~np.isnan(cluster_subhalo_masses)]
        valid_m500 = cluster_m500[~np.isnan(cluster_m500)]
        
        mean_subhalo_mass = np.mean(valid_subhalo_masses) if len(valid_subhalo_masses) > 0 else np.nan
        mean_m500 = np.mean(valid_m500) if len(valid_m500) > 0 else np.nan
        subhalo_mass_std = np.std(valid_subhalo_masses) if len(valid_subhalo_masses) > 0 else np.nan
        m500_std = np.std(valid_m500) if len(valid_m500) > 0 else np.nan
        log10_m500_std = np.std(np.log10(valid_m500)) if len(valid_m500) > 0 else np.nan

        # Extract all member data for this cluster
        member_data = {}
        for key, data in combined_data.items():
            member_data[key] = data[cluster_mask]
        
        stable_haloes.append({
            'cluster_id': cluster_id,
            'cluster_size': cluster_size,
            'mean_position': mean_position,
            'mean_m200_mass': mean_m200_mass,
            'mean_subhalo_mass': mean_subhalo_mass,
            'mean_m500': mean_m500,
            'position_std': position_std,
            'm200_mass_std': m200_mass_std,
            'subhalo_mass_std': subhalo_mass_std,
            'm500_std': m500_std,
            'members': cluster_provenance,
            'member_data': member_data,
            'log10_m200_mass_std': log10_m200_mass_std,
            'log10_m500_std': log10_m500_std,
        })
    
    return stable_haloes, positions, m200_masses, halo_provenance, cluster_labels

def find_stable_haloes_with_mass_filtering(mcmc_data, config, eps=None, min_samples=None, 
                                          mass_outlier_threshold=0.3, use_mass_distance=False,
                                          mass_weighted_clustering=False, mass_weight_power=0.5):
    """
    Enhanced version of find_stable_haloes with mass-based filtering and optional mass weighting.
    
    Parameters:
    -----------
    mass_outlier_threshold : float
        Threshold in log-mass units for outlier detection (e.g., 0.3 = ~2x mass difference)
        If >= 1.0, interpreted as number of standard deviations in log-mass space
    use_mass_distance : bool
        Whether to use combined position+log-mass distance for MCMC constraint
    mass_weighted_clustering : bool
        Whether to use mass-weighted positions for clustering
    mass_weight_power : float
        Power to raise mass weights to (affects weighting strength)
    """
    combined_data, halo_provenance = combine_haloes(mcmc_data)
    
    positions = combined_data['SO/200_crit/CentreOfMass']
    m200_masses = combined_data['SO/200_crit/TotalMass']
    log_m200_masses = np.log10(m200_masses)  # Work in log-mass space
    mcmc_ids = np.array([p['mcmc_id'] for p in halo_provenance])
    
    # Use provided eps and min_samples, otherwise default to mode1 config
    clustering_eps = eps if eps is not None else config.mode1.eps
    clustering_min_samples = min_samples if min_samples is not None else config.mode1.min_samples
    
    # Optional: Weight positions by mass for clustering
    if mass_weighted_clustering:
        # Create mass weights from log-masses
        log_m200_mass_weights = (log_m200_masses - np.min(log_m200_masses)) + 1  # Ensure positive weights
        m200_mass_weights = log_m200_mass_weights ** mass_weight_power
        weighted_positions = positions * m200_mass_weights.reshape(-1, 1)
        clustering_input = weighted_positions
    else:
        clustering_input = positions
    
    # Run standard DBSCAN
    clustering = DBSCAN(eps=clustering_eps, min_samples=clustering_min_samples)
    cluster_labels = clustering.fit_predict(clustering_input)
    
    # Enforce MCMC constraint with mass filtering
    cluster_labels = enforce_mcmc_constraint_with_mass_filter(
        cluster_labels, positions, m200_masses, mcmc_ids, 
        mass_outlier_threshold, use_mass_distance
    )
    
    stable_haloes = []
    
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue
            
        cluster_mask = cluster_labels == cluster_id
        cluster_positions = positions[cluster_mask]
        cluster_m200_masses = m200_masses[cluster_mask]
        cluster_log_m200_masses = log_m200_masses[cluster_mask]
        cluster_provenance = [halo_provenance[i] for i in range(len(halo_provenance)) if cluster_mask[i]]
        
        cluster_size = len(cluster_positions)
        mean_position = np.mean(cluster_positions, axis=0)
        mean_m200_mass = np.mean(cluster_m200_masses)
        mean_log_m200_mass = np.mean(cluster_log_m200_masses)
        position_std = np.std(cluster_positions, axis=0)
        m200_mass_std = np.std(cluster_m200_masses)
        log10_m200_mass_std = np.std(np.log10(cluster_m200_masses))
        log_m200_mass_std = np.std(cluster_log_m200_masses)
        
        # Calculate mean subhalo mass and M500
        cluster_subhalo_masses = combined_data['BoundSubhalo/TotalMass'][cluster_mask]
        cluster_m500 = combined_data['SO/500_crit/TotalMass'][cluster_mask]
        
        # Handle NaN values
        valid_subhalo_masses = cluster_subhalo_masses[~np.isnan(cluster_subhalo_masses)]
        valid_m500 = cluster_m500[~np.isnan(cluster_m500)]
        
        mean_subhalo_mass = np.mean(valid_subhalo_masses) if len(valid_subhalo_masses) > 0 else np.nan
        mean_m500 = np.mean(valid_m500) if len(valid_m500) > 0 else np.nan
        subhalo_mass_std = np.std(valid_subhalo_masses) if len(valid_subhalo_masses) > 0 else np.nan
        m500_std = np.std(valid_m500) if len(valid_m500) > 0 else np.nan
        log10_m500_std = np.std(np.log10(valid_m500)) if len(valid_m500) > 0 else np.nan

        # Mass statistics in both linear and log space
        m200_mass_cv = m200_mass_std / mean_m200_mass if mean_m200_mass > 0 else 0  # Coefficient of variation
        m200_mass_range = np.max(cluster_m200_masses) - np.min(cluster_m200_masses)
        log_m200_mass_range = np.max(cluster_log_m200_masses) - np.min(cluster_log_m200_masses)
        m200_mass_ratio_range = 10**log_m200_mass_range  # Mass ratio between min and max
        
        # Extract all member data for this cluster
        member_data = {}
        for key, data in combined_data.items():
            member_data[key] = data[cluster_mask]
        
        stable_haloes.append({
            'cluster_id': cluster_id,
            'cluster_size': cluster_size,
            'mean_position': mean_position,
            'mean_m200_mass': mean_m200_mass,
            'mean_subhalo_mass': mean_subhalo_mass,
            'mean_m500': mean_m500,
            'mean_log_m200_mass': mean_log_m200_mass,
            'position_std': position_std,
            'm200_mass_std': m200_mass_std,
            'subhalo_mass_std': subhalo_mass_std,
            'm500_std': m500_std,
            'log_m200_mass_std': log_m200_mass_std,
            'm200_mass_cv': m200_mass_cv,
            'm200_mass_range': m200_mass_range,
            'log_m200_mass_range': log_m200_mass_range,
            'm200_mass_ratio_range': m200_mass_ratio_range,
            'members': cluster_provenance,
            'member_data': member_data,
            'log10_m200_mass_std': log10_m200_mass_std,
            'log10_m500_std': log10_m500_std,
        })
    
    return stable_haloes, positions, m200_masses, halo_provenance, cluster_labels

def analyze_mass_distribution_in_clusters(stable_haloes):
    """
    Analyze mass distributions within clusters to help tune mass filtering parameters.
    Now works in log-mass space which is more appropriate for halo masses.
    """
    print("\nM200 mass distribution analysis (log-mass space):")
    print("="*60)
    
    for i, cluster in enumerate(stable_haloes[:5]):  # Top 5 clusters
        m200_masses = cluster['member_data']['SO/200_crit/TotalMass']
        log_m200_masses = np.log10(m200_masses)
        
        print(f"\nCluster {cluster['cluster_id']} (size={cluster['cluster_size']}):")
        print(f"  M200 range: {np.min(m200_masses):.2e} - {np.max(m200_masses):.2e} M☉")
        print(f"  Log-M200 range: {np.min(log_m200_masses):.2f} - {np.max(log_m200_masses):.2f}")
        print(f"  M200 ratio (max/min): {np.max(m200_masses)/np.min(m200_masses):.1f}x")
        print(f"  Log-M200 mean ± std: {np.mean(log_m200_masses):.2f} ± {np.std(log_m200_masses):.2f}")
        print(f"  Log-M200 spread: {cluster['log_m200_mass_range']:.2f} dex")
        
        # Show which masses would be flagged as outliers at different thresholds
        mean_log_m200_mass = np.mean(log_m200_masses)
        log_m200_mass_std = np.std(log_m200_masses)
        
        if log_m200_mass_std > 0:
            # Standard deviation based outliers
            z_scores = np.abs((log_m200_masses - mean_log_m200_mass) / log_m200_mass_std)
            outliers_2sigma = np.sum(z_scores > 2.0)
            outliers_3sigma = np.sum(z_scores > 3.0)
            print(f"  Potential outliers (2σ in log-M200): {outliers_2sigma}/{len(m200_masses)}")
            print(f"  Potential outliers (3σ in log-M200): {outliers_3sigma}/{len(m200_masses)}")
        
        # Direct log-mass difference outliers
        log_deviations = np.abs(log_m200_masses - mean_log_m200_mass)
        outliers_02dex = np.sum(log_deviations > 0.2)  # Factor of ~1.6
        outliers_03dex = np.sum(log_deviations > 0.3)  # Factor of ~2.0
        outliers_05dex = np.sum(log_deviations > 0.5)  # Factor of ~3.2
        
        print(f"  Outliers >0.2 dex (~1.6x): {outliers_02dex}/{len(m200_masses)}")
        print(f"  Outliers >0.3 dex (~2.0x): {outliers_03dex}/{len(m200_masses)}")
        print(f"  Outliers >0.5 dex (~3.2x): {outliers_05dex}/{len(m200_masses)}")

def enhanced_find_stable_haloes(mcmc_data, config, 
                               mass_outlier_threshold=0.3,  # 0.3 dex = factor of ~2
                               use_mass_distance=True,
                               mass_weighted_clustering=False):
    """
    Wrapper function that can be used as a drop-in replacement for find_stable_haloes.
    
    Default mass_outlier_threshold of 0.3 dex means halos with M200 masses differing by more 
    than a factor of ~2 from the cluster mean will be filtered out.
    """
    return find_stable_haloes_with_mass_filtering(
        mcmc_data, config,
        eps=config.mode1.eps,
        min_samples=config.mode1.min_samples,
        mass_outlier_threshold=mass_outlier_threshold,
        use_mass_distance=use_mass_distance,
        mass_weighted_clustering=mass_weighted_clustering
    )
