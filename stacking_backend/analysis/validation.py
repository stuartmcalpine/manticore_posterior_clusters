# stacking_backend/analysis/validation.py
import numpy as np
from .photometry import AperturePhotometry

class NullTestValidator:
    """Validation through null tests with mask-bias correction"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
    def run_null_tests(self, n_random_pointings, coord_list, inner_r200_factor, outer_r200_factor,
                      patch_size_deg, npix, min_coverage):
        """Run null tests with mask-bias correction"""
        
        print(f"🎲 Running mask-bias-corrected null tests with {n_random_pointings} random pointings...")
        
        # First, analyze mask coverage distribution of actual cluster sample
        cluster_coverage_stats = self._analyze_cluster_mask_coverage(
            coord_list, patch_size_deg, npix, inner_r200_factor, outer_r200_factor
        )
        
        # Generate random pointings with mask-bias correction
        random_coords = self._generate_mask_matched_random_pointings(
            n_random_pointings, coord_list, patch_size_deg, cluster_coverage_stats
        )
        
        # Calculate measurements for random pointings
        random_measurements = []
        rejection_count = 0
        rejection_reasons = {'insufficient_mask_coverage': 0, 'extraction_error': 0, 
                           'insufficient_pixels': 0, 'mask_mismatch': 0}
        
        for i, coords in enumerate(random_coords):
            try:
                lon_gal, lat_gal, r200_deg = coords[0], coords[1], coords[2]
                
                # Extract patch
                patch_data, mask_patch = self.patch_extractor.extract_patch(
                    center_coords=(lon_gal, lat_gal),
                    patch_size_deg=patch_size_deg,
                    npix=npix
                )
                
                if mask_patch is not None:
                    patch_data[~mask_patch] = np.nan
                
                # Check if this random pointing has similar masking to cluster sample
                if not self._validate_mask_similarity(mask_patch, cluster_coverage_stats, 
                                                    inner_r200_factor, outer_r200_factor, 
                                                    r200_deg, patch_size_deg, npix):
                    rejection_reasons['mask_mismatch'] += 1
                    rejection_count += 1
                    continue
                
                # Perform aperture photometry
                dummy_mask = np.isfinite(patch_data)
                result, diagnostics = AperturePhotometry.calculate_individual_r200_photometry(
                    patch=patch_data,
                    mask_patch=dummy_mask,
                    r200_deg=r200_deg,
                    inner_r200_factor=inner_r200_factor,
                    outer_r200_factor=outer_r200_factor,
                    patch_size_deg=patch_size_deg,
                    npix=npix,
                    min_coverage=min_coverage
                )
                
                if result is not None:
                    random_measurements.append(result['delta_y'])
                else:
                    reason = diagnostics.get('rejection_reason', 'unknown')
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    rejection_count += 1
                    
            except Exception as e:
                rejection_reasons['extraction_error'] += 1
                rejection_count += 1
                continue
        
        if not random_measurements:
            print(f"   ❌ No valid random measurements!")
            return None
        
        # Calculate null test statistics
        random_mean = np.mean(random_measurements)
        random_std = np.std(random_measurements)
        random_error = random_std / np.sqrt(len(random_measurements))
        
        print(f"   ✅ Mask-bias-corrected null tests complete:")
        print(f"      Valid measurements: {len(random_measurements)}/{n_random_pointings}")
        print(f"      Random mean: {random_mean:.2e} ± {random_error:.2e}")
        print(f"      Random std: {random_std:.2e}")
        print(f"      Rejection breakdown:")
        for reason, count in rejection_reasons.items():
            if count > 0:
                print(f"        - {reason}: {count}")
        
        return {
            'random_measurements': random_measurements,
            'random_mean': random_mean,
            'random_std': random_std,
            'random_error': random_error,
            'n_valid_random': len(random_measurements),
            'n_rejected_random': rejection_count,
            'rejection_reasons': rejection_reasons,
            'cluster_coverage_stats': cluster_coverage_stats
        }
    
    def _analyze_cluster_mask_coverage(self, coord_list, patch_size_deg, npix, 
                                     inner_r200_factor, outer_r200_factor):
        """Analyze mask coverage statistics of actual cluster sample"""
        
        print(f"   📊 Analyzing mask coverage of {len(coord_list)} clusters...")
        
        coverage_stats = {
            'inner_coverages': [],
            'outer_coverages': [],
            'total_coverages': [],
            'r200_values': []
        }
        
        for coords in coord_list:
            try:
                lon_gal, lat_gal, r200_deg = coords[0], coords[1], coords[2]
                
                # Extract patch
                patch_data, mask_patch = self.patch_extractor.extract_patch(
                    center_coords=(lon_gal, lat_gal),
                    patch_size_deg=patch_size_deg,
                    npix=npix
                )
                
                if mask_patch is not None:
                    # Calculate aperture coverages
                    inner_mask, outer_mask = AperturePhotometry.create_aperture_masks(
                        patch_size_deg, npix, 
                        inner_r200_factor * r200_deg, 
                        outer_r200_factor * r200_deg
                    )
                    
                    inner_coverage = np.sum(inner_mask & mask_patch) / np.sum(inner_mask)
                    outer_coverage = np.sum(outer_mask & mask_patch) / np.sum(outer_mask)
                    total_coverage = np.mean(mask_patch)
                    
                    coverage_stats['inner_coverages'].append(inner_coverage)
                    coverage_stats['outer_coverages'].append(outer_coverage)
                    coverage_stats['total_coverages'].append(total_coverage)
                    coverage_stats['r200_values'].append(r200_deg)
                    
            except Exception:
                continue
        
        # Convert to arrays and calculate statistics
        for key in ['inner_coverages', 'outer_coverages', 'total_coverages', 'r200_values']:
            coverage_stats[key] = np.array(coverage_stats[key])
        
        coverage_stats['median_inner_coverage'] = np.median(coverage_stats['inner_coverages'])
        coverage_stats['median_outer_coverage'] = np.median(coverage_stats['outer_coverages'])
        coverage_stats['median_total_coverage'] = np.median(coverage_stats['total_coverages'])
        coverage_stats['median_r200'] = np.median(coverage_stats['r200_values'])
        
        print(f"      Median inner coverage: {coverage_stats['median_inner_coverage']:.2f}")
        print(f"      Median outer coverage: {coverage_stats['median_outer_coverage']:.2f}")
        print(f"      Median total coverage: {coverage_stats['median_total_coverage']:.2f}")
        
        return coverage_stats
    
    def _generate_mask_matched_random_pointings(self, n_random, cluster_coords, patch_size_deg, 
                                              cluster_coverage_stats):
        """Generate random pointings with mask statistics matching cluster sample"""
        
        random_coords = []
        exclusion_radius = max(patch_size_deg * 2.0, 5.0)  # Reduced from 3x
        
        # Extract cluster positions for exclusion
        cluster_positions = []
        for coords in cluster_coords:
            if len(coords) >= 3:
                cluster_positions.append([coords[0], coords[1]])
        cluster_positions = np.array(cluster_positions)
        
        # Define galactic latitude ranges (avoid galactic plane)
        lat_ranges = [(-80, -20), (20, 80)]
        
        attempts = 0
        max_attempts = n_random * 100  # Increased attempts for mask matching
        
        while len(random_coords) < n_random and attempts < max_attempts:
            # Choose random latitude range
            lat_range = lat_ranges[np.random.randint(len(lat_ranges))]
            
            # Generate random coordinates
            lon_gal = np.random.uniform(0, 360)
            lat_gal = np.random.uniform(lat_range[0], lat_range[1])
            
            # Use median R200 from cluster sample
            median_r200 = cluster_coverage_stats['median_r200']
            
            # Check distance from all clusters
            position = np.array([lon_gal, lat_gal])
            if len(cluster_positions) > 0:
                distances = np.array([self._angular_distance(position, cluster_pos) 
                                    for cluster_pos in cluster_positions])
                if np.min(distances) < exclusion_radius:
                    attempts += 1
                    continue
            
            # Check if this position has similar mask coverage to cluster sample
            try:
                test_patch, test_mask = self.patch_extractor.extract_patch(
                    center_coords=(lon_gal, lat_gal),
                    patch_size_deg=patch_size_deg,
                    npix=256  # Use smaller npix for speed in test
                )
                
                if test_mask is not None:
                    test_coverage = np.mean(test_mask)
                    target_coverage = cluster_coverage_stats['median_total_coverage']
                    
                    # Accept if coverage is within reasonable range of cluster sample
                    coverage_tolerance = 0.2  # ±20% tolerance
                    if abs(test_coverage - target_coverage) < coverage_tolerance:
                        random_coords.append([lon_gal, lat_gal, median_r200])
                    
            except Exception:
                pass
            
            attempts += 1
        
        if len(random_coords) < n_random:
            print(f"   ⚠️  Warning: Only generated {len(random_coords)}/{n_random} mask-matched random pointings")
        
        return random_coords
    
    def _validate_mask_similarity(self, mask_patch, cluster_coverage_stats, 
                                inner_r200_factor, outer_r200_factor, r200_deg, 
                                patch_size_deg, npix):
        """Validate that random pointing has similar mask characteristics to cluster sample"""
        
        if mask_patch is None:
            return True  # No masking applied
        
        try:
            # Calculate aperture coverages for this random pointing
            inner_mask, outer_mask = AperturePhotometry.create_aperture_masks(
                patch_size_deg, npix,
                inner_r200_factor * r200_deg,
                outer_r200_factor * r200_deg
            )
            
            inner_coverage = np.sum(inner_mask & mask_patch) / np.sum(inner_mask)
            outer_coverage = np.sum(outer_mask & mask_patch) / np.sum(outer_mask)
            
            # Compare to cluster sample statistics
            target_inner = cluster_coverage_stats['median_inner_coverage']
            target_outer = cluster_coverage_stats['median_outer_coverage']
            
            # Allow reasonable tolerance (±30%)
            tolerance = 0.3
            
            inner_ok = abs(inner_coverage - target_inner) < tolerance
            outer_ok = abs(outer_coverage - target_outer) < tolerance
            
            return inner_ok and outer_ok
            
        except Exception:
            return False
    
    def _angular_distance(self, pos1, pos2):
        """Calculate angular distance between two positions in degrees"""
        lon1, lat1 = np.radians(pos1)
        lon2, lat2 = np.radians(pos2)
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return np.degrees(c)
