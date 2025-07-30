# stacking_backend/analysis/validation.py
import numpy as np
from .photometry import AperturePhotometry

class NullTestValidator:
    """Validation through null tests with random pointings"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
    def run_null_tests(self, n_random_pointings, coord_list, inner_r200_factor, outer_r200_factor,
                      patch_size_deg, npix, min_coverage):
        """Run null tests with random sky pointings"""
        
        print(f"🎲 Running null tests with {n_random_pointings} random pointings...")
        
        # Generate random pointings avoiding cluster regions
        random_coords = self._generate_random_pointings(n_random_pointings, coord_list, patch_size_deg)
        
        # Calculate measurements for random pointings
        random_measurements = []
        rejection_count = 0
        
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
                    rejection_count += 1
                    
            except Exception as e:
                rejection_count += 1
                continue
        
        if not random_measurements:
            print(f"   ❌ No valid random measurements!")
            return None
        
        # Calculate null test statistics
        random_mean = np.mean(random_measurements)
        random_std = np.std(random_measurements)
        random_error = random_std / np.sqrt(len(random_measurements))
        
        print(f"   ✅ Null tests complete:")
        print(f"      Valid measurements: {len(random_measurements)}/{n_random_pointings}")
        print(f"      Random mean: {random_mean:.2e} ± {random_error:.2e}")
        print(f"      Random std: {random_std:.2e}")
        
        return {
            'random_measurements': random_measurements,
            'random_mean': random_mean,
            'random_std': random_std,
            'random_error': random_error,
            'n_valid_random': len(random_measurements),
            'n_rejected_random': rejection_count
        }
    
    def _generate_random_pointings(self, n_random, cluster_coords, patch_size_deg):
        """Generate random pointings avoiding cluster regions"""
        random_coords = []
        exclusion_radius = patch_size_deg * 2  # Avoid overlap with cluster patches
        
        # Extract cluster positions
        cluster_positions = []
        for coords in cluster_coords:
            if len(coords) >= 3:
                cluster_positions.append([coords[0], coords[1]])
        cluster_positions = np.array(cluster_positions)
        
        attempts = 0
        while len(random_coords) < n_random and attempts < n_random * 10:
            # Generate random galactic coordinates
            # Avoid galactic plane |b| < 10 degrees
            lon_gal = np.random.uniform(0, 360)
            lat_gal_abs = np.random.uniform(10, 80)  # Avoid galactic plane
            lat_gal = lat_gal_abs * np.random.choice([-1, 1])  # Random hemisphere
            
            # Use median R200 from cluster sample for random pointings
            median_r200 = np.median([coords[2] for coords in cluster_coords if len(coords) >= 3])
            
            # Check if far enough from all clusters
            position = np.array([lon_gal, lat_gal])
            if len(cluster_positions) > 0:
                distances = np.array([self._angular_distance(position, cluster_pos) 
                                    for cluster_pos in cluster_positions])
                if np.min(distances) < exclusion_radius:
                    attempts += 1
                    continue
            
            random_coords.append([lon_gal, lat_gal, median_r200])
            attempts += 1
        
        if len(random_coords) < n_random:
            print(f"   ⚠️  Warning: Only generated {len(random_coords)}/{n_random} random pointings")
        
        return random_coords
    
    def _angular_distance(self, pos1, pos2):
        """Calculate angular distance between two positions in degrees"""
        lon1, lat1 = np.radians(pos1)
        lon2, lat2 = np.radians(pos2)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return np.degrees(c)
