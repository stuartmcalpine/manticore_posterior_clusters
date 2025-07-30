# stacking_backend/analysis/pipeline.py
import numpy as np
from astropy.cosmology import Planck18
from ..data import load_pr4_data, PatchExtractor
from .stacking import PatchStacker
from .profiles import RadialProfileCalculator
from .individual_clusters import IndividualClusterAnalyzer
from .validation import NullTestValidator

class ClusterAnalysisPipeline:
    """Main analysis pipeline for cluster stacking analysis with validation"""
    
    def __init__(self):
        self._initialize_data()
    
    def _initialize_data(self):
        """Load PR4 data and initialize components"""
        data = load_pr4_data()
        
        self.patch_extractor = PatchExtractor(
            y_map=data["y_map"],
            combined_mask=data["combined_mask"],
            nside=data["nside"]
        )
        
        self.stacker = PatchStacker(self.patch_extractor)
        self.individual_analyzer = IndividualClusterAnalyzer(self.patch_extractor)
        self.validator = NullTestValidator(self.patch_extractor)
    
    def run_individual_r200_analysis_with_validation(self, coord_list, inner_r200_factor=1.0, outer_r200_factor=3.0,
                                                   patch_size_deg=15.0, npix=256, max_patches=None,
                                                   min_coverage=0.9, n_radial_bins=20, run_null_tests=True,
                                                   n_bootstrap=500, n_random=500):
        """Full analysis pipeline with physical scaling and validation"""
        
        print("🚀 CLUSTER ANALYSIS PIPELINE (Physical Scaling + Validation)")
        print("="*70)
        
        # Convert to physical coordinates
        physical_coords = self._convert_to_physical_coordinates(coord_list)
        
        # Step 1: Calculate individual cluster measurements
        print(f"\n🔍 Step 1: Individual cluster measurements...")
        individual_results, rejection_stats = self.individual_analyzer.calculate_measurements(
            coord_list=physical_coords,
            inner_r200_factor=inner_r200_factor,
            outer_r200_factor=outer_r200_factor,
            patch_size_deg=patch_size_deg,
            npix=npix,
            min_coverage=min_coverage
        )
        
        if not individual_results:
            return {'success': False, 'error': 'No valid individual measurements'}
        
        # Extract measurements for bootstrap
        individual_delta_y = [result['delta_y'] for result in individual_results]
        valid_coords = [result['coords'] for result in individual_results]
        
        # Step 2: Bootstrap error estimation
        print(f"\n🔄 Step 2: Bootstrap error estimation ({n_bootstrap} samples)...")
        bootstrap_results = self._bootstrap_analysis(valid_coords, individual_delta_y, 
                                                    inner_r200_factor, outer_r200_factor,
                                                    patch_size_deg, npix, min_coverage, n_bootstrap)
        
        # Step 3: Create stacked patch
        print(f"\n📚 Step 3: Stacking patches...")
        stacked_patch, stacking_info, stack_rejection_stats = self.stacker.stack_patches(
            coord_list=valid_coords,
            patch_size_deg=patch_size_deg,
            npix=npix,
            min_coverage=min_coverage,
            max_patches=max_patches
        )
        
        if stacked_patch is None:
            return {'success': False, 'error': 'No valid patches for stacking'}
        
        # Step 4: Null tests with random pointings
        null_results = None
        if run_null_tests:
            print(f"\n🎯 Step 4: Null tests with random pointings ({n_random} samples)...")
            null_results = self.validator.run_null_tests(
                n_random_pointings=n_random,
                coord_list=valid_coords,
                inner_r200_factor=inner_r200_factor,
                outer_r200_factor=outer_r200_factor,
                patch_size_deg=patch_size_deg,
                npix=npix,
                min_coverage=min_coverage
            )
        
        # Step 5: Calculate radial profile
        print(f"\n📊 Step 5: Radial profile...")
        radii, profile, profile_errors, profile_counts = RadialProfileCalculator.calculate_profile(
            stacked_patch=stacked_patch,
            patch_size_deg=patch_size_deg,
            n_radial_bins=n_radial_bins
        )
        
        # Step 6: Compile results with validation
        results = self._compile_results_with_validation(
            individual_results, individual_delta_y, bootstrap_results, 
            stacked_patch, stacking_info, radii, profile, profile_errors, profile_counts,
            rejection_stats, coord_list, patch_size_deg, npix, 
            inner_r200_factor, outer_r200_factor, null_results
        )
        
        self._print_final_summary_with_validation(results)
        
        return results
    
    def _convert_to_physical_coordinates(self, coord_list):
        """Convert observer frame coordinates to physical frame with E(z) corrections"""
        physical_coords = []
        
        for coords in coord_list:
            if len(coords) >= 4:  # Has redshift
                lon_gal, lat_gal, r200_deg, z = coords[0], coords[1], coords[2], coords[3]
                
                # Calculate E(z) = H(z)/H_0
                Ez = Planck18.efunc(z)
                
                # Scale R200 to physical coordinates: θ_phys = θ_obs * D_A(z=0) / D_A(z)
                # For small z, this is approximately θ_phys ≈ θ_obs / (1 + z)
                D_A_z = Planck18.angular_diameter_distance(z).value  # Mpc
                D_A_0 = Planck18.angular_diameter_distance(0.01).value  # Reference
                
                r200_physical = r200_deg * D_A_0 / D_A_z
                
                # Apply E(z) correction to pressure normalization (handled in photometry)
                physical_coords.append([lon_gal, lat_gal, r200_physical, z, Ez])
            else:
                # No redshift information, keep as-is
                physical_coords.append(coords)
        
        return physical_coords
    
    def _bootstrap_analysis(self, valid_coords, individual_delta_y, inner_r200_factor, 
                           outer_r200_factor, patch_size_deg, npix, min_coverage, n_bootstrap):
        """Bootstrap resampling for robust error estimation"""
        bootstrap_means = []
        n_clusters = len(valid_coords)
        
        for i in range(n_bootstrap):
            # Resample with replacement
            bootstrap_indices = np.random.choice(n_clusters, size=n_clusters, replace=True)
            bootstrap_measurements = [individual_delta_y[idx] for idx in bootstrap_indices]
            bootstrap_means.append(np.mean(bootstrap_measurements))
        
        bootstrap_mean = np.mean(bootstrap_means)
        bootstrap_std = np.std(bootstrap_means)
        bootstrap_error = bootstrap_std  # Standard error from bootstrap
        
        return {
            'bootstrap_mean': bootstrap_mean,
            'bootstrap_std': bootstrap_std,
            'bootstrap_error': bootstrap_error,
            'bootstrap_samples': bootstrap_means
        }
    
    def _compile_results_with_validation(self, individual_results, individual_delta_y, bootstrap_results,
                                       stacked_patch, stacking_info, radii, profile, profile_errors, profile_counts,
                                       rejection_stats, coord_list, patch_size_deg, npix, 
                                       inner_r200_factor, outer_r200_factor, null_results):
        """Compile results with validation metrics"""
        
        # Use bootstrap error for final significance
        mean_delta_y = bootstrap_results['bootstrap_mean']
        error_delta_y = bootstrap_results['bootstrap_error']
        significance = mean_delta_y / error_delta_y if error_delta_y > 0 else 0
        
        # Calculate null test significance if available
        null_significance = None
        if null_results:
            null_mean = null_results['random_mean']
            null_std = null_results['random_std']
            # Compare cluster signal to null distribution
            null_significance = (mean_delta_y - null_mean) / null_std if null_std > 0 else 0
        
        # Calculate R200 statistics
        r200_values = [result['r200_deg'] for result in individual_results]
        inner_radii = [result['inner_radius_deg'] for result in individual_results]
        outer_radii = [result['outer_radius_deg'] for result in individual_results]
        
        return {
            'success': True,
            
            # Main measurements with bootstrap errors
            'mean_delta_y': mean_delta_y,
            'error_mean': error_delta_y,
            'std_delta_y': bootstrap_results['bootstrap_std'],
            'significance': significance,
            'error_type': 'bootstrap',
            
            # Bootstrap results
            'bootstrap_results': bootstrap_results,
            
            # Validation results
            'null_results': null_results,
            'null_significance': null_significance,
            
            # Individual cluster results
            'individual_results': individual_results,
            'individual_measurements': individual_delta_y,
            
            # Stacked patch data
            'stacked_patch': stacked_patch,
            'stacking_info': stacking_info,
            
            # Radial profile
            'profile_radii': radii,
            'profile_mean': profile,
            'profile_errors': profile_errors,
            'profile_counts': profile_counts,
            
            # R200 statistics
            'r200_values': r200_values,
            'r200_median': np.median(r200_values),
            'r200_range': (np.min(r200_values), np.max(r200_values)),
            'inner_radii_range': (np.min(inner_radii), np.max(inner_radii)),
            'outer_radii_range': (np.min(outer_radii), np.max(outer_radii)),
            
            # Sample information
            'n_measurements': len(individual_results),
            'n_input_coords': len(coord_list),
            'n_rejected': len(coord_list) - len(individual_results),
            'rejection_stats': rejection_stats,
            
            # Analysis parameters
            'patch_size_deg': patch_size_deg,
            'npix': npix,
            'inner_r200_factor': inner_r200_factor,
            'outer_r200_factor': outer_r200_factor
        }
    
    def _print_final_summary_with_validation(self, results):
        """Print final analysis summary with validation metrics"""
        print(f"\n🎯 Final Results:")
        print(f"✅ Bootstrap statistics:")
        print(f"   Mean Δy: {results['mean_delta_y']:.2e}")
        print(f"   Bootstrap error: {results['error_mean']:.2e}")
        print(f"   Detection significance: {results['significance']:.1f}σ")
        
        if results['null_results']:
            print(f"\n🎲 Null test validation:")
            print(f"   Random mean: {results['null_results']['random_mean']:.2e}")
            print(f"   Random std: {results['null_results']['random_std']:.2e}")
            print(f"   Null-corrected significance: {results['null_significance']:.1f}σ")
            
            if abs(results['null_results']['random_mean']) > 3 * results['null_results']['random_std']:
                print(f"   ⚠️  WARNING: Null test shows potential systematic bias")
            else:
                print(f"   ✅ Null test passed: random pointings consistent with zero")
        
        print(f"\n📏 Sample statistics:")
        print(f"   R200 median: {results['r200_median']:.3f}°")
        print(f"   Sample: {results['n_measurements']}/{results['n_input_coords']} clusters")
        
        print(f"\n🎉 Analysis complete with validation!")
