import numpy as np
from ..data import load_pr4_data, PatchExtractor
from .stacking import PatchStacker
from .profiles import RadialProfileCalculator
from .individual_clusters import IndividualClusterAnalyzer

class ClusterAnalysisPipeline:
    """Main analysis pipeline for cluster stacking analysis"""
    
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
    
    def run_individual_r200_analysis(self, coord_list, inner_r200_factor=1.0, outer_r200_factor=3.0,
                                   patch_size_deg=15.0, npix=256, max_patches=None,
                                   min_coverage=0.9, n_radial_bins=20):
        """Full analysis pipeline using individual R200 values for each cluster"""
        
        print("🚀 CLUSTER ANALYSIS PIPELINE (Individual R200)")
        print("="*60)
        print(f"Input coordinates: {len(coord_list)}")
        print(f"Patch parameters: {patch_size_deg}° × {patch_size_deg}° ({npix}×{npix})")
        print(f"Aperture photometry: {inner_r200_factor}R200 inner, {outer_r200_factor}R200 outer")
        print(f"All clusters expected to have R200 data")
        
        # Step 1: Calculate individual cluster measurements
        print(f"\n🔍 Step 1: Individual cluster measurements...")
        individual_results, rejection_stats = self.individual_analyzer.calculate_measurements(
            coord_list=coord_list,
            inner_r200_factor=inner_r200_factor,
            outer_r200_factor=outer_r200_factor,
            patch_size_deg=patch_size_deg,
            npix=npix,
            min_coverage=min_coverage
        )
        
        if not individual_results:
            return {'success': False, 'error': 'No valid individual measurements'}
        
        # Extract measurements for stacking
        individual_delta_y = [result['delta_y'] for result in individual_results]
        valid_coords = [result['coords'] for result in individual_results]
        
        # Step 2: Create stacked patch
        print(f"\n📚 Step 2: Stacking patches from valid clusters...")
        stacked_patch, stacking_info, stack_rejection_stats = self.stacker.stack_patches(
            coord_list=valid_coords,
            patch_size_deg=patch_size_deg,
            npix=npix,
            min_coverage=min_coverage,
            max_patches=max_patches
        )
        
        if stacked_patch is None:
            return {'success': False, 'error': 'No valid patches for stacking'}
        
        # Step 3: Calculate radial profile
        print(f"\n📊 Step 3: Radial profile...")
        radii, profile, profile_errors, profile_counts = RadialProfileCalculator.calculate_profile(
            stacked_patch=stacked_patch,
            patch_size_deg=patch_size_deg,
            n_radial_bins=n_radial_bins
        )
        
        # Step 4: Calculate final statistics
        results = self._compile_results(
            individual_results, individual_delta_y, stacked_patch, stacking_info,
            radii, profile, profile_errors, profile_counts, rejection_stats,
            coord_list, patch_size_deg, npix, inner_r200_factor, outer_r200_factor
        )
        
        self._print_final_summary(results)
        
        return results
    
    def _compile_results(self, individual_results, individual_delta_y, stacked_patch, stacking_info,
                        radii, profile, profile_errors, profile_counts, rejection_stats,
                        coord_list, patch_size_deg, npix, inner_r200_factor, outer_r200_factor):
        """Compile final results dictionary"""
        
        # Calculate sample statistics
        mean_delta_y = np.mean(individual_delta_y)
        std_delta_y = np.std(individual_delta_y)
        error_delta_y = std_delta_y / np.sqrt(len(individual_delta_y))
        significance = mean_delta_y / error_delta_y if error_delta_y > 0 else 0
        
        # Calculate R200 statistics
        r200_values = [result['r200_deg'] for result in individual_results]
        inner_radii = [result['inner_radius_deg'] for result in individual_results]
        outer_radii = [result['outer_radius_deg'] for result in individual_results]
        
        return {
            'success': True,
            
            # Main measurements
            'mean_delta_y': mean_delta_y,
            'error_mean': error_delta_y,
            'std_delta_y': std_delta_y,
            'significance': significance,
            'error_type': 'sample_variance_individual_r200',
            
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
    
    def _print_final_summary(self, results):
        """Print final analysis summary"""
        print(f"\n🎯 Step 4: Final statistics from individual measurements...")
        print(f"✅ Sample statistics:")
        print(f"   Mean Δy: {results['mean_delta_y']:.2e}")
        print(f"   Sample std: {results['std_delta_y']:.2e}")
        print(f"   Standard error: {results['error_mean']:.2e}")
        print(f"   Significance: {results['significance']:.1f}σ")
        
        print(f"\n📏 R200 statistics:")
        print(f"   R200 range: {results['r200_range'][0]:.3f}° - {results['r200_range'][1]:.3f}°")
        print(f"   R200 median: {results['r200_median']:.3f}°")
        print(f"   Inner aperture range: {results['inner_radii_range'][0]:.3f}° - {results['inner_radii_range'][1]:.3f}°")
        print(f"   Outer aperture range: {results['outer_radii_range'][0]:.3f}° - {results['outer_radii_range'][1]:.3f}°")
        
        print(f"\n🎉 Analysis complete!")
        print(f"   Final result: Δy = {results['mean_delta_y']:.2e} ± {results['error_mean']:.2e}")
        print(f"   Significance: {results['significance']:.1f}σ")
        print(f"   Sample: {results['n_measurements']}/{results['n_input_coords']} clusters")
        print(f"   Used individual R200 for each cluster")
