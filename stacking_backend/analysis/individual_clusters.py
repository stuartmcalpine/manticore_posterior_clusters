# stacking_backend/analysis/individual_clusters.py
import numpy as np
from .photometry import AperturePhotometry

class IndividualClusterAnalyzer:
    """Analyze individual clusters using their own R500 values"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
    def calculate_measurements(self, coord_list, inner_r500_factor=1.0, outer_r500_factor=3.0,
                             patch_size_deg=15.0, npix=256, min_coverage=0.9):
        """Calculate individual cluster measurements using their own R500 values"""
        
        print(f"🔍 Calculating individual cluster measurements with individual R500...")
        
        individual_results = []
        rejection_stats = {
            'insufficient_coverage': 0, 'insufficient_pixels': 0,
            'extraction_error': 0, 'invalid_format': 0,
            'y500_calculation_failed': 0
        }
        
        for i, coords in enumerate(coord_list):
            try:
                # Extract coordinates and R500
                lon_gal, lat_gal, r500_deg = coords[0], coords[1], coords[2]
                
                # Extract patch for this cluster using observed coordinates
                patch_data, mask_patch = self.patch_extractor.extract_patch(
                    center_coords=(lon_gal, lat_gal),
                    patch_size_deg=patch_size_deg,
                    npix=npix
                )
                
                if mask_patch is not None:
                    patch_data[~mask_patch] = np.nan
                
                # Perform aperture photometry with this cluster's R500
                dummy_mask = np.isfinite(patch_data)
                aperture_result, aperture_diagnostics = AperturePhotometry.calculate_individual_r500_photometry(
                    patch=patch_data,
                    mask_patch=dummy_mask,
                    r500_deg=r500_deg,  # Using R500 value directly
                    inner_r500_factor=inner_r500_factor,
                    outer_r500_factor=outer_r500_factor,
                    patch_size_deg=patch_size_deg,
                    npix=npix,
                    min_coverage=min_coverage
                )
                
                if aperture_result is not None:
                    # Calculate Y500 integration
                    y500_result, y500_diagnostics = AperturePhotometry.calculate_y500_integration(
                        patch=patch_data,
                        r500_deg=r500_deg,
                        patch_size_deg=patch_size_deg,
                        npix=npix,
                        background_subtract=True
                    )
                    
                    if y500_result is not None:
                        # Combine results
                        combined_result = {**aperture_result, **y500_result}
                        combined_result['cluster_index'] = i
                        combined_result['coords'] = coords
                        individual_results.append(combined_result)
                    else:
                        # Y500 calculation failed
                        reason = y500_diagnostics.get('rejection_reason', 'y500_calculation_failed')
                        rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
                else:
                    # Track rejection reason from aperture photometry
                    reason = aperture_diagnostics.get('rejection_reason', 'unknown')
                    rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
                    
            except Exception as e:
                print(f"   Error processing cluster {i}: {e}")
                rejection_stats['extraction_error'] += 1
                continue
        
        print(f"   ✅ Calculated {len(individual_results)} valid measurements")
        print(f"   ❌ Rejected: {sum(rejection_stats.values())} clusters")
        for reason, count in rejection_stats.items():
            if count > 0:
                print(f"      - {reason}: {count}")
        
        return individual_results, rejection_stats
