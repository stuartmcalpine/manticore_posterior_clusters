import numpy as np
from .photometry import AperturePhotometry

class IndividualClusterAnalyzer:
    """Analyze individual clusters using their own R200 values"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
    def calculate_measurements(self, coord_list, inner_r200_factor=1.0, outer_r200_factor=3.0,
                             patch_size_deg=15.0, npix=256, min_coverage=0.9):
        """Calculate individual cluster measurements using their own R200 values"""
        
        print(f"🔍 Calculating individual cluster measurements with individual R200...")
        
        individual_results = []
        rejection_stats = {
            'insufficient_coverage': 0, 'insufficient_pixels': 0,
            'extraction_error': 0, 'invalid_format': 0
        }
        
        for i, coords in enumerate(coord_list):
            try:
                # Extract coordinates and R200
                lon_gal, lat_gal, r200_deg = coords[0], coords[1], coords[2]
                
                # Extract patch for this cluster
                patch_data, mask_patch = self.patch_extractor.extract_patch(
                    center_coords=(lon_gal, lat_gal),
                    patch_size_deg=patch_size_deg,
                    npix=npix
                )
                
                if mask_patch is not None:
                    patch_data[~mask_patch] = np.nan
                
                # Perform aperture photometry with this cluster's R200
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
                    # Add cluster index for tracking
                    result['cluster_index'] = i
                    result['coords'] = coords
                    individual_results.append(result)
                else:
                    # Track rejection reason
                    reason = diagnostics.get('rejection_reason', 'unknown')
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
