import numpy as np
from .photometry import AperturePhotometry

class IndividualClusterAnalyzer:
    """Analyze individual clusters using their own R500 values with error tracking"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
    def calculate_measurements(self, coord_list, inner_r500_factor=1.0, outer_r500_factor=3.0,
                             patch_size_deg=15.0, npix=256, min_coverage=0.9,
                             weights=None):
        """
        Calculate individual cluster measurements with error estimation.
        
        Parameters
        ----------
        coord_list : list
            List of cluster coordinates (lon, lat, r500[, z, ...])
        inner_r500_factor, outer_r500_factor : float
            Aperture definition in units of R500
        patch_size_deg : float
            Patch size (degrees)
        npix : int
            Patch resolution
        min_coverage : float
            Minimum coverage fraction
        weights : array-like or None
            Optional per-cluster weights (e.g. LOS velocities). When provided,
            the weight for each successfully measured cluster is stored in the
            result dict as 'weight'.
        """
        
        print(f"🔍 Calculating individual cluster measurements with error estimation...")
        
        if weights is not None:
            weights = np.asarray(weights)
            if len(weights) != len(coord_list):
                raise ValueError(
                    f"weights must have same length as coord_list "
                    f"({len(weights)} vs {len(coord_list)})"
                )
        
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
                
                # Perform aperture photometry with error estimation
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
                    # Calculate Y500 integration with error
                    y500_result, y500_diagnostics = AperturePhotometry.calculate_y500_integration(
                        patch=patch_data,
                        r500_deg=r500_deg,
                        patch_size_deg=patch_size_deg,
                        npix=npix,
                        background_subtract=True
                    )
                    
                    if y500_result is not None:
                        # Combine results including error estimates
                        combined_result = {**aperture_result, **y500_result}
                        combined_result['cluster_index'] = i
                        combined_result['coords'] = coords
                        
                        # Attach weight if provided
                        if weights is not None:
                            combined_result['weight'] = float(weights[i])
                        
                        # Add measurement quality metrics
                        combined_result['measurement_quality'] = {
                            'snr': aperture_result.get('snr', 0),
                            'inner_coverage': aperture_result.get('inner_coverage', 0),
                            'outer_coverage': aperture_result.get('outer_coverage', 0),
                            'n_pixels_total': aperture_result.get('n_inner_valid', 0) + 
                                            aperture_result.get('n_outer_valid', 0)
                        }
                        
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
        
        # Calculate statistics on measurement quality
        if individual_results:
            snr_values = [r['snr'] for r in individual_results]
            error_values = [r['delta_y_error'] for r in individual_results]
            
            quality_stats = {
                'mean_snr': np.mean(snr_values),
                'median_snr': np.median(snr_values),
                'mean_error': np.mean(error_values),
                'median_error': np.median(error_values),
                'high_snr_fraction': np.sum(np.array(snr_values) > 3) / len(snr_values)
            }
        else:
            quality_stats = None
        
        print(f"   ✅ Calculated {len(individual_results)} valid measurements")
        if quality_stats:
            print(f"   📊 Mean S/N: {quality_stats['mean_snr']:.2f}")
            print(f"   📊 High S/N fraction (>3σ): {quality_stats['high_snr_fraction']*100:.1f}%")
        print(f"   ❌ Rejected: {sum(rejection_stats.values())} clusters")
        for reason, count in rejection_stats.items():
            if count > 0:
                print(f"      - {reason}: {count}")
        
        return individual_results, rejection_stats, quality_stats

