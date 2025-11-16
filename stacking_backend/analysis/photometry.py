# stacking_backend/analysis/photometry.py
import numpy as np

class AperturePhotometry:
    """Perform aperture photometry on 2D patches with proper error estimation"""
    
    @staticmethod
    def create_aperture_masks(patch_size_deg, npix, inner_radius_deg, outer_radius_deg):
        """Create circular aperture masks"""
        pixel_size = patch_size_deg / npix
        center = npix // 2
        y, x = np.ogrid[:npix, :npix]
        
        radius_pixels = np.sqrt((x - center)**2 + (y - center)**2)
        radius_deg = radius_pixels * pixel_size
        
        inner_mask = radius_deg <= inner_radius_deg
        outer_mask = (radius_deg > inner_radius_deg) & (radius_deg <= outer_radius_deg)
        
        return inner_mask, outer_mask
    
    @staticmethod
    def calculate_y500_integration(patch, r500_deg, patch_size_deg, npix, background_subtract=True):
        """Calculate Y500 by integrating y-parameter within R500 with error estimation"""
        
        pixel_size_deg = patch_size_deg / npix
        center = npix // 2
        y, x = np.ogrid[:npix, :npix]
        
        radius_pixels = np.sqrt((x - center)**2 + (y - center)**2)
        radius_deg = radius_pixels * pixel_size_deg
        
        # Create R500 mask
        r500_mask = radius_deg <= r500_deg
        
        # Background subtraction from outer annulus if requested
        patch_data = patch.copy()
        bg_level = 0
        bg_error = 0
        if background_subtract:
            # Use annulus from 1.5*R500 to 2.5*R500 for background
            bg_inner = 1.5 * r500_deg
            bg_outer = 2.5 * r500_deg
            bg_mask = (radius_deg >= bg_inner) & (radius_deg <= bg_outer)
            
            if np.sum(bg_mask) > 50:  # Ensure sufficient background pixels
                bg_values = patch_data[bg_mask][np.isfinite(patch_data[bg_mask])]
                bg_level = np.median(bg_values)
                bg_error = np.std(bg_values) / np.sqrt(len(bg_values))
                patch_data -= bg_level
        
        # Calculate Y500 integration
        if np.sum(r500_mask) == 0:
            return None, {'rejection_reason': 'no_pixels_in_r500'}
        
        # Extract values within R500
        y_values = patch_data[r500_mask]
        finite_mask = np.isfinite(y_values)
        
        if np.sum(finite_mask) < 10:
            return None, {'rejection_reason': 'insufficient_finite_pixels'}
        
        # Calculate pixel solid angle in arcmin²
        pixel_solid_angle_deg2 = pixel_size_deg**2
        pixel_solid_angle_arcmin2 = pixel_solid_angle_deg2 * 3600  # deg² to arcmin²
        
        # Sum y-values with area weighting
        y_values_finite = y_values[finite_mask]
        y500_raw = np.sum(y_values_finite) * pixel_solid_angle_arcmin2
        
        # Error estimation: propagate pixel noise and background uncertainty
        pixel_variance = np.var(y_values_finite)
        n_pixels = np.sum(finite_mask)
        y500_error = np.sqrt(pixel_variance * n_pixels + (bg_error * n_pixels)**2) * pixel_solid_angle_arcmin2
        
        # Calculate effective area for error estimation
        effective_area_arcmin2 = n_pixels * pixel_solid_angle_arcmin2
        
        result = {
            'y500': y500_raw,
            'y500_error': y500_error,
            'effective_area_arcmin2': effective_area_arcmin2,
            'n_pixels': n_pixels,
            'r500_deg': r500_deg,
            'background_subtracted': background_subtract,
            'background_level': bg_level,
            'background_error': bg_error
        }
        
        diagnostics = {
            'rejection_reason': None,
            'bg_pixels': np.sum((radius_deg >= 1.5*r500_deg) & (radius_deg <= 2.5*r500_deg)) if background_subtract else 0
        }
        
        return result, diagnostics
    
    @staticmethod
    def calculate_individual_r500_photometry(patch, mask_patch, r500_deg,
                                           inner_r500_factor, outer_r500_factor,
                                           patch_size_deg, npix, min_coverage=0.9):
        """Perform aperture photometry with proper error propagation and covariance"""
        
        # Calculate aperture sizes for this specific cluster
        inner_radius_deg = inner_r500_factor * r500_deg
        outer_radius_deg = outer_r500_factor * r500_deg
        
        # Create aperture masks
        inner_mask, outer_mask = AperturePhotometry.create_aperture_masks(
            patch_size_deg, npix, inner_radius_deg, outer_radius_deg
        )
        
        # Track overlap between apertures for covariance
        overlap_mask = inner_mask & outer_mask  # Should be zero for annular outer
        
        # Apply data mask if available
        if mask_patch is not None:
            inner_mask = inner_mask & mask_patch
            outer_mask = outer_mask & mask_patch
            
            # Check coverage
            total_inner = np.sum(AperturePhotometry.create_aperture_masks(
                patch_size_deg, npix, inner_radius_deg, outer_radius_deg
            )[0])
            total_outer = np.sum(AperturePhotometry.create_aperture_masks(
                patch_size_deg, npix, inner_radius_deg, outer_radius_deg
            )[1])
            
            inner_coverage = np.sum(inner_mask) / total_inner if total_inner > 0 else 0
            outer_coverage = np.sum(outer_mask) / total_outer if total_outer > 0 else 0
            
            if inner_coverage < min_coverage or outer_coverage < min_coverage:
                return None, {'rejection_reason': 'insufficient_mask_coverage', 
                             'inner_coverage': inner_coverage, 
                             'outer_coverage': outer_coverage}
        else:
            inner_coverage = outer_coverage = 1.0
        
        # Check minimum pixel counts
        n_inner = np.sum(inner_mask)
        n_outer = np.sum(outer_mask)
        
        if n_inner < 10 or n_outer < 50:
            return None, {'rejection_reason': 'insufficient_pixels',
                         'n_inner': n_inner, 'n_outer': n_outer}
        
        # Calculate photometry
        inner_values = patch[inner_mask]
        outer_values = patch[outer_mask]
        
        # Clean NaN values
        inner_values = inner_values[np.isfinite(inner_values)]
        outer_values = outer_values[np.isfinite(outer_values)]
        
        inner_mean = np.mean(inner_values)
        outer_mean = np.mean(outer_values)
        
        inner_std = np.std(inner_values)
        outer_std = np.std(outer_values)
        
        # Y-parameter difference
        delta_y = inner_mean - outer_mean
        
        # Error propagation with covariance
        # var(A - B) = var(A) + var(B) - 2*cov(A,B)
        inner_var = inner_std**2 / len(inner_values)
        outer_var = outer_std**2 / len(outer_values)
        
        # Estimate covariance (should be small for non-overlapping annuli)
        # For overlapping regions, we'd need to calculate the actual covariance
        # Here we assume independence for non-overlapping apertures
        covariance = 0  # Conservative assumption for annular apertures
        
        # Total error on delta_y
        delta_y_variance = inner_var + outer_var - 2*covariance
        delta_y_error = np.sqrt(delta_y_variance)
        
        # Calculate signal-to-noise for this measurement
        snr = delta_y / delta_y_error if delta_y_error > 0 else 0
        
        result = {
            'delta_y': delta_y,
            'delta_y_error': delta_y_error,
            'inner_mean': inner_mean,
            'outer_mean': outer_mean,
            'inner_std': inner_std,
            'outer_std': outer_std,
            'n_inner': n_inner,
            'n_outer': n_outer,
            'n_inner_valid': len(inner_values),
            'n_outer_valid': len(outer_values),
            'inner_coverage': inner_coverage,
            'outer_coverage': outer_coverage,
            'r500_deg': r500_deg,
            'inner_radius_deg': inner_radius_deg,
            'outer_radius_deg': outer_radius_deg,
            'snr': snr,
            'covariance': covariance
        }
        
        diagnostics = {
            'inner_coverage': inner_coverage,
            'outer_coverage': outer_coverage,
            'rejection_reason': None,
            'has_overlap': np.sum(overlap_mask) > 0
        }
        
        return result, diagnostics

