import numpy as np

class AperturePhotometry:
    """Perform aperture photometry on 2D patches"""
    
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
    def calculate_aperture_photometry(patch, mask_patch, inner_radius_deg, outer_radius_deg,
                                    patch_size_deg, npix, min_coverage=0.9, 
                                    individual_measurements=None):
        """Perform aperture photometry on y-map patch"""
        
        # Create aperture masks
        inner_mask, outer_mask = AperturePhotometry.create_aperture_masks(
            patch_size_deg, npix, inner_radius_deg, outer_radius_deg
        )
        
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
            
            inner_coverage = np.sum(inner_mask) / total_inner
            outer_coverage = np.sum(outer_mask) / total_outer
            
            if inner_coverage < min_coverage or outer_coverage < min_coverage:
                return None, {'rejection_reason': 'insufficient_mask_coverage'}
        else:
            inner_coverage = outer_coverage = 1.0
        
        # Check minimum pixel counts
        n_inner = np.sum(inner_mask)
        n_outer = np.sum(outer_mask)
        
        if n_inner < 10 or n_outer < 50:
            return None, {'rejection_reason': 'insufficient_pixels'}
        
        # Calculate photometry
        inner_values = patch[inner_mask]
        outer_values = patch[outer_mask]
        
        inner_mean = np.mean(inner_values)
        outer_mean = np.mean(outer_values)
        
        inner_std = np.std(inner_values)
        outer_std = np.std(outer_values)
        
        # Y-parameter difference
        delta_y = inner_mean - outer_mean
        
        # Calculate error
        if individual_measurements is not None and len(individual_measurements) > 1:
            cluster_scatter = np.std(individual_measurements)
            error_y = cluster_scatter / np.sqrt(len(individual_measurements))
            error_type = "sample_variance"
        else:
            inner_error = inner_std / np.sqrt(n_inner)
            outer_error = outer_std / np.sqrt(n_outer)
            error_y = np.sqrt(inner_error**2 + outer_error**2)
            error_type = "pixel_level"
        
        significance = delta_y / error_y if error_y > 0 else 0
        
        result = {
            'delta_y': delta_y,           
            'error': error_y,             
            'error_type': error_type,
            'significance': significance,  
            'inner_mean': inner_mean,
            'outer_mean': outer_mean,
            'n_inner': n_inner,
            'n_outer': n_outer,
            'inner_coverage': inner_coverage,
            'outer_coverage': outer_coverage
        }
        
        diagnostics = {
            'inner_coverage': inner_coverage,
            'outer_coverage': outer_coverage,
            'rejection_reason': None
        }
        
        return result, diagnostics
    
    @staticmethod
    def calculate_individual_r200_photometry(patch, mask_patch, r200_deg,
                                           inner_r200_factor, outer_r200_factor,
                                           patch_size_deg, npix, min_coverage=0.9):
        """Perform aperture photometry using individual cluster R200"""
        
        # Calculate aperture sizes for this specific cluster
        inner_radius_deg = inner_r200_factor * r200_deg
        outer_radius_deg = outer_r200_factor * r200_deg
        
        # Create aperture masks
        inner_mask, outer_mask = AperturePhotometry.create_aperture_masks(
            patch_size_deg, npix, inner_radius_deg, outer_radius_deg
        )
        
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
            
            inner_coverage = np.sum(inner_mask) / total_inner
            outer_coverage = np.sum(outer_mask) / total_outer
            
            if inner_coverage < min_coverage or outer_coverage < min_coverage:
                return None, {'rejection_reason': 'insufficient_mask_coverage'}
        else:
            inner_coverage = outer_coverage = 1.0
        
        # Check minimum pixel counts
        n_inner = np.sum(inner_mask)
        n_outer = np.sum(outer_mask)
        
        if n_inner < 10 or n_outer < 50:
            return None, {'rejection_reason': 'insufficient_pixels'}
        
        # Calculate photometry
        inner_values = patch[inner_mask]
        outer_values = patch[outer_mask]
        
        inner_mean = np.mean(inner_values)
        outer_mean = np.mean(outer_values)
        
        # Y-parameter difference
        delta_y = inner_mean - outer_mean
        
        result = {
            'delta_y': delta_y,
            'inner_mean': inner_mean,
            'outer_mean': outer_mean,
            'n_inner': n_inner,
            'n_outer': n_outer,
            'inner_coverage': inner_coverage,
            'outer_coverage': outer_coverage,
            'r200_deg': r200_deg,
            'inner_radius_deg': inner_radius_deg,
            'outer_radius_deg': outer_radius_deg
        }
        
        diagnostics = {
            'inner_coverage': inner_coverage,
            'outer_coverage': outer_coverage,
            'rejection_reason': None
        }
        
        return result, diagnostics
