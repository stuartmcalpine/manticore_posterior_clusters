# stacking_backend/analysis/profiles.py
import numpy as np

class RadialProfileCalculator:
    """Calculate radial profiles from stacked patches in r/r500 coordinates"""
    
    @staticmethod
    def calculate_profile(stacked_patch, patch_size_r500, n_radial_bins=20, 
                         max_radius_r500=None):
        """
        Calculate radial profile from stacked patch in r/r500 units.
        
        Parameters
        ----------
        stacked_patch : np.ndarray
            2D stacked map already in r/r500 coordinates
        patch_size_r500 : float
            Size of patch in r/r500 units (patch spans ±patch_size_r500/2)
        n_radial_bins : int
            Number of radial bins
        max_radius_r500 : float, optional
            Maximum radius in r/r500 units. If None, uses patch_size_r500/2
        
        Returns
        -------
        radius_centers : np.ndarray
            Radial bin centers in r/r500 units
        profile_mean : np.ndarray
            Mean value in each radial bin
        profile_error : np.ndarray
            Standard error in each bin
        profile_count : np.ndarray
            Number of pixels in each bin
        """
        
        if stacked_patch is None:
            return None, None, None, None
        
        npix = stacked_patch.shape[0]
        
        # Default max radius is half the patch size
        if max_radius_r500 is None:
            max_radius_r500 = patch_size_r500 / 2.0
        
        print(f"📊 Calculating radial profile in R/R500 coordinates...")
        print(f"   Radial bins: {n_radial_bins}, Max radius: {max_radius_r500:.1f} × R500")
        
        # Create coordinate grids in r/r500 units
        center = npix // 2
        y, x = np.ogrid[:npix, :npix]
        radius_pixels = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Convert pixel radius to r/r500 units
        # The patch spans ±patch_size_r500/2, so total span is patch_size_r500
        pixel_size_r500 = patch_size_r500 / (npix - 1)
        radius_r500 = radius_pixels * pixel_size_r500

        # Define radial bins in r/r500 units
        radius_bins = np.linspace(0, max_radius_r500, n_radial_bins + 1)
        radius_centers = (radius_bins[:-1] + radius_bins[1:]) / 2
        
        # Calculate profile
        profile_mean = np.zeros(n_radial_bins)
        profile_std = np.zeros(n_radial_bins)
        profile_count = np.zeros(n_radial_bins)
        
        for i in range(n_radial_bins):
            r_min, r_max = radius_bins[i], radius_bins[i + 1]
            
            in_bin = (radius_r500 >= r_min) & (radius_r500 < r_max)
            values_in_bin = stacked_patch[in_bin]
            finite_values = values_in_bin[np.isfinite(values_in_bin)]
            
            if len(finite_values) > 0:
                profile_mean[i] = np.mean(finite_values)
                profile_std[i] = np.std(finite_values)
                profile_count[i] = len(finite_values)
            else:
                profile_mean[i] = np.nan
                profile_std[i] = np.nan
                profile_count[i] = 0
        
        # Calculate standard error
        profile_error = profile_std / np.sqrt(np.maximum(profile_count, 1))
        
        # Set invalid bins to NaN
        invalid_bins = profile_count == 0
        profile_mean[invalid_bins] = np.nan
        profile_error[invalid_bins] = np.nan
        
        print(f"   Profile calculated: {np.sum(~np.isnan(profile_mean))}/{n_radial_bins} valid bins")
        print(f"   Radial range: 0 to {max_radius_r500:.1f} × R500")
        
        return radius_centers, profile_mean, profile_error, profile_count
    
    @staticmethod
    def calculate_profile_from_results(results, n_radial_bins=20, max_radius_r500=None):
        """
        Convenience function to calculate profile directly from pipeline results.
        
        Parameters
        ----------
        results : dict
            Results dictionary from run_individual_r500_analysis_with_validation
        n_radial_bins : int
            Number of radial bins
        max_radius_r500 : float, optional
            Maximum radius in r/r500 units
        
        Returns
        -------
        radius_centers, profile_mean, profile_error, profile_count : np.ndarray
            Profile statistics
        """
        
        stacked_patch = results['stacked_patch']
        patch_size_r500 = results['patch_size_r500']
        
        return RadialProfileCalculator.calculate_profile(
            stacked_patch=stacked_patch,
            patch_size_r500=patch_size_r500,
            n_radial_bins=n_radial_bins,
            max_radius_r500=max_radius_r500
        )
