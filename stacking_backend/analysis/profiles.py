# stacking_backend/analysis/profiles.py
import numpy as np

class RadialProfileCalculator:
    """Calculate radial profiles from stacked patches"""
    
    @staticmethod
    def calculate_profile(stacked_patch, patch_size_deg, n_radial_bins=20, max_radius_deg=None):
        """Calculate radial profile from stacked patch in degree units"""
        
        if stacked_patch is None:
            return None, None, None, None
        
        npix = stacked_patch.shape[0]
        pixel_size = patch_size_deg / npix
        
        if max_radius_deg is None:
            max_radius_deg = patch_size_deg / 2.0
        
        print(f"📊 Calculating radial profile...")
        print(f"   Radial bins: {n_radial_bins}, Max radius: {max_radius_deg:.1f}°")
        
        # Create coordinate grids
        center = npix // 2
        y, x = np.ogrid[:npix, :npix]
        radius_pixels = np.sqrt((x - center)**2 + (y - center)**2)
        radius_deg = radius_pixels * pixel_size
        
        # Define radial bins
        radius_bins = np.linspace(0, max_radius_deg, n_radial_bins + 1)
        radius_centers = (radius_bins[:-1] + radius_bins[1:]) / 2
        
        # Calculate profile
        profile_mean = np.zeros(n_radial_bins)
        profile_std = np.zeros(n_radial_bins)
        profile_count = np.zeros(n_radial_bins)
        
        for i in range(n_radial_bins):
            r_min, r_max = radius_bins[i], radius_bins[i + 1]
            
            # Pixels in this radial bin
            in_bin = (radius_deg >= r_min) & (radius_deg < r_max)
            
            # Get values in this bin
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
        
        return radius_centers, profile_mean, profile_error, profile_count
    
    @staticmethod
    def calculate_profile_scaled(stacked_patch, patch_size_deg, r500_deg, 
                                n_radial_bins=20, max_radius_r500=5.0):
        """Calculate radial profile from stacked patch in r/r500 units
        
        Parameters
        ----------
        stacked_patch : np.array
            2D stacked y-map
        patch_size_deg : float
            Size of patch in degrees
        r500_deg : float
            Median or characteristic r500 in degrees for scaling
        n_radial_bins : int
            Number of radial bins
        max_radius_r500 : float
            Maximum radius in units of r500
            
        Returns
        -------
        radius_centers : np.array
            Radial bin centers in r/r500 units
        profile_mean : np.array
            Mean y-parameter in each bin
        profile_error : np.array
            Standard error in each bin
        profile_count : np.array
            Number of pixels in each bin
        """
        
        if stacked_patch is None:
            return None, None, None, None
        
        npix = stacked_patch.shape[0]
        pixel_size = patch_size_deg / npix
        
        print(f"📊 Calculating radial profile in r/r500 units...")
        print(f"   Radial bins: {n_radial_bins}, Max radius: {max_radius_r500:.1f} × r500")
        print(f"   Using median r500 = {r500_deg:.3f}° for scaling")
        
        # Create coordinate grids
        center = npix // 2
        y, x = np.ogrid[:npix, :npix]
        radius_pixels = np.sqrt((x - center)**2 + (y - center)**2)
        radius_deg = radius_pixels * pixel_size
        
        # Convert to r/r500 units
        radius_scaled = radius_deg / r500_deg
        
        # Define radial bins in r/r500 units
        radius_bins = np.linspace(0, max_radius_r500, n_radial_bins + 1)
        radius_centers = (radius_bins[:-1] + radius_bins[1:]) / 2
        
        # Calculate profile
        profile_mean = np.zeros(n_radial_bins)
        profile_std = np.zeros(n_radial_bins)
        profile_count = np.zeros(n_radial_bins)
        
        for i in range(n_radial_bins):
            r_min, r_max = radius_bins[i], radius_bins[i + 1]
            
            # Pixels in this radial bin (now in r/r500 units)
            in_bin = (radius_scaled >= r_min) & (radius_scaled < r_max)
            
            # Get values in this bin
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
        print(f"   Radial range: 0 to {max_radius_r500:.1f} × r500")
        
        return radius_centers, profile_mean, profile_error, profile_count
