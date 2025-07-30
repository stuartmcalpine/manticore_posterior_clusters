import numpy as np

class RadialProfileCalculator:
    """Calculate radial profiles from stacked patches"""
    
    @staticmethod
    def calculate_profile(stacked_patch, patch_size_deg, n_radial_bins=20, max_radius_deg=None):
        """Calculate radial profile from stacked patch"""
        
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
