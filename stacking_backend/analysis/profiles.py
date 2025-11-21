# stacking_backend/analysis/profiles.py
import numpy as np

class RadialProfileCalculator:
    """Calculate radial profiles from stacked patches in r/r500 or angular coordinates"""
    
    @staticmethod
    def calculate_profile(stacked_patch, patch_size_deg, n_radial_bins=20, 
                         max_radius_deg=None, scaling_mode='angular', 
                         max_radius_r500=None):
        """
        Calculate radial profile from stacked patch.
        
        This is a unified interface that calls either the angular or r/r500 method
        depending on the scaling_mode.
        
        Parameters
        ----------
        stacked_patch : np.ndarray
            2D stacked map
        patch_size_deg : float
            Size of patch in degrees (for angular mode)
        n_radial_bins : int
            Number of radial bins
        max_radius_deg : float, optional
            Maximum radius in degrees (for angular mode)
        scaling_mode : str
            'angular': compute profile in degrees
            'r500': compute profile in r/r500 units (patch must already be in r/r500 coords)
        max_radius_r500 : float, optional
            Maximum radius in r/r500 units (for r500 mode, defaults to extent of patch)
        
        Returns
        -------
        radius_centers : np.ndarray
            Radial bin centers (in degrees or r/r500 depending on mode)
        profile_mean : np.ndarray
            Mean value in each radial bin
        profile_error : np.ndarray
            Standard error in each bin
        profile_count : np.ndarray
            Number of pixels in each bin
        """
        
        if stacked_patch is None:
            return None, None, None, None
        
        if scaling_mode == 'angular':
            return RadialProfileCalculator._calculate_profile_angular(
                stacked_patch, patch_size_deg, n_radial_bins, max_radius_deg
            )
        elif scaling_mode == 'r500':
            return RadialProfileCalculator._calculate_profile_r500_native(
                stacked_patch, n_radial_bins, max_radius_r500
            )
        else:
            raise ValueError(f"Unknown scaling_mode: {scaling_mode}. Must be 'angular' or 'r500'")
    
    @staticmethod
    def _calculate_profile_angular(stacked_patch, patch_size_deg, n_radial_bins=20, 
                                   max_radius_deg=None):
        """Calculate radial profile in angular (degree) units"""
        
        npix = stacked_patch.shape[0]
        pixel_size = patch_size_deg / npix
        
        if max_radius_deg is None:
            max_radius_deg = patch_size_deg / 2.0
        
        print(f"📊 Calculating radial profile in ANGULAR coordinates...")
        print(f"   Radial bins: {n_radial_bins}, Max radius: {max_radius_deg:.2f}°")
        
        # Create coordinate grids
        center = npix // 2
        y, x = np.ogrid[:npix, :npix]
        radius_pixels = np.sqrt((x - center)**2 + (y - center)**2)
        radius_deg = radius_pixels * pixel_size
        
        # Define radial bins in degrees
        radius_bins = np.linspace(0, max_radius_deg, n_radial_bins + 1)
        radius_centers = (radius_bins[:-1] + radius_bins[1:]) / 2
        
        # Calculate profile
        profile_mean = np.zeros(n_radial_bins)
        profile_std = np.zeros(n_radial_bins)
        profile_count = np.zeros(n_radial_bins)
        
        for i in range(n_radial_bins):
            r_min, r_max = radius_bins[i], radius_bins[i + 1]
            
            in_bin = (radius_deg >= r_min) & (radius_deg < r_max)
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
        print(f"   Radial range: 0 to {max_radius_deg:.2f}°")
        
        return radius_centers, profile_mean, profile_error, profile_count
    
    @staticmethod
    def _calculate_profile_r500_native(stacked_patch, n_radial_bins=20, max_radius_r500=None):
        """
        Calculate radial profile from a patch that is ALREADY in r/r500 coordinates.
        
        This is used when patches have been rescaled to r/r500 units before stacking.
        No additional R500 information is needed since the patch is already in scaled units.
        
        Parameters
        ----------
        stacked_patch : np.ndarray
            2D stacked map already in r/r500 coordinates
        n_radial_bins : int
            Number of radial bins
        max_radius_r500 : float, optional
            Maximum radius in r/r500 units (if None, use patch extent)
        
        Returns
        -------
        radius_centers : np.ndarray
            Radial bin centers in r/r500 units
        profile_mean, profile_error, profile_count : np.ndarray
            Profile statistics
        """
        
        npix = stacked_patch.shape[0]
        
        # The patch spans ±max_radius_r500 in r/r500 units
        if max_radius_r500 is None:
            # Infer from patch - assume it spans full extent
            max_radius_r500 = 5.0  # Default conservative estimate
        
        print(f"📊 Calculating radial profile in R/R500 coordinates...")
        print(f"   Radial bins: {n_radial_bins}, Max radius: {max_radius_r500:.1f} × R500")
        print(f"   Patch already in r/r500 units (no additional scaling needed)")
        
        # Create coordinate grids in r/r500 units
        center = npix // 2
        y, x = np.ogrid[:npix, :npix]
        radius_pixels = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Convert pixel radius to r/r500 units
        # The patch spans ±max_radius_r500, so total span is 2*max_radius_r500
        pixel_size_r500 = (2 * max_radius_r500) / npix
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
    def calculate_profile_scaled(stacked_patch, patch_size_deg, r500_deg, 
                                n_radial_bins=20, max_radius_r500=5.0):
        """
        LEGACY method: Calculate radial profile from patch in degrees, 
        converting to r/r500 units using a reference R500.
        
        This is the OLD approach where stacking was done in degree space
        and we convert to r/r500 after the fact.
        
        For the NEW approach (Tanimura method), use calculate_profile() with
        scaling_mode='r500' on a patch that was already rescaled before stacking.
        
        Parameters
        ----------
        stacked_patch : np.ndarray
            2D stacked y-map in angular coordinates
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
        
        print(f"📊 Calculating radial profile in r/r500 units (LEGACY post-scaling)...")
        print(f"   Radial bins: {n_radial_bins}, Max radius: {max_radius_r500:.1f} × r500")
        print(f"   Using reference r500 = {r500_deg:.3f}° for post-hoc scaling")
        print(f"   ⚠️  Consider using scaling_mode='r500' for true Tanimura method")
        
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
