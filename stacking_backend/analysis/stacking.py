import numpy as np
import warnings
from scipy.interpolate import RectBivariateSpline

class PatchStacker:
    """Stack multiple patches for improved signal-to-noise with r/r500 or angular scaling"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
    def stack_patches(self, coord_list, patch_size_deg=15.0, npix=256,
                 min_coverage=0.9, max_patches=None, weights=None,
                 subtract_background=True, bg_inner_radius_deg=5.0, 
                 bg_outer_radius_deg=7.0, scaling_mode='r500', 
                 max_radius_r500=10.0):
        """
        Stack multiple patches with optional r/r500 rescaling
        
        Parameters
        ----------
        coord_list : list
            List of coordinates (lon, lat, r500[, z, ...])
        patch_size_deg : float
            Size of square patch in degrees for extraction
        npix : int
            Number of pixels per side of patch
        min_coverage : float
            Minimum fraction of valid pixels required
        max_patches : int or None
            Optional maximum number of patches to stack
        weights : array-like or None
            Optional per-cluster weights (e.g. LOS velocities for kSZ)
        subtract_background : bool
            Whether to subtract background from outer annulus
        bg_inner_radius_deg : float
            Inner radius for background annulus in degrees
        bg_outer_radius_deg : float
            Outer radius for background annulus in degrees
        scaling_mode : str
            'r500': Rescale each patch to r/r500 units before stacking (Tanimura method)
            'angular': Stack in native angular (degree) coordinates
        max_radius_r500 : float
            Maximum radius in r/r500 units for rescaled stacking (only used if scaling_mode='r500')
        """

        print(f"🔄 Stacking {len(coord_list)} patches in {scaling_mode.upper()} mode...")
        
        if weights is not None:
            weights = np.asarray(weights)
            if len(weights) != len(coord_list):
                raise ValueError(
                    f"weights must have same length as coord_list "
                    f"({len(weights)} vs {len(coord_list)})"
                )
        
        valid_patches = []
        valid_coords = []
        valid_weights = [] if weights is not None else None
        valid_r500s = []  # Track R500 values for rescaling
        rejection_stats = {'insufficient_coverage': 0, 'extraction_error': 0, 
                          'rescaling_failed': 0}
        
        # Process each coordinate
        for i, coords in enumerate(coord_list):
            if max_patches and len(valid_patches) >= max_patches:
                print(f"   Reached maximum patches limit ({max_patches})")
                break
            
            try:
                # Extract coordinates
                if len(coords) >= 3:
                    lon_gal, lat_gal, r500_deg = coords[0], coords[1], coords[2]
                else:
                    print(f"   Warning: Coordinate {i} missing R500, skipping")
                    continue
                
                # Extract patch in native angular coordinates
                patch_data, mask_patch = self.patch_extractor.extract_patch(
                    center_coords=(lon_gal, lat_gal),
                    patch_size_deg=patch_size_deg,
                    npix=npix
                )
                
                # Check coverage if mask exists
                if mask_patch is not None:
                    coverage = np.mean(mask_patch)
                    if coverage < min_coverage:
                        rejection_stats['insufficient_coverage'] += 1
                        continue
                
                # Apply mask to data
                if mask_patch is not None:
                    patch_data[~mask_patch] = np.nan
                
                # Check for sufficient valid pixels
                finite_mask = np.isfinite(patch_data)
                if np.sum(finite_mask) < npix**2 * min_coverage:
                    rejection_stats['insufficient_coverage'] += 1
                    continue
                
                # Subtract background from outer ring if requested
                if subtract_background:
                    patch_data = self._subtract_background(
                        patch_data, patch_size_deg, npix, i,
                        bg_inner_radius_deg, bg_outer_radius_deg
                    )
                
                # For r/r500 mode, rescale patch to r/r500 units
                if scaling_mode == 'r500':
                    try:
                        patch_data = self._rescale_to_r500(
                            patch_data, patch_size_deg, r500_deg, 
                            npix, max_radius_r500
                        )
                    except Exception as e:
                        print(f"   Warning: Failed to rescale patch {i}: {e}")
                        rejection_stats['rescaling_failed'] += 1
                        continue
                
                valid_patches.append(patch_data)
                valid_coords.append(coords)
                valid_r500s.append(r500_deg)
                if valid_weights is not None:
                    valid_weights.append(weights[i])
                
            except Exception as e:
                print(f"   Error extracting patch {i}: {e}")
                rejection_stats['extraction_error'] += 1
                continue
        
        if not valid_patches:
            print("❌ No valid patches to stack!")
            return None, None, rejection_stats
        
        print(f"✅ Using {len(valid_patches)} valid patches")
        print(f"   Rejected: {rejection_stats['insufficient_coverage']} (coverage), "
              f"{rejection_stats['extraction_error']} (errors)")
        if scaling_mode == 'r500':
            print(f"   Rejected: {rejection_stats['rescaling_failed']} (rescaling failed)")
        
        # Stack patches
        stacked_patch, stacking_info = self._compute_stack(
            valid_patches, valid_coords, patch_size_deg, npix,
            rejection_stats, valid_weights=valid_weights,
            scaling_mode=scaling_mode, valid_r500s=valid_r500s,
            max_radius_r500=max_radius_r500
        )
        
        return stacked_patch, stacking_info, rejection_stats
    
    def _rescale_to_r500(self, patch_data, patch_size_deg, r500_deg, npix, max_radius_r500):
        """
        Rescale a patch from angular (degree) coordinates to r/r500 coordinates.
        
        This implements the Tanimura et al. method: place each cluster on a grid
        in scaled angular distance θ/θ500.
        
        Parameters
        ----------
        patch_data : np.ndarray
            Input patch in angular coordinates (degrees)
        patch_size_deg : float
            Physical size of input patch in degrees
        r500_deg : float
            R500 for this cluster in degrees
        npix : int
            Number of pixels in output scaled patch
        max_radius_r500 : float
            Maximum radius in r/r500 units for output patch
        
        Returns
        -------
        rescaled_patch : np.ndarray
            Patch rescaled to r/r500 coordinates, spanning ±max_radius_r500
        """
        # Input patch coordinates in degrees
        input_coords_deg = np.linspace(-patch_size_deg/2, patch_size_deg/2, 
                                       patch_data.shape[0])
        
        # Output patch coordinates in r/r500 units
        output_coords_r500 = np.linspace(-max_radius_r500, max_radius_r500, npix)
        
        # Convert output coordinates from r/r500 to degrees for this cluster
        output_coords_deg = output_coords_r500 * r500_deg
        
        # Create interpolator for the input patch
        # Handle NaNs by masking them during interpolation
        valid_mask = np.isfinite(patch_data)
        
        # Replace NaNs with 0 for interpolation, we'll mask them back later
        patch_for_interp = patch_data.copy()
        patch_for_interp[~valid_mask] = 0
        
        # Use RectBivariateSpline for smooth interpolation
        interp = RectBivariateSpline(input_coords_deg, input_coords_deg, 
                                     patch_for_interp, kx=1, ky=1)
        
        # Interpolate onto the r/r500 grid
        rescaled_patch = interp(output_coords_deg, output_coords_deg)
        
        # Interpolate the mask as well to know which pixels are valid
        mask_for_interp = valid_mask.astype(float)
        mask_interp = RectBivariateSpline(input_coords_deg, input_coords_deg,
                                          mask_for_interp, kx=1, ky=1)
        rescaled_mask = mask_interp(output_coords_deg, output_coords_deg)
        
        # Set pixels with low mask values to NaN (threshold at 0.5)
        rescaled_patch[rescaled_mask < 0.5] = np.nan
        
        return rescaled_patch
    
    def _subtract_background(self, patch_data, patch_size_deg, npix, patch_index,
                            bg_inner_radius_deg=5.0, bg_outer_radius_deg=7.0):
        """
        Subtract background from patch using outer annulus
        
        Parameters
        ----------
        patch_data : array
            The patch data to subtract background from
        patch_size_deg : float
            Size of the patch in degrees
        npix : int
            Number of pixels per side
        patch_index : int
            Index of the current patch (for warnings)
        bg_inner_radius_deg : float
            Inner radius of background annulus in degrees
        bg_outer_radius_deg : float
            Outer radius of background annulus in degrees
        """
        npix_half = npix // 2
        y, x = np.ogrid[:npix, :npix]
        radius = np.sqrt((x - npix_half)**2 + (y - npix_half)**2) * (patch_size_deg / npix)
        
        bg_mask = (radius > bg_inner_radius_deg) & (radius < bg_outer_radius_deg) & np.isfinite(patch_data)
        if np.sum(bg_mask) > 100:  # Ensure enough pixels
            bg_level = np.median(patch_data[bg_mask])
            patch_data -= bg_level
        else:
            print(f"   Warning: not enough background pixels for patch {patch_index}")
        
        return patch_data

    def _compute_stack(self, valid_patches, valid_coords, patch_size_deg, npix,
                       rejection_stats, valid_weights=None, scaling_mode='r500',
                       valid_r500s=None, max_radius_r500=10.0):
        """
        Compute the final stacked patch.

        If valid_weights is None:
            - Use simple unweighted mean (backwards compatible).
        If valid_weights is not None:
            - Use Tanimura-style weighted stack with variance normalization.
        """
        patch_stack = np.array(valid_patches)  # shape: (N, ny, nx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if valid_weights is None:
                # Unweighted stack
                stacked_patch = np.nanmean(patch_stack, axis=0)
                stacked_std = np.nanstd(patch_stack, axis=0)
                n_contributing = np.sum(np.isfinite(patch_stack), axis=0)
            else:
                # Weighted stack with variance normalization
                w = np.asarray(valid_weights)  # shape: (N,)

                # Compute variance for each patch
                variances = np.zeros(len(valid_patches))
                for i, patch in enumerate(valid_patches):
                    finite_values = patch[np.isfinite(patch)]
                    if len(finite_values) > 0:
                        variances[i] = np.var(finite_values)
                    else:
                        variances[i] = 1.0

                variances[variances < 1e-10] = 1e-10

                # Weight with variance normalization
                weights_over_var = w / variances
                weights_over_var_2d = weights_over_var[:, None, None]

                abs_weights_over_var = np.abs(w) / variances
                abs_weights_over_var_2d = abs_weights_over_var[:, None, None]

                finite_mask = np.isfinite(patch_stack)

                num = np.nansum(patch_stack * weights_over_var_2d, axis=0)
                den = np.sum(abs_weights_over_var_2d * finite_mask, axis=0)

                stacked_patch = np.full_like(num, np.nan, dtype=float)
                valid = den > 0
                stacked_patch[valid] = num[valid] / den[valid]

                n_contributing = np.sum(finite_mask, axis=0)
                stacked_std = np.nanstd(patch_stack, axis=0)

                print(f"   Variance-weighted stack: mean variance = {np.mean(variances):.2e}")
                print(f"   Variance range: [{np.min(variances):.2e}, {np.max(variances):.2e}]")

        # Calculate standard error
        stacked_error = stacked_std / np.sqrt(np.maximum(n_contributing, 1))

        # Determine the effective coordinate system of the stacked patch
        if scaling_mode == 'r500':
            coord_system = 'r500_units'
            extent_description = f"±{max_radius_r500:.1f} × R500"
            median_r500 = np.median(valid_r500s) if valid_r500s else None
        else:
            coord_system = 'angular_degrees'
            extent_description = f"±{patch_size_deg/2:.1f}°"
            median_r500 = None

        stacking_info = {
            'n_patches': len(valid_patches),
            'n_rejected': len(valid_coords) + sum(rejection_stats.values()) - len(valid_patches),
            'rejection_stats': rejection_stats,
            'patch_size_deg': patch_size_deg,
            'npix': npix,
            'valid_coords': valid_coords,
            'stacked_std': stacked_std,
            'stacked_error': stacked_error,
            'n_contributing': n_contributing,
            'weights_used': valid_weights is not None,
            'variance_weighted': valid_weights is not None,
            'scaling_mode': scaling_mode,
            'coord_system': coord_system,
            'extent_description': extent_description,
            'median_r500_deg': median_r500,
            'max_radius_r500': max_radius_r500 if scaling_mode == 'r500' else None
        }

        print(f"   Stack dimensions: {stacked_patch.shape}")
        print(f"   Coordinate system: {coord_system} ({extent_description})")
        print(f"   Valid pixel range: {np.nanmin(n_contributing)}-{np.nanmax(n_contributing)} patches")
        if valid_weights is not None:
            print(f"   Weighted stack: using {len(valid_patches)} weights with variance normalization")

        return stacked_patch, stacking_info
