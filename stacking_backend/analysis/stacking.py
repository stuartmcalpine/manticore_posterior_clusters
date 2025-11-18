import numpy as np
import warnings

class PatchStacker:
    """Stack multiple patches for improved signal-to-noise"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
    def stack_patches(self, coord_list, patch_size_deg=15.0, npix=256,
                 min_coverage=0.9, max_patches=None, weights=None,
                 subtract_background=True, bg_inner_radius_deg=5.0, 
                 bg_outer_radius_deg=7.0):
        """
        Stack multiple patches and return stacked data
        
        Parameters
        ----------
        coord_list : list
            List of coordinates (and optionally r500, z, etc.)
        patch_size_deg : float
            Size of square patch in degrees
        npix : int
            Number of pixels per side of patch
        min_coverage : float
            Minimum fraction of valid pixels required
        max_patches : int or None
            Optional maximum number of patches to stack
        weights : array-like or None
            Optional per-cluster weights (e.g. LOS velocities for kSZ).
            If None, a simple unweighted average is used (backwards compatible).
        subtract_background : bool
            Whether to subtract background from outer annulus (default: True)
        bg_inner_radius_deg : float
            Inner radius for background annulus in degrees (default: 5.0)
        bg_outer_radius_deg : float
            Outer radius for background annulus in degrees (default: 7.0)
        """

        print(f"🔄 Stacking {len(coord_list)} patches...")
        
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
        rejection_stats = {'insufficient_coverage': 0, 'extraction_error': 0}
        
        # Process each coordinate
        for i, coords in enumerate(coord_list):
            if max_patches and len(valid_patches) >= max_patches:
                print(f"   Reached maximum patches limit ({max_patches})")
                break
            
            try:
                # Extract patch
                if len(coords) == 2:
                    lon_gal, lat_gal = coords
                elif len(coords) >= 4:  # (lon, lat, r500, z) or (lon, lat, r500, z, Ez) format
                    lon_gal, lat_gal = coords[0], coords[1]
                else:
                    print(f"   Warning: Invalid coordinate format for patch {i}")
                    continue
                
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
                
                valid_patches.append(patch_data)
                valid_coords.append(coords)
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
        
        # Stack patches
        stacked_patch, stacking_info = self._compute_stack(
            valid_patches, valid_coords, patch_size_deg, npix,
            rejection_stats, valid_weights=valid_weights
        )
        
        return stacked_patch, stacking_info, rejection_stats
    
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
                       rejection_stats, valid_weights=None):
        """
        Compute the final stacked patch.

        If valid_weights is None:
            - Use simple unweighted mean (backwards compatible).
        If valid_weights is not None:
            - Use Tanimura-style weighted stack with variance normalization:
              T_stack(x,y) = sum_i (w_i/σᵢ²) T_i(x,y) / sum_i |w_i|/σᵢ² over valid pixels.
        """
        patch_stack = np.array(valid_patches)  # shape: (N, ny, nx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if valid_weights is None:
                # Unweighted stack (tSZ / legacy behaviour)
                stacked_patch = np.nanmean(patch_stack, axis=0)
                stacked_std = np.nanstd(patch_stack, axis=0)
                n_contributing = np.sum(np.isfinite(patch_stack), axis=0)
            else:
                # Weighted stack with variance normalization (Tanimura et al. kSZ method)
                w = np.asarray(valid_weights)  # shape: (N,)

                # Compute variance for each patch (σᵢ² in Tanimura's notation)
                variances = np.zeros(len(valid_patches))
                for i, patch in enumerate(valid_patches):
                    finite_values = patch[np.isfinite(patch)]
                    if len(finite_values) > 0:
                        variances[i] = np.var(finite_values)
                    else:
                        variances[i] = 1.0  # Default to avoid division by zero

                # Replace any zero or very small variances to avoid division issues
                variances[variances < 1e-10] = 1e-10

                # Create 2D weight arrays with variance normalization
                # Weight is w_i / σᵢ² for numerator, |w_i| / σᵢ² for denominator
                weights_over_var = w / variances  # shape: (N,)
                weights_over_var_2d = weights_over_var[:, None, None]  # shape: (N, ny, nx)

                abs_weights_over_var = np.abs(w) / variances  # shape: (N,)
                abs_weights_over_var_2d = abs_weights_over_var[:, None, None]  # shape: (N, ny, nx)

                finite_mask = np.isfinite(patch_stack)

                # Numerator: sum_i (w_i/σᵢ²) * T_i over valid pixels
                num = np.nansum(patch_stack * weights_over_var_2d, axis=0)

                # Denominator per pixel: sum_i |w_i|/σᵢ² for clusters that contribute at that pixel
                den = np.sum(abs_weights_over_var_2d * finite_mask, axis=0)

                stacked_patch = np.full_like(num, np.nan, dtype=float)
                valid = den > 0
                stacked_patch[valid] = num[valid] / den[valid]

                # For diagnostics: how many clusters contributed at each pixel
                n_contributing = np.sum(finite_mask, axis=0)

                # Keep a simple std definition for diagnostics
                stacked_std = np.nanstd(patch_stack, axis=0)

                # Log the variance weighting info
                print(f"   Variance-weighted stack: mean variance = {np.mean(variances):.2e}")
                print(f"   Variance range: [{np.min(variances):.2e}, {np.max(variances):.2e}]")

        # Calculate standard error
        stacked_error = stacked_std / np.sqrt(np.maximum(n_contributing, 1))

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
            'variance_weighted': valid_weights is not None  # Flag for Tanimura-style weighting
        }

        print(f"   Stack dimensions: {stacked_patch.shape}")
        print(f"   Valid pixel range: {np.nanmin(n_contributing)}-{np.nanmax(n_contributing)} patches")
        if valid_weights is not None:
            print(f"   Weighted stack: using {len(valid_patches)} weights with variance normalization")

        return stacked_patch, stacking_info
