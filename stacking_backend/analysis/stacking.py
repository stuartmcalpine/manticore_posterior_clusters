import numpy as np
import warnings
from scipy.interpolate import RectBivariateSpline

class PatchStacker:
    """Stack multiple patches in r/r500 units with rescaling"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
    def stack_patches(self, coord_list, patch_size_r500=10.0, npix=256,
                 min_coverage=0.9, max_patches=None, weights=None, weight_vars=None,
                 velocity_weighting_scheme='tanimura',
                 subtract_background=False, bg_inner_r500=3.0,
                 bg_outer_r500=5.0):
        """
        Stack multiple patches with r/r500 rescaling

        Parameters
        ----------
        coord_list : list
            List of coordinates (lon, lat, r500[, z, ...])
        patch_size_r500 : float
            Size of patch in units of R500 (patch spans ±patch_size_r500/2)
        npix : int
            Number of pixels per side of patch
        min_coverage : float
            Minimum fraction of valid pixels required
        max_patches : int or None
            Optional maximum number of patches to stack
        weights : array-like or None
            Optional per-cluster weights (e.g. LOS velocities for kSZ)
        weight_vars : array-like or None
            Optional per-cluster weight variances (e.g. velocity variances from posterior)
        velocity_weighting_scheme : str
            Weighting scheme for kSZ stacking. One of:
            - 'simple': Simple velocity-weighted mean without variance weighting.
              Uses w_i = v_i. Equal weighting for all clusters regardless of noise.
              Use as a baseline or when CMB variance is uniform across patches.
            - 'tanimura': Original Tanimura et al. (2021) estimator.
              Uses w_i = v_i / σ²_T,i. Treats velocities as perfectly known.
              Use when velocities have negligible uncertainty.
            - 'optimal_posterior': Optimal estimator for velocity posteriors.
              Accounts for both CMB variance and velocity uncertainty.
              Downweights clusters with uncertain or small velocities.
              Requires weight_vars (velocity variances from posterior).
              This is the minimum-variance unbiased estimator when you have
              velocity posteriors v̂_i ~ N(v_i, σ²_v,i).
        subtract_background : bool, default=False
            Whether to subtract background from outer annulus. Default is False
            because aperture photometry already performs background subtraction at
            physical scales (1.5-2.5 × R500). Set to True only if you need to remove
            large-scale map systematics or gradients.

            IMPORTANT: When True, background is subtracted at physical scales
            (bg_inner_r500 to bg_outer_r500) to ensure consistent zero-points
            after r/r500 rescaling.
        bg_inner_r500 : float, default=3.0
            Inner radius for background annulus in units of R500. Only used when
            subtract_background=True. Default 3.0 × R500 is chosen to be outside
            typical cluster signal extent (~2 × R500).
        bg_outer_r500 : float, default=5.0
            Outer radius for background annulus in units of R500. Only used when
            subtract_background=True. Default 5.0 × R500 provides sufficient area
            for robust background estimation in noise-dominated regime.

        Notes
        -----
        Background Subtraction Strategy:
        - Default (subtract_background=False): Relies on aperture photometry
          background subtraction at 1.5-2.5 × R500, which is the standard approach
          in the literature for tSZ cluster studies.
        - When subtract_background=True: Removes large-scale modes from each patch
          at consistent physical scales before stacking. Use this when working with
          maps containing significant DC offsets or gradients.
        """

        print(f"🔄 Stacking {len(coord_list)} patches in R/R500 mode...")
        print(f"   Patch size: ±{patch_size_r500/2:.1f} × R500")
        
        if weights is not None:
            weights = np.asarray(weights)
            if len(weights) != len(coord_list):
                raise ValueError(
                    f"weights must have same length as coord_list "
                    f"({len(weights)} vs {len(coord_list)})"
                )
        
        if weight_vars is not None:
            weight_vars = np.asarray(weight_vars)
            if len(weight_vars) != len(coord_list):
                raise ValueError(
                    f"weight_vars must have same length as coord_list "
                    f"({len(weight_vars)} vs {len(coord_list)})"
                )
        
        valid_patches = []
        valid_coords = []
        valid_weights = [] if weights is not None else None
        valid_weight_vars = [] if weight_vars is not None else None
        valid_r500s = []
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
                
                # Calculate patch size in degrees for this cluster
                patch_size_deg = patch_size_r500 * r500_deg

                if subtract_background:
                    # Ensure patch is large enough to include background annulus
                    bg_outer_radius_deg = bg_outer_r500 * r500_deg
                    patch_size_deg = max(patch_size_deg, 2 * bg_outer_radius_deg)

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
                        r500_deg, bg_inner_r500, bg_outer_r500
                    )
                
                # Rescale patch to r/r500 units
                try:
                    patch_data = self._rescale_to_r500(
                        patch_data, patch_size_deg, r500_deg, 
                        npix, patch_size_r500
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
                if valid_weight_vars is not None:
                    valid_weight_vars.append(weight_vars[i])
                
            except Exception as e:
                print(f"   Error extracting patch {i}: {e}")
                rejection_stats['extraction_error'] += 1
                continue
        
        if not valid_patches:
            print("❌ No valid patches to stack!")
            return None, None, rejection_stats
        
        print(f"✅ Using {len(valid_patches)} valid patches")
        print(f"   Rejected: {rejection_stats['insufficient_coverage']} (coverage), "
              f"{rejection_stats['extraction_error']} (errors), "
              f"{rejection_stats['rescaling_failed']} (rescaling failed)")
        
        # Convert to arrays
        if valid_weights is not None:
            valid_weights = np.array(valid_weights)
        if valid_weight_vars is not None:
            valid_weight_vars = np.array(valid_weight_vars)
        
        # Stack patches
        stacked_patch, stacking_info = self._compute_stack(
            valid_patches, valid_coords, patch_size_r500, npix,
            rejection_stats, valid_weights=valid_weights,
            valid_r500s=valid_r500s, valid_weight_vars=valid_weight_vars,
            velocity_weighting_scheme=velocity_weighting_scheme
        )
        
        return stacked_patch, stacking_info, rejection_stats

    def _rescale_to_r500(self, patch_data, patch_size_deg, r500_deg, npix, patch_size_r500):
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
        patch_size_r500 : float
            Output patch size in r/r500 units (spans ±patch_size_r500/2)
    
        Returns
        -------
        rescaled_patch : np.ndarray
            Patch rescaled to r/r500 coordinates, spanning ±patch_size_r500/2
        """
        # Input patch
        ny, nx = patch_data.shape
        if ny != nx:
            raise ValueError(f"Input patch must be square, got shape {patch_data.shape}")
        n_in = ny
    
        # 1D input coordinates in degrees (native patch grid)
        input_coords_deg = np.linspace(-patch_size_deg / 2.0, patch_size_deg / 2.0, n_in)
    
        # Output patch coordinates in r/r500 units
        output_coords_r500 = np.linspace(-patch_size_r500 / 2.0, patch_size_r500 / 2.0, npix)
        # Convert output coordinates from r/r500 to degrees for this cluster
        output_coords_deg = output_coords_r500 * r500_deg
    
        # ---- Data interpolation ----
    
        # Valid mask in the input patch
        valid_mask_in = np.isfinite(patch_data)
    
        # Replace NaNs with 0 for interpolation; we'll reapply mask after
        patch_for_interp = patch_data.copy()
        patch_for_interp[~valid_mask_in] = 0.0
    
        # Interpolator on the native angular grid
        interp = RectBivariateSpline(
            input_coords_deg, input_coords_deg,
            patch_for_interp,
            kx=1, ky=1
        )
    
        # Interpolate data onto the r/r500 grid
        rescaled_patch = interp(output_coords_deg, output_coords_deg)
    
        # ---- Proper mask propagation: simple nearest neighbour ----
    
        # Pixel scale in degrees in the input patch
        pixel_scale_deg = patch_size_deg / (n_in - 1)
    
        # Precompute mapping from output coords (deg) -> input pixel indices (float)
        # index = (theta_deg - theta_min) / pixel_scale, where theta_min = -patch_size_deg/2
        xi_float = (output_coords_deg + patch_size_deg / 2.0) / pixel_scale_deg  # length npix
        yi_float = (output_coords_deg + patch_size_deg / 2.0) / pixel_scale_deg  # length npix
    
        # Nearest-neighbour integer indices
        xi_nn = np.rint(xi_float).astype(int)  # length npix
        yi_nn = np.rint(yi_float).astype(int)  # length npix
    
        # Build rescaled validity mask
        valid_rescaled = np.zeros((npix, npix), dtype=bool)
    
        for iy in range(npix):
            j = yi_nn[iy]
            if j < 0 or j >= n_in:
                # Entire row is out of bounds in y
                continue
            for ix in range(npix):
                i = xi_nn[ix]
                if i < 0 or i >= n_in:
                    # Out of bounds in x
                    continue
                # Valid if the nearest neighbour input pixel was valid
                if valid_mask_in[j, i]:
                    valid_rescaled[iy, ix] = True
    
        # Apply mask: anything whose NN parent was invalid or out-of-bounds becomes NaN
        rescaled_patch[~valid_rescaled] = np.nan
    
        return rescaled_patch
    
    
    def _subtract_background(self, patch_data, patch_size_deg, npix, patch_index,
                            r500_deg, bg_inner_r500, bg_outer_r500):
        """
        Subtract background from patch using outer annulus at physical scales

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
        r500_deg : float
            R500 of this cluster in degrees
        bg_inner_r500 : float
            Inner radius of background annulus in units of R500
        bg_outer_r500 : float
            Outer radius of background annulus in units of R500
        """
        # Convert r500 factors to degrees for this specific cluster
        bg_inner_radius_deg = bg_inner_r500 * r500_deg
        bg_outer_radius_deg = bg_outer_r500 * r500_deg

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

    def _compute_stack(self, valid_patches, valid_coords, patch_size_r500, npix,
                      rejection_stats, valid_weights=None, valid_r500s=None,
                      valid_weight_vars=None, velocity_weighting_scheme='tanimura'):
        """Compute the stacked patch with proper weighting"""
    
        patch_stack = np.array(valid_patches)
        n_patches = len(valid_patches)
        # Add right after: patch_stack = np.array(valid_patches)
        print(f"    npix parameter: {npix}")
        print(f"    patch_stack.shape: {patch_stack.shape}")
        actual_npix = patch_stack.shape[1]
        actual_center = actual_npix // 2
        print(f"    actual center: [{actual_center},{actual_center}]")
    
        # Track finite values
        finite_mask = np.isfinite(patch_stack)

        # Calculate variance of each patch (σ²_T,i) for inverse-variance weighting
        # Following Tanimura et al. (2021): variance calculated over spatial region
        # to estimate CMB+noise variance. For kSZ with pre-filtered maps (ℓ < 720),
        # signal << CMB variance (< 2%), so signal contamination is negligible.
        # This variance is used to downweight noisy patches in the stacking.
        variances = np.nanvar(patch_stack, axis=(1, 2))
        variances = np.maximum(variances, 1e-20)  # Prevent division by zero
    
        w = valid_weights  # velocities

        if w is not None:
            # Validate weight_vars if needed for optimal_posterior scheme
            if velocity_weighting_scheme == 'optimal_posterior':
                if valid_weight_vars is None:
                    raise ValueError(
                        f"velocity_weighting_scheme='optimal_posterior' "
                        f"requires valid_weight_vars (velocity variances from posterior)"
                    )
                valid_weight_vars = np.array(valid_weight_vars)
                if len(valid_weight_vars) != n_patches:
                    raise ValueError(
                        f"valid_weight_vars length mismatch: "
                        f"{len(valid_weight_vars)} vs {n_patches}"
                    )
                # Prevent division by zero
                valid_weight_vars = np.maximum(valid_weight_vars, 1e-20)

            # Compute numerator and denominator weights based on scheme
            # All schemes: result = Σ(T × num_weight) / Σ(den_weight)

            if velocity_weighting_scheme == 'simple':
                # Simple velocity-weighted mean without inverse-variance weighting
                # Stacked result: Σ[T_i(r) · v_i] / Σ[|v_i|]
                # Equal weighting for all clusters regardless of CMB noise level.
                # Use as a baseline or when CMB variance is uniform across patches.
                num_weights = w
                den_weights = np.abs(w)

            elif velocity_weighting_scheme == 'tanimura':
                # Tanimura et al. (2021): w_i = v_i / σ²_T,i
                # Assumes velocities are perfectly known (no velocity uncertainty)
                # Stacked result: Σ[T_i(r) · v_i/σ²_T,i] / Σ[|v_i|/σ²_T,i]
                num_weights = w / variances
                den_weights = np.abs(w) / variances

            elif velocity_weighting_scheme == 'optimal_posterior':
                # Optimal estimator for velocity posteriors v̂_i ~ N(v_i, σ²_v,i)
                # Derivation: var(T_i × v̂_i) ≈ σ²_T,i × v̂_i²
                # Optimal weight: w_i = 1/(σ²_T,i × v̂_i²)
                # Stacked result: Σ[T_i(r)/(σ²_T,i·v̂_i)] / Σ[1/(σ²_T,i·v̂_i²)]
                # This downweights clusters with:
                #   - High CMB noise (large σ²_T,i)
                #   - Small velocities (small |v̂_i|)
                #   - Uncertain velocities (appears via v̂_i² in denominator)
                # IMPORTANT: Numerator preserves sign of v̂_i to properly cancel velocities
                num_weights = 1.0 / (variances * w)  # Uses v̂_i with sign
                den_weights = 1.0 / (variances * w**2)  # Uses v̂_i² (always positive)

            else:
                raise ValueError(
                    f"Unknown velocity_weighting_scheme: '{velocity_weighting_scheme}'. "
                    f"Valid options: 'simple', 'tanimura', 'optimal_posterior'"
                )
    
            # Reshape for broadcasting: (n_patches,) -> (n_patches, 1, 1)
            num_weights_2d = num_weights[:, np.newaxis, np.newaxis]
            den_weights_2d = den_weights[:, np.newaxis, np.newaxis]
    
            # Compute weighted stack
            # Replace NaN with 0 for summation (masked pixels don't contribute)
            patch_stack_filled = np.where(finite_mask, patch_stack, 0.0)
    
            num = np.sum(patch_stack_filled * num_weights_2d, axis=0)
            den = np.sum(finite_mask.astype(float) * den_weights_2d, axis=0)
    
            # Compute result where denominator is non-zero
            stacked_patch = np.full((npix, npix), np.nan)
            valid = den > 0
            stacked_patch[valid] = num[valid] / den[valid]
    
            stacking_info = {
                'n_stacked': n_patches,
                'weighted': True,
                'velocity_weighting_scheme': velocity_weighting_scheme,
                'mean_weight': np.mean(w),
                'std_weight': np.std(w),
                'mean_variance': np.mean(variances),
            }
    
            if valid_weight_vars is not None:
                stacking_info['mean_weight_var'] = np.mean(valid_weight_vars)
    
        else:
            # Unweighted mean stack
            stacked_patch = np.nanmean(patch_stack, axis=0)
    
            stacking_info = {
                'n_stacked': n_patches,
                'weighted': False,
                'velocity_weighting_scheme': None,
            }
    
        return stacked_patch, stacking_info
