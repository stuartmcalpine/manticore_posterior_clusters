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
                 subtract_background=True, bg_inner_radius_deg=5.0, 
                 bg_outer_radius_deg=7.0):
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
            - 'tanimura': w_i = v_i / σ²_T,i (original Tanimura estimator)
            - 'product': w_i = v_i / (σ²_T,i · σ²_v,i) (independent uncertainties)
            - 'velocity_snr': w_i = v_i · |v_i| / (σ²_T,i · σ_v,i)
            - 'velocity_snr_direct': w_i = v_i · SNR_v,i / σ²_T,i
        subtract_background : bool
            Whether to subtract background from outer annulus
        bg_inner_radius_deg : float
            Inner radius for background annulus in degrees
        bg_outer_radius_deg : float
            Outer radius for background annulus in degrees
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
                    patch_size_deg = max(patch_size_deg, 2*bg_outer_radius_deg)

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
    
        # Calculate variance of each patch (σ²_T,i)
        variances = np.nanvar(patch_stack, axis=(1, 2))
        variances = np.maximum(variances, 1e-20)  # Prevent division by zero
    
        w = valid_weights  # velocities
    
        if w is not None:
            # DEBUG - add this block temporarily
            print(f"\n  DEBUG _compute_stack:")
            print(f"    n_patches: {n_patches}")
            print(f"    velocity_weighting_scheme: {velocity_weighting_scheme}")
            print(f"    w (velocities): min={np.min(w):.1f}, max={np.max(w):.1f}, mean={np.mean(w):.1f}")
            print(f"    Fraction w < 0: {np.mean(w < 0):.1%}")
            if valid_weight_vars is not None:
                print(f"    valid_weight_vars: min={np.min(valid_weight_vars):.1f}, max={np.max(valid_weight_vars):.1f}")
            print(f"    variances (CMB): min={np.min(variances):.2e}, max={np.max(variances):.2e}")
            # END DEBUG

            # Validate weight_vars if needed for non-tanimura schemes
            if velocity_weighting_scheme != 'tanimura':
                if valid_weight_vars is None:
                    raise ValueError(
                        f"velocity_weighting_scheme='{velocity_weighting_scheme}' "
                        f"requires valid_weight_vars (velocity variances)"
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
    
            if velocity_weighting_scheme == 'tanimura':
                # Tanimura: w_i = v_i / σ²_T,i
                num_weights = w / variances
                den_weights = np.abs(w) / variances
    
            elif velocity_weighting_scheme == 'product':
                # Product: w_i = v_i / (σ²_T,i · σ²_v,i)
                num_weights = w / (variances * valid_weight_vars)
                den_weights = np.abs(w) / (variances * valid_weight_vars)
    
            elif velocity_weighting_scheme == 'velocity_snr':
                # Velocity S/N: w_i = v_i · |v_i| / (σ²_T,i · σ_v,i)
                sigma_v = np.sqrt(valid_weight_vars)
                num_weights = w * np.abs(w) / (variances * sigma_v)
                den_weights = w**2 / (variances * sigma_v)
    
            elif velocity_weighting_scheme == 'velocity_snr_direct':
                # Direct S/N: w_i = v_i · SNR_v,i / σ²_T,i
                # where SNR_v,i = |v_i| / σ_v,i
                sigma_v = np.sqrt(valid_weight_vars)
                snr_v = np.abs(w) / sigma_v
                num_weights = w * snr_v / variances
                den_weights = np.abs(w) * snr_v / variances
    
            else:
                raise ValueError(f"Unknown velocity_weighting_scheme: {velocity_weighting_scheme}")
    
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
            # Add this debug output INSIDE _compute_stack, after computing num and den
            # Right before: stacked_patch[valid] = num[valid] / den[valid]
            
            # DEBUG - trace computation
            print(f"\n  DEBUG computation:")
            print(f"    num_weights: min={np.min(num_weights):.2e}, max={np.max(num_weights):.2e}, mean={np.mean(num_weights):.2e}")
            print(f"    den_weights: min={np.min(den_weights):.2e}, max={np.max(den_weights):.2e}, mean={np.mean(den_weights):.2e}")
            print(f"    All num_weights < 0? {np.all(num_weights < 0)}")
            print(f"    All den_weights > 0? {np.all(den_weights > 0)}")
            
            # Check central pixel
            center = npix // 2
            print(f"\n  Central pixel [{center},{center}]:")
            print(f"    patch values at center: min={np.min(patch_stack[:, center, center]):.3f}, max={np.max(patch_stack[:, center, center]):.3f}, mean={np.mean(patch_stack[:, center, center]):.3f}")
            print(f"    num[center,center] = {num[center, center]:.3e}")
            print(f"    den[center,center] = {den[center, center]:.3e}")
            print(f"    result = num/den = {num[center, center] / den[center, center]:.3f}")
            # END DEBUG
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
