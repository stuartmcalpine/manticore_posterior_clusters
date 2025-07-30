import numpy as np
from .pr4_data import load_pr4_data
import healpy as hp
from astropy.io import fits
import warnings

class PR4YMapAnalyzer:
    """Adapter class for PR4 NILC y-map analysis"""
    
    def __init__(self):
        self.y_map = None
        self.combined_mask = None
        self.nside = None

        self.load_pr4_data()

    def load_pr4_data(self):
        data = load_pr4_data()

        self.y_map = data["y_map"]
        self.nside = data["nside"]
        self.combined_mask = data["combined_mask"]

        print(f"✅ Loaded y-map: NSIDE={self.nside}, {len(self.y_map):,} pixels")
        print(f"✅ Combined mask: {np.mean(self.combined_mask):.1%} sky coverage")
        print(f"✅ Y-map noise level: {np.std(self.y_map):.2e}")

    def extract_patch(self, center_coords, patch_size_deg, npix):
        """Extract patch from y-map"""

        lon_gal, lat_gal = center_coords
        pixel_size = patch_size_deg / npix

        # Create coordinate grid
        x = np.linspace(-patch_size_deg/2, patch_size_deg/2, npix)
        y = np.linspace(-patch_size_deg/2, patch_size_deg/2, npix)
        xx, yy = np.meshgrid(x, y)

        # Account for coordinate distortion
        lon_patch = lon_gal + xx / np.cos(np.radians(lat_gal))
        lat_patch = lat_gal + yy

        # Convert to HEALPix coordinates (theta, phi)
        theta = np.radians(90 - lat_patch)
        phi = np.radians(lon_patch)

        # Interpolate y-map
        y_patch = hp.get_interp_val(self.y_map, theta, phi, nest=False)

        # Interpolate mask
        mask_patch = hp.get_interp_val(self.combined_mask.astype(float), theta, phi, nest=False)
        mask_patch = (mask_patch > 0.5).astype(bool)

        return y_patch, mask_patch

    def perform_aperture_photometry(self, patch, mask_patch, inner_radius_deg, outer_radius_deg,
                                  patch_size_deg, npix, min_coverage=0.9, 
                                  individual_measurements=None):  # NEW PARAMETER
        """Perform aperture photometry on y-map patch"""
    
        pixel_size = patch_size_deg / npix
        center = npix // 2
        y, x = np.ogrid[:npix, :npix]
    
        radius_pixels = np.sqrt((x - center)**2 + (y - center)**2)
        radius_deg = radius_pixels * pixel_size
    
        # Create aperture masks
        inner_mask = radius_deg <= inner_radius_deg
        outer_mask = (radius_deg > inner_radius_deg) & (radius_deg <= outer_radius_deg)
    
        # Apply data mask if available
        if mask_patch is not None:
            inner_mask = inner_mask & mask_patch
            outer_mask = outer_mask & mask_patch
    
            # Check coverage
            inner_coverage = np.sum(inner_mask) / np.sum(radius_deg <= inner_radius_deg)
            outer_coverage = np.sum(outer_mask) / np.sum((radius_deg > inner_radius_deg) &
                                                        (radius_deg <= outer_radius_deg))
    
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
    
        # Y-parameter difference (should be positive for clusters)
        delta_y = inner_mean - outer_mean
    
        if individual_measurements is not None and len(individual_measurements) > 1:
            # Use cluster-to-cluster variance (for stacked analysis)
            cluster_scatter = np.std(individual_measurements)
            error_y = cluster_scatter / np.sqrt(len(individual_measurements))
            error_type = "sample_variance"
        else:
            # Fall back to pixel-level error (for single clusters)
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

    def stack_patches(self, coord_list, patch_size_deg=15.0, npix=256,
                  min_coverage=0.9, max_patches=None):
        """Stack multiple patches and return stacked data"""

        print(f"🔄 Stacking {len(coord_list)} patches...")

        valid_patches = []
        valid_coords = []
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
                elif len(coords) == 4:  # (lon, lat, r200, z) format
                    lon_gal, lat_gal = coords[0], coords[1]
                else:
                    print(f"   Warning: Invalid coordinate format for patch {i}")
                    continue

                patch_data, mask_patch = self.extract_patch(
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

                # Estimate background from outer ring or far annulus
                npix_half = npix // 2
                y, x = np.ogrid[:npix, :npix]
                radius = np.sqrt((x - npix_half)**2 + (y - npix_half)**2) * (patch_size_deg / npix)
                
                bg_mask = (radius > 5.0) & (radius < 7.0) & np.isfinite(patch_data)
                if np.sum(bg_mask) > 100:  # Ensure enough pixels
                    bg_level = np.median(patch_data[bg_mask])
                    patch_data -= bg_level
                else:
                    print(f"   Warning: not enough background pixels for patch {i}")


                valid_patches.append(patch_data)
                valid_coords.append(coords)

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
        patch_stack = np.array(valid_patches)

        # Calculate mean, ignoring NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stacked_patch = np.nanmean(patch_stack, axis=0)
            stacked_std = np.nanstd(patch_stack, axis=0)
            n_contributing = np.sum(np.isfinite(patch_stack), axis=0)

        # Calculate standard error
        stacked_error = stacked_std / np.sqrt(np.maximum(n_contributing, 1))

        stacking_info = {
            'n_patches': len(valid_patches),
            'n_rejected': len(coord_list) - len(valid_patches),
            'rejection_stats': rejection_stats,
            'patch_size_deg': patch_size_deg,
            'npix': npix,
            'valid_coords': valid_coords,
            'stacked_std': stacked_std,
            'stacked_error': stacked_error,
            'n_contributing': n_contributing
        }

        print(f"   Stack dimensions: {stacked_patch.shape}")
        print(f"   Valid pixel range: {np.nanmin(n_contributing)}-{np.nanmax(n_contributing)} patches")

        return stacked_patch, stacking_info, rejection_stats

    def calculate_stacked_profile(self, stacked_patch, patch_size_deg, n_radial_bins=20,
                                 max_radius_deg=None):
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

    def perform_aperture_photometry_individual_r200(self, patch, mask_patch, r200_deg,
                                               inner_r200_factor, outer_r200_factor,
                                               patch_size_deg, npix, min_coverage=0.9):
        """Perform aperture photometry using individual cluster R200"""

        # Calculate aperture sizes for this specific cluster
        inner_radius_deg = inner_r200_factor * r200_deg
        outer_radius_deg = outer_r200_factor * r200_deg

        pixel_size = patch_size_deg / npix
        center = npix // 2
        y, x = np.ogrid[:npix, :npix]

        radius_pixels = np.sqrt((x - center)**2 + (y - center)**2)
        radius_deg = radius_pixels * pixel_size

        # Create aperture masks
        inner_mask = radius_deg <= inner_radius_deg
        outer_mask = (radius_deg > inner_radius_deg) & (radius_deg <= outer_radius_deg)

        # Apply data mask if available
        if mask_patch is not None:
            inner_mask = inner_mask & mask_patch
            outer_mask = outer_mask & mask_patch

            # Check coverage
            inner_coverage = np.sum(inner_mask) / np.sum(radius_deg <= inner_radius_deg)
            outer_coverage = np.sum(outer_mask) / np.sum((radius_deg > inner_radius_deg) &
                                                        (radius_deg <= outer_radius_deg))

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

        # Y-parameter difference (should be positive for clusters)
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

    def calculate_individual_cluster_measurements(self, coord_list, inner_r200_factor=1.0,
                                                outer_r200_factor=3.0, patch_size_deg=15.0,
                                                npix=256, min_coverage=0.9):
        """Calculate individual cluster measurements using their own R200 values"""

        print(f"🔍 Calculating individual cluster measurements with individual R200...")

        individual_results = []
        rejection_stats = {'insufficient_coverage': 0, 'insufficient_pixels': 0,
                          'extraction_error': 0, 'invalid_format': 0}

        for i, coords in enumerate(coord_list):
            try:
                # Extract coordinates and R200 - all clusters must have R200
                lon_gal, lat_gal, r200_deg = coords[0], coords[1], coords[2]

                # Extract patch for this cluster
                patch_data, mask_patch = self.extract_patch(
                    center_coords=(lon_gal, lat_gal),
                    patch_size_deg=patch_size_deg,
                    npix=npix
                )

                if mask_patch is not None:
                    patch_data[~mask_patch] = np.nan

                # Perform aperture photometry with this cluster's R200
                dummy_mask = np.isfinite(patch_data)
                result, diagnostics = self.perform_aperture_photometry_individual_r200(
                    patch=patch_data,
                    mask_patch=dummy_mask,
                    r200_deg=r200_deg,
                    inner_r200_factor=inner_r200_factor,
                    outer_r200_factor=outer_r200_factor,
                    patch_size_deg=patch_size_deg,
                    npix=npix,
                    min_coverage=min_coverage
                )

                if result is not None:
                    # Add cluster index for tracking
                    result['cluster_index'] = i
                    result['coords'] = coords
                    individual_results.append(result)
                else:
                    # Track rejection reason
                    reason = diagnostics.get('rejection_reason', 'unknown')
                    rejection_stats[reason] = rejection_stats.get(reason, 0) + 1

            except Exception as e:
                print(f"   Error processing cluster {i}: {e}")
                rejection_stats['extraction_error'] += 1
                continue

        print(f"   ✅ Calculated {len(individual_results)} valid measurements")
        print(f"   ❌ Rejected: {sum(rejection_stats.values())} clusters")
        for reason, count in rejection_stats.items():
            if count > 0:
                print(f"      - {reason}: {count}")

        return individual_results, rejection_stats

    def run_cluster_analysis_individual_r200(self, coord_list, inner_r200_factor=1.0, outer_r200_factor=3.0,
                                            patch_size_deg=15.0, npix=256, max_patches=None,
                                            min_coverage=0.9, n_radial_bins=20):
        """Full analysis pipeline using individual R200 values for each cluster"""

        print("🚀 CLUSTER ANALYSIS PIPELINE (Individual R200)")
        print("="*60)
        print(f"Input coordinates: {len(coord_list)}")
        print(f"Patch parameters: {patch_size_deg}° × {patch_size_deg}° ({npix}×{npix})")
        print(f"Aperture photometry: {inner_r200_factor}R200 inner, {outer_r200_factor}R200 outer")
        print(f"All clusters expected to have R200 data")

        # Step 1: Calculate individual cluster measurements with their own R200
        print(f"\n🔍 Step 1: Individual cluster measurements...")
        individual_results, rejection_stats = self.calculate_individual_cluster_measurements(
            coord_list=coord_list,
            inner_r200_factor=inner_r200_factor,
            outer_r200_factor=outer_r200_factor,
            patch_size_deg=patch_size_deg,
            npix=npix,
            min_coverage=min_coverage
        )

        if not individual_results:
            return {'success': False, 'error': 'No valid individual measurements'}

        # Extract measurements for stacking
        individual_delta_y = [result['delta_y'] for result in individual_results]
        valid_coords = [result['coords'] for result in individual_results]

        # Step 2: Create stacked patch using valid clusters
        print(f"\n📚 Step 2: Stacking patches from valid clusters...")
        stacked_patch, stacking_info, stack_rejection_stats = self.stack_patches(
            coord_list=valid_coords,  # Only use clusters that passed individual analysis
            patch_size_deg=patch_size_deg,
            npix=npix,
            min_coverage=min_coverage,
            max_patches=max_patches
        )

        if stacked_patch is None:
            return {'success': False, 'error': 'No valid patches for stacking'}

        # Step 3: Calculate radial profile
        print(f"\n📊 Step 3: Radial profile...")
        radii, profile, profile_errors, profile_counts = self.calculate_stacked_profile(
            stacked_patch=stacked_patch,
            patch_size_deg=patch_size_deg,
            n_radial_bins=n_radial_bins
        )

        # Step 4: Calculate final statistics from individual measurements
        print(f"\n🎯 Step 4: Final statistics from individual measurements...")

        # Calculate sample statistics
        mean_delta_y = np.mean(individual_delta_y)
        std_delta_y = np.std(individual_delta_y)
        error_delta_y = std_delta_y / np.sqrt(len(individual_delta_y))
        significance = mean_delta_y / error_delta_y if error_delta_y > 0 else 0

        print(f"✅ Sample statistics:")
        print(f"   Mean Δy: {mean_delta_y:.2e}")
        print(f"   Sample std: {std_delta_y:.2e}")
        print(f"   Standard error: {error_delta_y:.2e}")
        print(f"   Significance: {significance:.1f}σ")

        # Calculate R200 statistics
        r200_values = [result['r200_deg'] for result in individual_results]
        inner_radii = [result['inner_radius_deg'] for result in individual_results]
        outer_radii = [result['outer_radius_deg'] for result in individual_results]

        print(f"\n📏 R200 statistics:")
        print(f"   R200 range: {np.min(r200_values):.3f}° - {np.max(r200_values):.3f}°")
        print(f"   R200 median: {np.median(r200_values):.3f}°")
        print(f"   Inner aperture range: {np.min(inner_radii):.3f}° - {np.max(inner_radii):.3f}°")
        print(f"   Outer aperture range: {np.min(outer_radii):.3f}° - {np.max(outer_radii):.3f}°")

        # Step 5: Compile results
        results = {
            'success': True,

            # Main measurements
            'mean_delta_y': mean_delta_y,
            'error_mean': error_delta_y,
            'std_delta_y': std_delta_y,
            'significance': significance,
            'error_type': 'sample_variance_individual_r200',

            # Individual cluster results
            'individual_results': individual_results,
            'individual_measurements': individual_delta_y,

            # Stacked patch data
            'stacked_patch': stacked_patch,
            'stacking_info': stacking_info,

            # Radial profile
            'profile_radii': radii,
            'profile_mean': profile,
            'profile_errors': profile_errors,
            'profile_counts': profile_counts,

            # R200 statistics
            'r200_values': r200_values,
            'r200_median': np.median(r200_values),
            'r200_range': (np.min(r200_values), np.max(r200_values)),
            'inner_radii_range': (np.min(inner_radii), np.max(inner_radii)),
            'outer_radii_range': (np.min(outer_radii), np.max(outer_radii)),

            # Sample information
            'n_measurements': len(individual_results),
            'n_input_coords': len(coord_list),
            'n_rejected': len(coord_list) - len(individual_results),
            'rejection_stats': rejection_stats,

            # Analysis parameters
            'patch_size_deg': patch_size_deg,
            'npix': npix,
            'inner_r200_factor': inner_r200_factor,
            'outer_r200_factor': outer_r200_factor
        }

        print(f"\n🎉 Analysis complete!")
        print(f"   Final result: Δy = {results['mean_delta_y']:.2e} ± {results['error_mean']:.2e}")
        print(f"   Significance: {results['significance']:.1f}σ")
        print(f"   Sample: {results['n_measurements']}/{results['n_input_coords']} clusters")
        print(f"   Used individual R200 for each cluster")

        return results
