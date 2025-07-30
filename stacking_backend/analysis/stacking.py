import numpy as np
import warnings

class PatchStacker:
    """Stack multiple patches for improved signal-to-noise"""
    
    def __init__(self, patch_extractor):
        self.patch_extractor = patch_extractor
    
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
                elif len(coords) >= 4:  # (lon, lat, r200, z) or (lon, lat, r200, z, Ez) format
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
                
                # Estimate background from outer ring
                patch_data = self._subtract_background(patch_data, patch_size_deg, npix, i)
                
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
        stacked_patch, stacking_info = self._compute_stack(valid_patches, valid_coords, 
                                                          patch_size_deg, npix, rejection_stats)
        
        return stacked_patch, stacking_info, rejection_stats
    
    def _subtract_background(self, patch_data, patch_size_deg, npix, patch_index):
        """Subtract background from patch using outer annulus"""
        npix_half = npix // 2
        y, x = np.ogrid[:npix, :npix]
        radius = np.sqrt((x - npix_half)**2 + (y - npix_half)**2) * (patch_size_deg / npix)
        
        bg_mask = (radius > 5.0) & (radius < 7.0) & np.isfinite(patch_data)
        if np.sum(bg_mask) > 100:  # Ensure enough pixels
            bg_level = np.median(patch_data[bg_mask])
            patch_data -= bg_level
        else:
            print(f"   Warning: not enough background pixels for patch {patch_index}")
        
        return patch_data
    
    def _compute_stack(self, valid_patches, valid_coords, patch_size_deg, npix, rejection_stats):
        """Compute the final stacked patch"""
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
            'n_rejected': len(valid_coords) + sum(rejection_stats.values()) - len(valid_patches),
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
        
        return stacked_patch, stacking_info
