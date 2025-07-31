# stacking_backend/data/patch_extractor.py
import numpy as np
import healpy as hp
from .coordinate_utils import CoordinateTransformer
import threading

class PatchExtractor:
    """Extract patches from HEALPix maps with thread safety and error handling"""
    
    def __init__(self, y_map, combined_mask, nside):
        if y_map is None or combined_mask is None:
            raise ValueError("y_map and combined_mask cannot be None")
        if nside <= 0 or not isinstance(nside, int):
            raise ValueError(f"nside must be positive integer, got {nside}")
        
        self.y_map = y_map
        self.combined_mask = combined_mask
        self.nside = nside
        self._lock = threading.Lock()
        
        # Validate map dimensions
        expected_npix = hp.nside2npix(nside)
        if len(y_map) != expected_npix:
            raise ValueError(f"y_map length {len(y_map)} doesn't match nside {nside} (expected {expected_npix})")
        if len(combined_mask) != expected_npix:
            raise ValueError(f"combined_mask length {len(combined_mask)} doesn't match nside {nside} (expected {expected_npix})")
    
    def extract_patch(self, center_coords, patch_size_deg, npix):
        """Extract patch from y-map with thread safety and error handling"""
        
        # Input validation
        if len(center_coords) != 2:
            raise ValueError(f"center_coords must have 2 elements, got {len(center_coords)}")
        
        lon_gal, lat_gal = center_coords
        
        # Validate coordinates
        if not (0 <= lon_gal <= 360):
            raise ValueError(f"Galactic longitude out of range [0, 360]: {lon_gal}")
        if not (-90 <= lat_gal <= 90):
            raise ValueError(f"Galactic latitude out of range [-90, 90]: {lat_gal}")
        
        if patch_size_deg <= 0 or patch_size_deg > 90:
            raise ValueError(f"patch_size_deg must be in (0, 90], got {patch_size_deg}")
        
        if npix <= 0 or not isinstance(npix, int):
            raise ValueError(f"npix must be positive integer, got {npix}")
        
        with self._lock:
            try:
                # Create coordinate grid
                lon_patch, lat_patch = CoordinateTransformer.create_patch_coordinates(
                    lon_gal, lat_gal, patch_size_deg, npix
                )
                
                # Convert to HEALPix coordinates
                theta, phi = CoordinateTransformer.galactic_to_healpix(lon_patch, lat_patch)
                
                # Interpolate y-map
                y_patch = hp.get_interp_val(self.y_map, theta, phi, nest=False)
                
                # Interpolate mask
                mask_patch = hp.get_interp_val(self.combined_mask.astype(float), theta, phi, nest=False)
                mask_patch = (mask_patch > 0.5).astype(bool)
                
                return y_patch, mask_patch
                
            except Exception as e:
                raise RuntimeError(f"Failed to extract patch at ({lon_gal}, {lat_gal}): {str(e)}") from e
