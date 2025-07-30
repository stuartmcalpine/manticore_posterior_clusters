import numpy as np
import healpy as hp
from .coordinate_utils import CoordinateTransformer

class PatchExtractor:
    """Extract patches from HEALPix maps"""
    
    def __init__(self, y_map, combined_mask, nside):
        self.y_map = y_map
        self.combined_mask = combined_mask
        self.nside = nside
    
    def extract_patch(self, center_coords, patch_size_deg, npix):
        """Extract patch from y-map"""
        lon_gal, lat_gal = center_coords
        
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
