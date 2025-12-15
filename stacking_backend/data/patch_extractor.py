import numpy as np
import healpy as hp
from .coordinate_utils import CoordinateTransformer
import threading

class PatchExtractor:
    """Extract patches from maps with flexible coordinate systems and error handling"""
    
    def __init__(self, y_map, combined_mask=None, nside=None, 
                 nested=False, coord_system='G'):
        """
        Initialize patch extractor with map data
        
        Parameters
        ----------
        y_map : array
            Map data (HEALPix array)
        combined_mask : array, optional
            Mask data (same format as map)
        nside : int, optional
            HEALPix NSIDE (auto-detect if None)
        nested : bool
            Whether HEALPix ordering is NESTED (vs RING)
        coord_system : str
            'G' for Galactic, 'C' for Celestial/Equatorial (ICRS)
        """
        if y_map is None:
            raise ValueError("y_map cannot be None")
        
        self.y_map = y_map
        self.combined_mask = combined_mask
        self.nested = nested
        self.coord_system = coord_system
        self._lock = threading.Lock()
        
        # Auto-detect NSIDE if needed
        if nside is None:
            self.nside = hp.npix2nside(len(y_map))
        else:
            self.nside = nside
        
        # Validate
        self._validate()
        
        print(f"   Patch extractor initialized:")
        print(f"   - NSIDE: {self.nside}")
        print(f"   - Nested: {self.nested}")
        print(f"   - Coordinate system: {'Galactic' if coord_system == 'G' else 'Celestial'}")
        print(f"   - Mask available: {combined_mask is not None}")
    
    def _validate(self):
        """Validate map dimensions and consistency"""
        if self.y_map is None:
            raise ValueError("Map data cannot be None")
        
        if self.nside <= 0 or not isinstance(self.nside, int):
            raise ValueError(f"nside must be positive integer, got {self.nside}")
        
        expected_npix = hp.nside2npix(self.nside)
        if len(self.y_map) != expected_npix:
            raise ValueError(f"Map length {len(self.y_map)} doesn't match "
                             f"NSIDE {self.nside} (expected {expected_npix})")
        
        if self.combined_mask is not None:
            if len(self.combined_mask) != len(self.y_map):
                raise ValueError(f"Mask length {len(self.combined_mask)} doesn't match "
                                 f"map length {len(self.y_map)}")

    def extract_patch(self, center_coords, patch_size_deg, npix, coord_system=None):
        """
        Extract patch using direct HEALPix queries (much faster than gnomview).
        """
        lon_c, lat_c = center_coords[:2]
    
        # Sanity checks
        if not (0.0 <= lon_c <= 360.0):
            raise ValueError(f"Longitude out of range [0, 360]: {lon_c}")
        if not (-90.0 <= lat_c <= 90.0):
            raise ValueError(f"Latitude out of range [-90, 90]: {lat_c}")
        if patch_size_deg <= 0 or npix <= 0:
            raise ValueError("patch_size_deg and npix must be positive")
    
        # Create pixel grid in tangent plane
        pixel_size = patch_size_deg / npix
        x = np.linspace(-patch_size_deg/2, patch_size_deg/2, npix)
        y = np.linspace(-patch_size_deg/2, patch_size_deg/2, npix)
        xx, yy = np.meshgrid(x, y)
        
        # Convert offsets to sky coordinates using gnomonic projection
        # For small angles: ξ ≈ (lon - lon_c) * cos(lat_c), η ≈ (lat - lat_c)
        cos_lat_c = np.cos(np.radians(lat_c))
        
        # Avoid poles
        if abs(cos_lat_c) < 1e-6:
            cos_lat_c = np.sign(cos_lat_c) * 1e-6 if cos_lat_c != 0 else 1e-6
        
        lon_grid = lon_c + xx / cos_lat_c
        lat_grid = lat_c + yy
        
        # Clip to valid ranges
        lon_grid = np.clip(lon_grid, 0, 360)
        lat_grid = np.clip(lat_grid, -90, 90)
        
        # Convert to HEALPix theta, phi
        theta_grid = np.radians(90.0 - lat_grid)
        phi_grid = np.radians(lon_grid)
        
        # Query HEALPix map at these positions
        with self._lock:
            # Get pixel indices
            pix_indices = hp.ang2pix(self.nside, theta_grid, phi_grid, nest=self.nested)
            
            # Extract values
            y_patch = self.y_map[pix_indices]
            
            # Extract mask if available
            if self.combined_mask is not None:
                mask_patch = self.combined_mask[pix_indices]
            else:
                mask_patch = None
        
        return y_patch, mask_patch

    def _convert_coordinates(self, lon, lat, from_system, to_system):
        """
        Convert coordinates between Galactic ('G') and Celestial/Equatorial ('C').

        Parameters
        ----------
        lon, lat : float or array
            Coordinates to convert (degrees).
        from_system : str
            Source coordinate system ('G' or 'C').
        to_system : str
            Target coordinate system ('G' or 'C').

        Returns
        -------
        lon_out, lat_out : float or array
            Converted coordinates in degrees.
        """
        if from_system == to_system:
            return lon, lat
        
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        if from_system == 'C' and to_system == 'G':
            # Celestial (ICRS) to Galactic
            if np.isscalar(lon):
                coord = SkyCoord(ra=lon * u.deg, dec=lat * u.deg, frame='icrs')
                gal_coord = coord.galactic
                return gal_coord.l.deg, gal_coord.b.deg
            else:
                coords = SkyCoord(ra=np.asarray(lon) * u.deg,
                                  dec=np.asarray(lat) * u.deg,
                                  frame='icrs')
                gal_coords = coords.galactic
                return gal_coords.l.deg, gal_coords.b.deg
        
        elif from_system == 'G' and to_system == 'C':
            # Galactic to Celestial (ICRS)
            if np.isscalar(lon):
                coord = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame='galactic')
                eq_coord = coord.icrs
                return eq_coord.ra.deg, eq_coord.dec.deg
            else:
                coords = SkyCoord(l=np.asarray(lon) * u.deg,
                                  b=np.asarray(lat) * u.deg,
                                  frame='galactic')
                eq_coords = coords.icrs
                return eq_coords.ra.deg, eq_coords.dec.deg
        
        else:
            raise ValueError(f"Unknown coordinate conversion: {from_system} to {to_system}")

