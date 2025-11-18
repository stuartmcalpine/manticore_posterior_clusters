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
        Extract patch from map using healpy's gnomonic projection, assuming
        center_coords are in the same coordinate system as the map.
        This is a minimal version for debugging: no extra conversions.
        """
        lon_c, lat_c = center_coords[:2]
    
        # Sanity checks
        if not (0.0 <= lon_c <= 360.0):
            raise ValueError(f"Longitude out of range [0, 360]: {lon_c}")
        if not (-90.0 <= lat_c <= 90.0):
            raise ValueError(f"Latitude out of range [-90, 90]: {lat_c}")
        if patch_size_deg <= 0 or npix <= 0:
            raise ValueError("patch_size_deg and npix must be positive")
    
        reso_arcmin = (patch_size_deg / npix) * 60.0  # arcmin per pixel
    
        with self._lock:
            # Map patch
            y_patch = hp.gnomview(
                self.y_map,
                rot=(lon_c, lat_c, 0.0),
                xsize=npix,
                reso=reso_arcmin,
                no_plot=True,
                return_projected_map=True,
                nest=self.nested,
            )
    
            # Mask patch (if any)
            if self.combined_mask is not None:
                mask_proj = hp.gnomview(
                    self.combined_mask.astype(float),
                    rot=(lon_c, lat_c, 0.0),
                    xsize=npix,
                    reso=reso_arcmin,
                    no_plot=True,
                    return_projected_map=True,
                    nest=self.nested,
                )
                mask_patch = mask_proj > 0.5
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

