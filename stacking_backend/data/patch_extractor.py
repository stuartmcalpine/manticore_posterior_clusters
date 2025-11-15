# stacking_backend/data/patch_extractor.py
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
            'G' for Galactic, 'C' for Celestial/Equatorial
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
    
    def extract_patch(self, center_coords, patch_size_deg, npix, 
                     coord_system=None):
        """
        Extract patch from map at given coordinates
        
        Parameters
        ----------
        center_coords : tuple
            (lon, lat) in degrees
        patch_size_deg : float
            Patch size in degrees
        npix : int
            Number of pixels in output patch
        coord_system : str, optional
            Coordinate system of center_coords ('G' or 'C')
            If None, uses the map's coordinate system
        
        Returns
        -------
        patch_data : ndarray
            2D array of map values
        mask_patch : ndarray or None
            2D boolean array of mask values
        """
        
        # Input validation
        if len(center_coords) < 2:
            raise ValueError(f"center_coords must have at least 2 elements, got {len(center_coords)}")
        
        lon, lat = center_coords[0], center_coords[1]
        
        # Use map's coordinate system if not specified
        if coord_system is None:
            coord_system = self.coord_system
        
        # Validate coordinates based on system
        if coord_system == 'G':  # Galactic
            if not (0 <= lon <= 360):
                raise ValueError(f"Galactic longitude out of range [0, 360]: {lon}")
            if not (-90 <= lat <= 90):
                raise ValueError(f"Galactic latitude out of range [-90, 90]: {lat}")
        elif coord_system == 'C':  # Celestial
            if not (0 <= lon <= 360):
                raise ValueError(f"RA out of range [0, 360]: {lon}")
            if not (-90 <= lat <= 90):
                raise ValueError(f"Dec out of range [-90, 90]: {lat}")
        
        if patch_size_deg <= 0 or patch_size_deg > 90:
            raise ValueError(f"patch_size_deg must be in (0, 90], got {patch_size_deg}")
        
        if npix <= 0 or not isinstance(npix, int):
            raise ValueError(f"npix must be positive integer, got {npix}")
        
        with self._lock:
            try:
                # Convert coordinates if needed
                if coord_system != self.coord_system:
                    lon, lat = self._convert_coordinates(lon, lat, 
                                                        from_system=coord_system, 
                                                        to_system=self.coord_system)
                
                # Create coordinate grid
                lon_patch, lat_patch = CoordinateTransformer.create_patch_coordinates(
                    lon, lat, patch_size_deg, npix
                )
                
                # Convert to HEALPix theta, phi
                if self.coord_system == 'G':
                    # Galactic coordinates
                    theta = np.radians(90 - lat_patch)
                    phi = np.radians(lon_patch)
                elif self.coord_system == 'C':
                    # Celestial coordinates (RA/Dec)
                    theta = np.radians(90 - lat_patch)
                    phi = np.radians(lon_patch)
                else:
                    raise ValueError(f"Unknown coordinate system: {self.coord_system}")
                
                # Interpolate y-map
                y_patch = hp.get_interp_val(self.y_map, theta, phi, nest=self.nested)
                
                # Interpolate mask if available
                if self.combined_mask is not None:
                    mask_patch = hp.get_interp_val(
                        self.combined_mask.astype(float), 
                        theta, phi, nest=self.nested
                    )
                    mask_patch = (mask_patch > 0.5).astype(bool)
                else:
                    mask_patch = None
                
                return y_patch, mask_patch
                
            except Exception as e:
                raise RuntimeError(f"Failed to extract patch at ({lon}, {lat}): {str(e)}") from e
    
    def _convert_coordinates(self, lon, lat, from_system, to_system):
        """
        Convert coordinates between systems
        
        Parameters
        ----------
        lon, lat : float or array
            Coordinates to convert
        from_system : str
            Source coordinate system ('G' or 'C')
        to_system : str
            Target coordinate system ('G' or 'C')
        
        Returns
        -------
        lon_out, lat_out : float or array
            Converted coordinates
        """
        if from_system == to_system:
            return lon, lat
        
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        if from_system == 'C' and to_system == 'G':
            # Celestial to Galactic
            if np.isscalar(lon):
                coord = SkyCoord(ra=lon*u.deg, dec=lat*u.deg, frame='icrs')
                gal_coord = coord.galactic
                return gal_coord.l.deg, gal_coord.b.deg
            else:
                coords = SkyCoord(ra=lon*u.deg, dec=lat*u.deg, frame='icrs')
                gal_coords = coords.galactic
                return gal_coords.l.deg, gal_coords.b.deg
        
        elif from_system == 'G' and to_system == 'C':
            # Galactic to Celestial
            if np.isscalar(lon):
                coord = SkyCoord(l=lon*u.deg, b=lat*u.deg, frame='galactic')
                eq_coord = coord.icrs
                return eq_coord.ra.deg, eq_coord.dec.deg
            else:
                coords = SkyCoord(l=lon*u.deg, b=lat*u.deg, frame='galactic')
                eq_coords = coords.icrs
                return eq_coords.ra.deg, eq_coords.dec.deg
        
        else:
            raise ValueError(f"Unknown coordinate conversion: {from_system} to {to_system}")
