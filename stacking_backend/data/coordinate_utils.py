import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

class CoordinateTransformer:
    """Handle coordinate transformations between different systems"""
    
    @staticmethod
    def radec_to_galactic(ra, dec):
        """Convert RA/Dec to Galactic coordinates"""
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        gal_coord = coord.galactic
        return gal_coord.l.deg, gal_coord.b.deg
    
    @staticmethod
    def galactic_to_healpix(lon_gal, lat_gal):
        """Convert Galactic coordinates to HEALPix theta, phi"""
        theta = np.radians(90 - lat_gal)
        phi = np.radians(lon_gal)
        return theta, phi
    
    @staticmethod
    def create_patch_coordinates(center_lon, center_lat, patch_size_deg, npix):
        """Create coordinate grids for patch extraction"""
        x = np.linspace(-patch_size_deg/2, patch_size_deg/2, npix)
        y = np.linspace(-patch_size_deg/2, patch_size_deg/2, npix)
        xx, yy = np.meshgrid(x, y)
        
        # Account for coordinate distortion
        lon_patch = center_lon + xx / np.cos(np.radians(center_lat))
        lat_patch = center_lat + yy
        
        return lon_patch, lat_patch
