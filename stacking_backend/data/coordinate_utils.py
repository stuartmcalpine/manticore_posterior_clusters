import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

class CoordinateTransformer:
    """Handle coordinate transformations between different systems and
    construct local tangent-plane patches robustly across the sphere.
    """
    
    @staticmethod
    def radec_to_galactic(ra, dec):
        """Convert RA/Dec (ICRS) to Galactic coordinates (ℓ, b) in degrees."""
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        gal_coord = coord.galactic
        return gal_coord.l.deg, gal_coord.b.deg
    
    @staticmethod
    def galactic_to_healpix(lon_gal, lat_gal):
        """Convert Galactic (ℓ, b) in degrees to HEALPix (theta, phi) in radians."""
        theta = np.radians(90.0 - lat_gal)
        phi = np.radians(lon_gal)
        return theta, phi

    # ------------------------------------------------------------------
    # Legacy flat-sky patch generator (kept for non-critical uses)
    # ------------------------------------------------------------------
    @staticmethod
    def create_patch_coordinates(center_lon, center_lat, patch_size_deg, npix):
        """(LEGACY) Create approximate flat-sky coordinate grids around a center.

        This uses a simple small-angle approximation:
            Δlon ≈ Δx / cos(lat_center)
            Δlat ≈ Δy
        and is NOT robust near poles. Prefer `create_tangent_patch` for
        HEALPix / full-sky work.
        """
        x = np.linspace(-patch_size_deg / 2.0, patch_size_deg / 2.0, npix)
        y = np.linspace(-patch_size_deg / 2.0, patch_size_deg / 2.0, npix)
        xx, yy = np.meshgrid(x, y)

        # Small-angle approximation
        denom = np.cos(np.radians(center_lat))
        # Avoid division by zero exactly at poles; still not ideal, hence legacy.
        if np.abs(denom) < 1e-3:
            denom = np.sign(denom) * 1e-3 if denom != 0 else 1e-3

        lon_patch = center_lon + xx / denom
        lat_patch = center_lat + yy

        return lon_patch, lat_patch

    # ------------------------------------------------------------------
    # New: proper tangent-plane patch (universal, pole-safe)
    # ------------------------------------------------------------------
    @staticmethod
    def create_tangent_patch(center_lon, center_lat, patch_size_deg, npix, frame="icrs"):
        """Create a lon/lat grid for a patch using a true tangent-plane
        projection, robust near poles.

        Parameters
        ----------
        center_lon, center_lat : float
            Center coordinates in degrees, in the given frame.
        patch_size_deg : float
            Total patch size in degrees (extent in both directions).
        npix : int
            Number of pixels along each axis (npix x npix grid).
        frame : str
            Coordinate frame name understood by SkyCoord, e.g. 'icrs' or 'galactic'.

        Returns
        -------
        lon, lat : 2D ndarrays
            Longitude and latitude grids in degrees, in the given frame.
        """
        center = SkyCoord(center_lon * u.deg, center_lat * u.deg, frame=frame)
        offset_frame = center.skyoffset_frame()

        # Regular grid in the tangent plane (offset frame)
        x = np.linspace(-patch_size_deg / 2.0, patch_size_deg / 2.0, npix)
        y = np.linspace(-patch_size_deg / 2.0, patch_size_deg / 2.0, npix)
        xx, yy = np.meshgrid(x, y)

        # In the offset frame, 'lon'/'lat' are angular offsets from the center
        offsets = SkyCoord(lon=xx * u.deg, lat=yy * u.deg, frame=offset_frame)
        coords = offsets.transform_to(frame)

        lon = coords.spherical.lon.deg
        lat = coords.spherical.lat.deg
        return lon, lat

