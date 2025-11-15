# stacking_backend/data/planck_comcomp_loader.py

import numpy as np
import healpy as hp
from astropy.io import fits
from pathlib import Path
import threading

from ..config.paths import DataPaths


class PlanckComptonyLoader:
    """
    Thread-safe loader for the official Planck Compton-y map
    (e.g. COM_CompMap_YSZ_R2.xx / milca_ymaps.fits / nilc_ymaps.fits).

    Expected DataPaths attributes:
        - data_paths.planck_y_map  : path to the Compton-y FITS file
        - data_paths.planck_y_mask : path to a mask FITS file (optional but recommended)

    Expected validate_paths() keys (you can adapt to your actual implementation):
        - validation['planck_y_map']['exists']
        - validation['planck_y_mask']['exists']
    """

    _instance = None
    _lock = threading.Lock()
    _data_cache = {}

    def __new__(cls, data_paths=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_paths=None):
        if getattr(self, "_initialized", False):
            return

        self.data_paths = data_paths or DataPaths.get_default()
        self._initialized = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_y_map_from_hdul(self, hdul):
        """
        Try to robustly extract a y-map and (optionally) half-mission maps
        from a Planck Compton-y FITS file.

        Handles:
          - Binary table with columns: FULL, HALF1, HALF2 (or similar)
          - Image HDU with a single all-sky map
        """
        hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        header = hdu.header
        data = hdu.data

        y_map = None
        y_half1 = None
        y_half2 = None

        # Case 1: Binary table with named columns
        if isinstance(data, np.recarray) or hasattr(data, "dtype") and data.dtype.names:
            colnames = [c.upper() for c in data.dtype.names]
            # Heuristics for column names
            # Prefer "FULL" if present (as in PR4-like products)
            if "FULL" in colnames:
                y_map = data[data.dtype.names[colnames.index("FULL")]]
            elif "Y" in colnames:
                y_map = data[data.dtype.names[colnames.index("Y")]]
            else:
                # Fall back to first column
                y_map = data[data.dtype.names[0]]

            # Try to grab half-mission / half-ring maps if present
            for name in colnames:
                if "HALF" in name or "HR1" in name or "HR2" in name:
                    # Very rough heuristic: store first two "half" columns
                    if y_half1 is None:
                        y_half1 = data[data.dtype.names[colnames.index(name)]]
                    elif y_half2 is None:
                        y_half2 = data[data.dtype.names[colnames.index(name)]]

        else:
            # Case 2: Image HDU
            # This should be a HEALPix map in RING or NESTED
            y_map = np.array(data).astype(float)
            y_half1 = None
            y_half2 = None

        return y_map, y_half1, y_half2, header

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_planck_comptony(self, validate_paths=True, use_cache=True):
        """
        Load the Planck Compton-y map and an optional mask, with error handling.

        Returns a dict:
            {
                'y_map'       : np.ndarray (full-sky y map),
                'y_half1'     : np.ndarray or None,
                'y_half2'     : np.ndarray or None,
                'mask'        : np.ndarray or None,
                'nside'       : int,
                'combined_mask': np.ndarray (boolean) or None
            }
        """
        if use_cache and "planck_comptony" in self._data_cache:
            print("📋 Using cached Planck Compton-y data")
            return self._data_cache["planck_comptony"]

        print("🔍 LOADING Planck COM_CompMap Compton-y MAP")
        print("=" * 60)

        if validate_paths:
            validation = self.data_paths.validate_paths()

            if not validation["planck_y_map"]["exists"]:
                raise FileNotFoundError(
                    f"Compton-y map file not found: {self.data_paths.planck_y_map}"
                )

            # Mask is optional but recommended; only warn if missing
            if "planck_y_mask" in validation:
                if not validation["planck_y_mask"]["exists"]:
                    print(f"⚠️  Warning: mask file not found: {self.data_paths.planck_y_mask}")
            else:
                print("⚠️  Warning: no 'planck_y_mask' entry in validate_paths() result.")

        try:
            # Load y-map
            y_map_path = Path(self.data_paths.planck_y_map)
            with fits.open(y_map_path) as hdul:
                y_map, y_half1, y_half2, y_header = self._load_y_map_from_hdul(hdul)

                if y_map is None:
                    raise RuntimeError("Could not extract y-map from FITS file.")

                # Heuristically get NSIDE from header or from map length
                if "NSIDE" in y_header:
                    nside = y_header["NSIDE"]
                else:
                    npix = len(y_map)
                    nside = hp.npix2nside(npix)

                coord = y_header.get("COORDSYS", "G")  # default to Galactic
                ordering = y_header.get("ORDERING", "RING")

                print(f"Y-map length: {len(y_map)}")
                print(f"NSIDE: {nside}")
                print(f"Coordinate system: {coord}")
                print(f"Ordering: {ordering}")

            # Load mask (optional)
            mask = None
            combined_mask = None

            if hasattr(self.data_paths, "planck_y_mask") and Path(self.data_paths.planck_y_mask).is_file():
                with fits.open(self.data_paths.planck_y_mask) as hdul:
                    # Either image or table
                    hdu = hdul[1] if len(hdul) > 1 else hdul[0]
                    mdata = hdu.data

                    if isinstance(mdata, np.recarray) or hasattr(mdata, "dtype") and mdata.dtype.names:
                        # If it's a table with named column(s), try to pick the first one
                        cname = mdata.dtype.names[0]
                        mask = np.array(mdata[cname]).astype(float)
                    else:
                        mask = np.array(mdata).astype(float)

                # Convert to boolean mask; assume non-zero = good
                combined_mask = mask > 0.5

                print(f"Mask length: {len(mask)}, non-zero fraction: {combined_mask.mean():.3f}")
            else:
                print("ℹ️  No Planck Compton-y mask file provided; 'mask' will be None.")

            data = {
                "y_map": y_map,
                "y_half1": y_half1,
                "y_half2": y_half2,
                "mask": mask,
                "nside": nside,
                "combined_mask": combined_mask,
            }

            if use_cache:
                self._data_cache["planck_comptony"] = data

            return data

        except Exception as e:
            raise RuntimeError(f"Failed to load Planck Compton-y data: {str(e)}") from e


def load_planck_comptony(data_paths=None, validate_paths=True, use_cache=True):
    """
    Convenience function to load the Planck COM_CompMap Compton-y data.
    """
    loader = PlanckComptonyLoader(data_paths)
    return loader.load_planck_comptony(validate_paths=validate_paths, use_cache=use_cache)

