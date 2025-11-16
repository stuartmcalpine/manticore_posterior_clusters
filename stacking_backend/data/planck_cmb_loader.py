# stacking_backend/data/planck_cmb_loader.py

import numpy as np
import healpy as hp
from astropy.io import fits
from pathlib import Path
import threading

from ..config.paths import DataPaths


class PlanckCMBLoader:
    """
    Thread-safe loader for Planck CMB temperature maps (SMICA/NILC)
    for kSZ signal extraction from galaxy clusters.
    
    Handles files like: COM_CMB_IQU-smica_2048_R3.00_full.fits
    
    Expected DataPaths attributes:
        - data_paths.planck_cmb_map  : path to the CMB temperature FITS file
        - data_paths.planck_cmb_mask : path to a CMB mask FITS file (optional)
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
    
    def _load_cmb_map_from_hdul(self, hdul):
        """
        Extract CMB temperature (and optionally polarization) maps from 
        Planck CMB FITS files.
        
        Handles:
          - Multi-column format with I, Q, U columns
          - Multi-dimensional arrays [3, npix] for I/Q/U
          - Simple temperature-only maps
          - Half-mission splits if available
        """
        hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        header = hdu.header
        data = hdu.data
        
        t_map = None    # Temperature (I)
        q_map = None    # Q polarization (not used for kSZ)
        u_map = None    # U polarization (not used for kSZ)
        t_half1 = None  # Half-mission 1
        t_half2 = None  # Half-mission 2
        
        # Case 1: Binary table with named columns
        if isinstance(data, np.recarray) or (hasattr(data, "dtype") and data.dtype.names):
            colnames = [c.upper() for c in data.dtype.names]
            
            # Temperature map (I component)
            if "I" in colnames:
                t_map = data[data.dtype.names[colnames.index("I")]]
            elif "I_STOKES" in colnames:
                t_map = data[data.dtype.names[colnames.index("I_STOKES")]]
            elif "TEMPERATURE" in colnames:
                t_map = data[data.dtype.names[colnames.index("TEMPERATURE")]]
            elif "T" in colnames:
                t_map = data[data.dtype.names[colnames.index("T")]]
            else:
                # Fall back to first column if no recognized name
                print("⚠️  Warning: No recognized temperature column name, using first column")
                t_map = data[data.dtype.names[0]]
            
            # Optional: Q and U polarization (for completeness, though not used for kSZ)
            if "Q" in colnames:
                q_map = data[data.dtype.names[colnames.index("Q")]]
            elif "Q_STOKES" in colnames:
                q_map = data[data.dtype.names[colnames.index("Q_STOKES")]]
                
            if "U" in colnames:
                u_map = data[data.dtype.names[colnames.index("U")]]
            elif "U_STOKES" in colnames:
                u_map = data[data.dtype.names[colnames.index("U_STOKES")]]
            
            # Check for half-mission maps
            for name in colnames:
                if ("HM1" in name or "HALFMISSION1" in name or "HALF1" in name) and t_half1 is None:
                    t_half1 = data[data.dtype.names[colnames.index(name)]]
                elif ("HM2" in name or "HALFMISSION2" in name or "HALF2" in name) and t_half2 is None:
                    t_half2 = data[data.dtype.names[colnames.index(name)]]
        
        # Case 2: Multi-dimensional array [3, npix] or [n_components, npix]
        elif len(data.shape) == 2 and data.shape[0] in [1, 3]:
            # Standard I/Q/U format
            t_map = data[0]  # First component is always temperature
            if data.shape[0] >= 3:
                q_map = data[1]
                u_map = data[2]
        
        # Case 3: Simple 1D array (temperature only)
        elif len(data.shape) == 1:
            t_map = np.array(data).astype(float)
        
        else:
            # Try to handle as temperature map
            t_map = np.array(data).astype(float)
            if len(t_map.shape) > 1:
                # If multi-dimensional, take first component
                t_map = t_map.flatten()[:hp.nside2npix(2048)]
        
        return t_map, q_map, u_map, t_half1, t_half2, header
    
    def _convert_temperature_units(self, t_map, header):
        """
        Convert temperature units if necessary.
        Planck CMB maps are typically in K_CMB (CMB temperature units).
        For kSZ analysis, we often want µK_CMB.
        """
        unit = header.get('TUNIT', header.get('TUNIT1', 'K_CMB')).upper()
        
        if 'MK' in unit or 'MILLIK' in unit:
            # milliKelvin to microKelvin
            print("   Converting from mK to µK")
            return t_map * 1000.0
        elif 'UK' in unit or 'MICROK' in unit:
            # Already in microKelvin
            print("   Temperature already in µK")
            return t_map
        elif 'K' in unit:
            # Kelvin to microKelvin
            print("   Converting from K to µK")
            return t_map * 1e6
        else:
            print(f"   ⚠️  Warning: Unknown temperature unit '{unit}', assuming K_CMB")
            return t_map * 1e6
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def load_planck_cmb(self, validate_paths=True, use_cache=True, convert_to_muK=True):
        """
        Load the Planck CMB temperature map and optional mask for kSZ analysis.
        
        Parameters
        ----------
        validate_paths : bool
            Whether to validate file paths exist
        use_cache : bool
            Whether to use cached data if available
        convert_to_muK : bool
            Whether to convert temperature to microKelvin (useful for kSZ)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 't_map': Temperature map in µK (or original units if convert_to_muK=False)
            - 'q_map': Q polarization map (if available, usually None for kSZ)
            - 'u_map': U polarization map (if available, usually None for kSZ)
            - 't_half1': Half-mission 1 temperature map (if available)
            - 't_half2': Half-mission 2 temperature map (if available)
            - 'mask': Temperature mask (if provided)
            - 'combined_mask': Boolean mask for valid pixels
            - 'nside': HEALPix NSIDE
            - 'coord_system': Coordinate system ('G' or 'C')
            - 'ordering': HEALPix ordering ('RING' or 'NESTED')
            - 'units': Temperature units
        """
        cache_key = f"planck_cmb_{convert_to_muK}"
        if use_cache and cache_key in self._data_cache:
            print("📋 Using cached Planck CMB temperature data")
            return self._data_cache[cache_key]
        
        print("🔍 LOADING PLANCK CMB TEMPERATURE MAP FOR kSZ ANALYSIS")
        print("=" * 60)
        
        if validate_paths:
            validation = self.data_paths.validate_paths()
            
            if "planck_cmb_map" not in validation or not validation["planck_cmb_map"]["exists"]:
                raise FileNotFoundError(
                    f"CMB temperature map file not found: {self.data_paths.planck_cmb_map}"
                )
            
            # Mask is optional for kSZ analysis
            if "planck_cmb_mask" in validation:
                if not validation["planck_cmb_mask"]["exists"]:
                    print(f"⚠️  Warning: CMB mask file not found: {self.data_paths.planck_cmb_mask}")
            else:
                print("ℹ️  No CMB mask file specified - will use full sky or internal TMASK if available")
        
        # Will hold TMASK from the CMB map file if present
        internal_mask = None
        
        try:
            # Load CMB temperature map
            cmb_map_path = Path(self.data_paths.planck_cmb_map)
            print(f"📍 Loading: {cmb_map_path.name}")
            
            with fits.open(cmb_map_path) as hdul:
                # Print HDU information
                print(f"   HDUs in file: {len(hdul)}")
                for i, hdu in enumerate(hdul):
                    print(f"   HDU {i}: {hdu.name} - {hdu.data.shape if hdu.data is not None else 'Header only'}")
                
                # Extract temperature map
                t_map, q_map, u_map, t_half1, t_half2, header = self._load_cmb_map_from_hdul(hdul)
                
                if t_map is None:
                    raise RuntimeError("Could not extract temperature map from FITS file.")
                
                # Also attempt to extract an internal TMASK from the same HDU
                map_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
                mdata = map_hdu.data
                if isinstance(mdata, np.recarray) or (hasattr(mdata, "dtype") and mdata.dtype.names):
                    colnames = [c.upper() for c in mdata.dtype.names]
                    if "TMASK" in colnames:
                        print("   Found internal TMASK column in CMB map file")
                        internal_mask = np.array(
                            mdata[mdata.dtype.names[colnames.index("TMASK")]],
                            dtype=float
                        )
                
                # Get metadata from header
                nside = header.get("NSIDE", None)
                if nside is None:
                    npix = len(t_map)
                    nside = hp.npix2nside(npix)
                
                coord_system = header.get("COORDSYS", "G")  # Default to Galactic
                ordering = header.get("ORDERING", "RING")
                
                print(f"   Temperature map length: {len(t_map)} pixels")
                print(f"   NSIDE: {nside}")
                print(f"   Coordinate system: {coord_system}")
                print(f"   Ordering: {ordering}")
                
                # Convert units if requested
                original_units = header.get('TUNIT', header.get('TUNIT1', 'K_CMB'))
                if convert_to_muK:
                    t_map = self._convert_temperature_units(t_map, header)
                    if t_half1 is not None:
                        t_half1 = self._convert_temperature_units(t_half1, header)
                    if t_half2 is not None:
                        t_half2 = self._convert_temperature_units(t_half2, header)
                    units = "µK_CMB"
                else:
                    units = original_units
                
                # Print statistics
                finite_mask = np.isfinite(t_map)
                print(f"   Temperature range: [{np.min(t_map[finite_mask]):.2f}, "
                      f"{np.max(t_map[finite_mask]):.2f}] {units}")
                print(f"   Temperature std: {np.std(t_map[finite_mask]):.2f} {units}")
                
                # Remove monopole and dipole for kSZ analysis (optional)
                # Note: For kSZ, you typically want to keep the CMB fluctuations
                # but you might want to remove any residual dipole
                if hasattr(self.data_paths, 'remove_cmb_dipole') and self.data_paths.remove_cmb_dipole:
                    print("   Removing residual dipole...")
                    t_map = hp.remove_dipole(t_map, verbose=False)
            
            # Load mask if provided
            mask = None
            combined_mask = None
            
            # External mask file (if specified and exists)
            if hasattr(self.data_paths, "planck_cmb_mask") and Path(self.data_paths.planck_cmb_mask).is_file():
                print(f"📍 Loading CMB mask: {Path(self.data_paths.planck_cmb_mask).name}")
                
                with fits.open(self.data_paths.planck_cmb_mask) as hdul:
                    hdu = hdul[1] if len(hdul) > 1 else hdul[0]
                    mdata = hdu.data
                    
                    if isinstance(mdata, np.recarray) or (hasattr(mdata, "dtype") and mdata.dtype.names):
                        # Named columns - look for common mask names
                        colnames = [c.upper() for c in mdata.dtype.names]
                        if "MASK" in colnames:
                            mask = mdata[mdata.dtype.names[colnames.index("MASK")]]
                        elif "T_MASK" in colnames:
                            mask = mdata[mdata.dtype.names[colnames.index("T_MASK")]]
                        else:
                            # Fall back to first column if no recognized name
                            print("⚠️  Warning: No recognized mask column name, using first column as mask")
                            mask = mdata[mdata.dtype.names[0]]
                        mask = np.array(mask).astype(float)
                    else:
                        mask = np.array(mdata).astype(float)
                
                # Convert to boolean mask
                combined_mask = mask > 0.5
                
                print(f"   Mask coverage (external): {combined_mask.mean()*100:.1f}% of sky")
            
            # No external mask: try to use internal TMASK from the map file
            else:
                if internal_mask is not None:
                    print("ℹ️  No external CMB mask provided - using TMASK from CMB map file")
                    mask = internal_mask
                    combined_mask = mask > 0.5
                    print(f"   Mask coverage (internal TMASK): {combined_mask.mean()*100:.1f}% of sky")
                else:
                    print("ℹ️  No CMB mask provided and no internal TMASK found - using full sky for kSZ analysis")
                    combined_mask = np.ones(len(t_map), dtype=bool)
            
            # Package results
            data = {
                "t_map": t_map,           # Main temperature map for kSZ
                "q_map": q_map,           # Q polarization (if available)
                "u_map": u_map,           # U polarization (if available)
                "t_half1": t_half1,       # Half-mission 1 (for noise estimation)
                "t_half2": t_half2,       # Half-mission 2 (for noise estimation)
                "mask": mask,             # Raw mask values (external or internal TMASK)
                "combined_mask": combined_mask,  # Boolean mask
                "nside": nside,
                "coord_system": coord_system,
                "ordering": ordering,
                "units": units,
                "original_units": original_units
            }
            
            if use_cache:
                self._data_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Planck CMB temperature data: {str(e)}") from e


def load_planck_cmb(data_paths=None, validate_paths=True, use_cache=True, convert_to_muK=True):
    """
    Convenience function to load Planck CMB temperature maps for kSZ analysis.
    
    Parameters
    ----------
    data_paths : DataPaths, optional
        Data paths configuration
    validate_paths : bool
        Whether to validate file paths
    use_cache : bool
        Whether to use cached data
    convert_to_muK : bool
        Whether to convert temperature to microKelvin (recommended for kSZ)
    
    Returns
    -------
    dict
        Dictionary with CMB temperature map and metadata
    """
    loader = PlanckCMBLoader(data_paths)
    return loader.load_planck_cmb(validate_paths=validate_paths, 
                                  use_cache=use_cache,
                                  convert_to_muK=convert_to_muK)

