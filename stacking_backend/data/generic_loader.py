# stacking_backend/data/generic_loader.py
import numpy as np
import healpy as hp
from astropy.io import fits
from pathlib import Path
from typing import Dict, Optional, Tuple
import threading
from ..config.map_config import MapConfig, MapFormat

class GenericMapLoader:
    """Flexible map loader for various formats"""
    
    _cache = {}
    _cache_lock = threading.Lock()
    
    def __init__(self, config: MapConfig):
        self.config = config
        self._validate_config()

    def _load_cl_from_file(self, path: str, lmax: int) -> np.ndarray:
        """
        Load C_ell from a file.

        Accepted formats:
        - 1D array saved with np.save
        - Text file with either:
          * one column: C_ell for ell = 0..N-1
          * two columns: ell, C_ell

        Returns array of length >= lmax+1; truncated if longer.
        """
        if path is None:
            raise ValueError("C_ell path is None but matched filter requested")

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"C_ell file not found: {path}")

        try:
            arr = np.load(p)
        except Exception:
            arr = np.loadtxt(p)

        arr = np.atleast_1d(arr)

        # If 2D (ell, Cl)
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                cl = arr[:, 0]
            else:
                # assume first column ell, second column C_ell
                cl = arr[:, 1]
        else:
            cl = arr

        if cl.shape[0] <= lmax:
            # pad with zeros if needed
            cl_padded = np.zeros(lmax + 1)
            cl_padded[:cl.shape[0]] = cl
            return cl_padded
        else:
            return cl[:lmax + 1]

    def _apply_tanimura_filter(self, map_data: np.ndarray) -> np.ndarray:
        """
        Apply the Tanimura et al.-style ℓ-space high-pass filter:

            - W_l = 0 for l < ell_filter_lmin
            - W_l = 1 for l > ell_filter_lmax
            - cosine ramp between ell_filter_lmin and ell_filter_lmax

        The filter operates in harmonic space and preserves map units (e.g. µK).
        """
        # Ensure we know nside
        nside = self.config.nside or hp.get_nside(map_data)
        lmax = 3 * nside - 1

        l1 = self.config.ell_filter_lmin
        l2 = self.config.ell_filter_lmax
        if not (0 <= l1 < l2 <= lmax):
            raise ValueError(
                f"Invalid (ell_filter_lmin, ell_filter_lmax)=({l1},{l2}) for lmax={lmax}"
            )

        ell = np.arange(lmax + 1, dtype=float)

        # Build W_l
        W = np.ones_like(ell, dtype=float)
        W[ell < l1] = 0.0

        mid = (ell >= l1) & (ell <= l2)
        ramp = (ell[mid] - l1) / (l2 - l1)          # 0 → 1
        W[mid] = 0.5 * (1.0 - np.cos(np.pi * ramp))  # smooth cosine 0→1

        # Alm transforms expect RING ordering
        if self.config.nested:
            print("   Reordering map NESTED → RING for harmonic filtering")
            map_ring = hp.reorder(map_data, n2r=True)
        else:
            map_ring = map_data

        # Forward transform
        alm = hp.map2alm(map_ring, lmax=lmax, iter=0)
        # Apply filter in-place
        hp.almxfl(alm, W, inplace=True)
        # Back to map
        filtered_ring = hp.alm2map(alm, nside, verbose=False)

        if self.config.nested:
            print("   Reordering filtered map RING → NESTED")
            filtered_map = hp.reorder(filtered_ring, r2n=True)
        else:
            filtered_map = filtered_ring

        var_before = np.nanvar(map_data)
        var_after = np.nanvar(filtered_map)
        print(f"   Tanimura filter variance: before={var_before:.3e}, "
              f"after={var_after:.3e} (ratio={var_after/var_before:.3f})")

        return filtered_map

    def _apply_matched_filter(self, map_data: np.ndarray) -> np.ndarray:
        """
        Apply a simple inverse-variance harmonic filter:
    
            F_l ∝ 1 / (C_l^CMB + N_l)
    
        This suppresses large-scale primary CMB (where CMB dominates)
        while leaving small scales (where noise dominates) mostly intact.
    
        We normalize so that max(F_l) = 1 to keep amplitudes reasonable.
        """
        if self.config.nside is None:
            self.config.nside = hp.get_nside(map_data)
        nside = self.config.nside
    
        lmax = self.config.matched_filter_lmax
        if lmax is None:
            lmax = 3 * nside - 1
    
        print(f"   Applying inverse-variance harmonic filter: lmax={lmax}")
    
        # Load CMB + noise spectra
        cl_cmb = self._load_cl_from_file(self.config.matched_filter_cmb_cl_path, lmax)
        cl_noise = self._load_cl_from_file(self.config.matched_filter_noise_cl_path, lmax)
    
        ell = np.arange(lmax + 1, dtype=float)
        cl_tot = cl_cmb + cl_noise
    
        # Avoid division by zero
        safe = cl_tot > 0
        f_l = np.zeros_like(ell, dtype=float)
        f_l[safe] = 1.0 / cl_tot[safe]
    
        # Normalize to max(F_l) = 1 so we don't wipe out amplitudes
        max_f = np.max(f_l[safe])
        if max_f > 0:
            f_l /= max_f
    
        # Transform map -> alm -> apply filter -> back to map
        alm = hp.map2alm(map_data, lmax=lmax)
        alm_filt = hp.almxfl(alm, f_l)
        map_filt = hp.alm2map(alm_filt, nside=nside, verbose=False)
    
        return map_filt


    def _apply_ell_filter(self, map_data: np.ndarray) -> np.ndarray:
        """
        Apply a harmonic-space filter to a HEALPix map.

        Uses a raised-cosine high-pass:
        - ℓ < ell_filter_lmin : 0
        - ℓ > ell_filter_lmax : 1
        - smooth transition between lmin and lmax

        If ell_filter_lmin/lmax are not set, defaults are chosen based on NSIDE.
        """
        if self.config.nside is None:
            # Try to infer NSIDE if not already set
            self.config.nside = hp.get_nside(map_data)
        
        nside = self.config.nside
        
        # Decide lmax
        lmax = self.config.lmax if self.config.lmax is not None else (3 * nside - 1)
        
        # Defaults for lmin, lmax if not provided
        lmin = self.config.ell_filter_lmin if self.config.ell_filter_lmin is not None else int(0.5 * nside)
        lmax_trans = self.config.ell_filter_lmax if self.config.ell_filter_lmax is not None else int(1.0 * nside)
        
        print(f"   Applying ℓ-space filter: lmin={lmin}, lmax={lmax_trans}, lmax_tot={lmax}")
        
        ell = np.arange(lmax + 1, dtype=float)
        fl = np.ones_like(ell)
        
        # Below lmin -> 0
        fl[ell < lmin] = 0.0
        
        # Between lmin and lmax_trans -> smooth raised cosine
        in_trans = (ell >= lmin) & (ell <= lmax_trans)
        if np.any(in_trans) and lmax_trans > lmin:
            x = (ell[in_trans] - lmin) / (lmax_trans - lmin)
            fl[in_trans] = 0.5 * (1.0 - np.cos(np.pi * x))
        
        # Above lmax_trans -> 1 (already default)
        
        # Map is assumed in RING ordering here
        alm = hp.map2alm(map_data, lmax=lmax)
        alm_f = hp.almxfl(alm, fl)
        map_filtered = hp.alm2map(alm_f, nside=nside, verbose=False)
        
        return map_filtered

    def _validate_config(self):
        """Validate that files exist"""
        if not Path(self.config.map_path).exists():
            raise FileNotFoundError(f"Map file not found: {self.config.map_path}")
        
        if self.config.mask_path and not Path(self.config.mask_path).exists():
            raise FileNotFoundError(f"Mask file not found: {self.config.mask_path}")
    
    def load_data(self, use_cache: bool = False) -> Dict:
        """Load map and mask data based on configuration"""
        
        # Check cache
        cache_key = (self.config.map_path, self.config.mask_path)
        if use_cache:
            with self._cache_lock:
                if cache_key in self._cache:
                    print("📋 Using cached map data")
                    return self._cache[cache_key]
        
        print("🔍 LOADING MAP DATA")
        print("="*50)
        print(f"📍 Map file: {Path(self.config.map_path).name}")
        if self.config.mask_path:
            print(f"📍 Mask file: {Path(self.config.mask_path).name}")
        print(f"📍 Format: {self.config.map_format.value}")
        print(f"📍 Coordinate system: {'Galactic' if self.config.coord_system == 'G' else 'Celestial'}")
        
        if self.config.map_format == MapFormat.HEALPIX:
            data = self._load_healpix_data()
        elif self.config.map_format == MapFormat.FLAT:
            data = self._load_flat_data()
        elif self.config.map_format == MapFormat.FITS_IMAGE:
            data = self._load_fits_image_data()
        else:
            raise ValueError(f"Unsupported map format: {self.config.map_format}")
        
        # Cache the data
        if use_cache:
            with self._cache_lock:
                self._cache[cache_key] = data
        
        return data
    
    def _load_healpix_data(self) -> Dict:
        """Load HEALPix format data"""
        
        # Load map
        if self.config.map_column:
            # Multi-column FITS (like Planck PR4)
            with fits.open(self.config.map_path) as hdul:
                data = hdul[self.config.map_hdu].data
                header = hdul[self.config.map_hdu].header
                
                print(f"   Available columns: {data.dtype.names}")
                
                if self.config.map_column not in data.dtype.names:
                    raise ValueError(f"Column '{self.config.map_column}' not found. "
                                   f"Available: {data.dtype.names}")
                
                map_data = data[self.config.map_column]
                
                # Get NSIDE from header if not specified
                if self.config.nside is None:
                    self.config.nside = header.get('NSIDE', None)
                    if self.config.nside is None:
                        # Try to infer from map size
                        self.config.nside = hp.npix2nside(len(map_data))
                
                print(f"   Using column: {self.config.map_column}")
                print(f"   NSIDE: {self.config.nside}")
                print(f"   Ordering: {header.get('ORDERING', 'RING')}")
                
                # Check for half-mission maps if available (for Planck)
                map_half1 = data.get('HALF-RING 1', None) if hasattr(data, 'get') else None
                map_half2 = data.get('HALF-RING 2', None) if hasattr(data, 'get') else None
        else:
            # Simple HEALPix map (single array)
            map_data, header = hp.read_map(self.config.map_path, 
                                          h=True)
            header = dict(header)
            
            if self.config.nside is None:
                self.config.nside = hp.get_nside(map_data)
            
            print(f"   Map loaded: {len(map_data)} pixels")
            print(f"   NSIDE: {self.config.nside}")
            
            map_half1 = None
            map_half2 = None
        
        # Apply preprocessing
        map_data = self._preprocess_map(map_data)
        
        # Load mask
        mask_data = self._load_mask()
        
        # Create combined mask if we have one
        combined_mask = mask_data
        if combined_mask is not None:
            print(f"   Mask coverage: {np.mean(combined_mask)*100:.1f}%")
        
        return {
            'map': map_data,
            'y_map': map_data,  # Alias for backward compatibility
            'mask': mask_data,
            'combined_mask': combined_mask,  # Alias for backward compatibility
            'nside': self.config.nside,
            'nested': self.config.nested,
            'coord_system': self.config.coord_system,
            'header': header,
            'map_half1': map_half1,
            'map_half2': map_half2
        }
    
    def _load_mask(self) -> Optional[np.ndarray]:
        """Load and combine masks"""
        
        if not self.config.mask_path:
            print("   No mask file specified - using full sky")
            return None
        
        print(f"   Loading mask...")
        
        if self.config.mask_columns:
            # Multi-column mask file
            with fits.open(self.config.mask_path) as hdul:
                mask_data = hdul[self.config.mask_hdu].data
                
                print(f"   Available mask columns: {mask_data.dtype.names}")
                
                masks = []
                for col in self.config.mask_columns:
                    if col not in mask_data.dtype.names:
                        print(f"   ⚠️ Warning: Mask column '{col}' not found, skipping")
                        continue
                    print(f"   Loading mask column: {col}")
                    masks.append(mask_data[col] > self.config.mask_threshold)
                
                if not masks:
                    print("   ⚠️ Warning: No valid mask columns found")
                    return None
                
                # Combine masks
                print(f"   Combining {len(masks)} masks with {self.config.mask_combine_method}")
                if self.config.mask_combine_method == "AND":
                    combined_mask = np.all(masks, axis=0)
                elif self.config.mask_combine_method == "OR":
                    combined_mask = np.any(masks, axis=0)
                else:  # SINGLE
                    combined_mask = masks[0]
        else:
            # Simple mask (single array)
            if self.config.map_format == MapFormat.HEALPIX:
                mask_data = hp.read_map(self.config.mask_path)
            else:
                with fits.open(self.config.mask_path) as hdul:
                    mask_data = hdul[self.config.mask_hdu].data
            
            combined_mask = mask_data > self.config.mask_threshold
            print(f"   Single mask loaded")
        
        return combined_mask.astype(bool)
    
    def _preprocess_map(self, map_data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to map"""
        
        # Apply calibration
        if self.config.calibration_factor != 1.0:
            print(f"   Applying calibration factor: {self.config.calibration_factor}")
            map_data = map_data * self.config.calibration_factor
        
        # Remove monopole/dipole if requested
        if self.config.remove_monopole or self.config.remove_dipole:
            if self.config.map_format == MapFormat.HEALPIX:
                if self.config.remove_dipole:
                    print("   Removing dipole...")
                    map_data = hp.remove_dipole(map_data)
                elif self.config.remove_monopole:
                    print("   Removing monopole...")
                    map_data = hp.remove_monopole(map_data)
            else:
                if self.config.remove_monopole:
                    print("   Removing mean...")
                    map_data = map_data - np.nanmean(map_data)

        # NEW: Tanimura ℓ-space high-pass
        if self.config.map_format == MapFormat.HEALPIX and self.config.ell_filter_type == "tanimura":
            print(f"   Applying Tanimura high-pass filter "
                  f"(ℓ ∈ [{self.config.ell_filter_lmin}, {self.config.ell_filter_lmax}])")
            map_data = self._apply_tanimura_filter(map_data)

        return map_data

    def _load_flat_data(self) -> Dict:
        """Load flat 2D array data"""
        raise NotImplementedError("Flat map support will be implemented when needed")
    
    def _load_fits_image_data(self) -> Dict:
        """Load FITS image data"""
        raise NotImplementedError("FITS image support will be implemented when needed")
