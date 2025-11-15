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
    
    def _validate_config(self):
        """Validate that files exist"""
        if not Path(self.config.map_path).exists():
            raise FileNotFoundError(f"Map file not found: {self.config.map_path}")
        
        if self.config.mask_path and not Path(self.config.mask_path).exists():
            raise FileNotFoundError(f"Mask file not found: {self.config.mask_path}")
    
    def load_data(self, use_cache: bool = True) -> Dict:
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
                                          h=True, verbose=False)
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
                mask_data = hp.read_map(self.config.mask_path, verbose=False)
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
                # Use healpy for monopole/dipole removal
                if self.config.remove_dipole:
                    print("   Removing dipole...")
                    map_data = hp.remove_dipole(map_data, verbose=False)
                elif self.config.remove_monopole:
                    print("   Removing monopole...")
                    map_data = hp.remove_monopole(map_data, verbose=False)
            else:
                # Simple mean subtraction for non-HEALPix
                if self.config.remove_monopole:
                    print("   Removing mean...")
                    map_data = map_data - np.nanmean(map_data)
        
        return map_data
    
    def _load_flat_data(self) -> Dict:
        """Load flat 2D array data"""
        raise NotImplementedError("Flat map support will be implemented when needed")
    
    def _load_fits_image_data(self) -> Dict:
        """Load FITS image data"""
        raise NotImplementedError("FITS image support will be implemented when needed")

# Backward compatibility function
def load_pr4_data(data_paths=None, validate_paths=True, use_cache=True):
    """
    Backward compatibility wrapper for loading Planck PR4 data
    
    This function maintains the original API while using the new generic loader
    """
    from ..config.paths import DataPaths
    from ..config.map_config import MapConfig
    
    if data_paths is None:
        data_paths = DataPaths.get_default()
    
    # Create MapConfig for PR4 data
    config = MapConfig.for_planck_pr4(data_paths.pr4_y_map, data_paths.pr4_masks)
    
    # Load using generic loader
    loader = GenericMapLoader(config)
    data = loader.load_data(use_cache=use_cache)
    
    # Return in the expected format for backward compatibility
    return {
        'y_map': data['map'],
        'y_half1': data.get('map_half1'),
        'y_half2': data.get('map_half2'),
        'nilc_mask': None,  # Individual masks not loaded separately anymore
        'gal_mask': None,
        'ps_mask': None,
        'nside': data['nside'],
        'combined_mask': data['mask']
    }
