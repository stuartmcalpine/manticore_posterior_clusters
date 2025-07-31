# stacking_backend/data/pr4_loader.py
import numpy as np
import healpy as hp
from astropy.io import fits
from pathlib import Path
import threading
from ..config.paths import DataPaths

class PR4DataLoader:
    """Thread-safe PR4 data loader with configurable paths"""
    
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
        if self._initialized:
            return
        
        self.data_paths = data_paths or DataPaths.get_default()
        self._initialized = True
    
    def load_pr4_data(self, validate_paths=True, use_cache=True):
        """Load the PR4 NILC y-map and masks with error handling"""
        
        if use_cache and 'pr4_data' in self._data_cache:
            print("📋 Using cached PR4 data")
            return self._data_cache['pr4_data']
        
        print("🔍 LOADING PR4 NILC Y-MAP AND MASKS")
        print("="*50)
        
        if validate_paths:
            validation = self.data_paths.validate_paths()
            if not validation['pr4_y_map']['exists']:
                raise FileNotFoundError(f"Y-map file not found: {self.data_paths.pr4_y_map}")
            if not validation['pr4_masks']['exists']:
                raise FileNotFoundError(f"Masks file not found: {self.data_paths.pr4_masks}")
        
        try:
            # Load y-map
            with fits.open(self.data_paths.pr4_y_map) as hdul:
                y_data = hdul[1].data
                y_header = hdul[1].header

                print(f"Y-map columns: {y_data.dtype.names}")
                print(f"NSIDE: {y_header['NSIDE']}")
                print(f"Coordinate system: {y_header['COORDSYS']}")
                print(f"Ordering: {y_header['ORDERING']}")

                # Use the FULL mission y-map
                y_map = y_data['FULL']
                y_half1 = y_data['HALF-RING 1']
                y_half2 = y_data['HALF-RING 2']

            # Load masks
            with fits.open(self.data_paths.pr4_masks) as hdul:
                mask_data = hdul[1].data

                print(f"Mask columns: {mask_data.dtype.names}")

                nilc_mask = mask_data['NILC-MASK']
                gal_mask = mask_data['GAL-MASK']
                ps_mask = mask_data['PS-MASK']

            data = {
                'y_map': y_map,
                'y_half1': y_half1,
                'y_half2': y_half2,
                'nilc_mask': nilc_mask,
                'gal_mask': gal_mask,
                'ps_mask': ps_mask,
                'nside': y_header['NSIDE'],
                'combined_mask': (nilc_mask > 0.5) & (gal_mask > 0.5) & (ps_mask > 0.5)
            }
            
            if use_cache:
                self._data_cache['pr4_data'] = data
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PR4 data: {str(e)}") from e

def load_pr4_data(data_paths=None, validate_paths=True, use_cache=True):
    """Convenience function to load PR4 data"""
    loader = PR4DataLoader(data_paths)
    return loader.load_pr4_data(validate_paths, use_cache)
