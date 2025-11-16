# stacking_backend/config/paths.py
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataPaths:
    """Configurable data paths for non-map data (catalogs, etc.)"""
    
    # Catalog paths only - maps/masks handled by MapConfig
    mcxc_catalog: Optional[str] = "/cosma7/data/dp004/rttw52/Manticore/observational_data/mcxc_2_clusters/mcxc_clusters.hdf5"
    erosita_catalog: Optional[str] = "/cosma7/data/dp004/rttw52/Manticore/observational_data/erosita_clusters/erosita_clusters.hdf5"
    manticore_catalog: Optional[str] = "/cosma7/data/dp004/rttw52/Manticore/new_analysis/clusters/temp/output/simplified_clusters.h5"

    # CMB temperature maps for kSZ
    planck_cmb_map: Optional[str] = None  # Path to COM_CMB_IQU-smica_2048_R3.00_full.fits
    planck_cmb_mask: Optional[str] = None  # Optional CMB mask
    remove_cmb_dipole: bool = False  # Whether to remove residual dipole

    # Add any other non-map related paths here
    output_dir: Optional[str] = None
    temp_dir: Optional[str] = None

    def __post_init__(self):
        # Override with environment variables if available
        self.mcxc_catalog = os.getenv('MCXC_CATALOG_PATH', self.mcxc_catalog)
        self.erosita_catalog = os.getenv('EROSITA_CATALOG_PATH', self.erosita_catalog)
        self.manticore_catalog = os.getenv('MANTICORE_CATALOG_PATH', self.manticore_catalog)
        self.output_dir = os.getenv('OUTPUT_DIR', self.output_dir or './outputs')
        self.temp_dir = os.getenv('TEMP_DIR', self.temp_dir or './temp')

    def validate_paths(self) -> dict:
        """Validate that catalog paths exist if provided"""
        validation_results = {}
        
        for field_name in ['mcxc_catalog', 'erosita_catalog', 'manticore_catalog']:
            path = getattr(self, field_name)
            if path is not None:
                path_obj = Path(path)
                validation_results[field_name] = {
                    'path': path,
                    'exists': path_obj.exists(),
                    'readable': path_obj.exists() and os.access(path, os.R_OK)
                }
        
        # Check directories
        for field_name in ['output_dir', 'temp_dir']:
            path = getattr(self, field_name)
            if path is not None:
                path_obj = Path(path)
                validation_results[field_name] = {
                    'path': path,
                    'exists': path_obj.exists(),
                    'is_dir': path_obj.is_dir() if path_obj.exists() else False
                }
        
        return validation_results
    
    @classmethod
    def get_default(cls):
        """Get default data paths (mainly for directories)"""
        return cls()
