# stacking_backend/config/paths.py
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataPaths:
    """Configurable data paths"""
    
    # Default paths - can be overridden by environment variables
    pr4_y_map: str = "/cosma7/data/dp004/rttw52/Manticore/new_analysis/clusters/posterior_associations/notebooks/Chandran_compy/PR4_NILC_y_map.fits"
    pr4_masks: str = "/cosma7/data/dp004/rttw52/Manticore/new_analysis/clusters/posterior_associations/notebooks/Chandran_compy/Masks.fits"
    mcxc_catalog: str = "/cosma7/data/dp004/rttw52/Manticore/observational_data/mcxc_2_clusters/mcxc_clusters.hdf5"
    erosita_catalog: str = "/cosma7/data/dp004/rttw52/Manticore/observational_data/erosita_clusters/erosita_clusters.hdf5"
    manticore_catalog: str = "/cosma7/data/dp004/rttw52/Manticore/new_analysis/clusters/temp/output/simplified_clusters.h5"

    def __post_init__(self):
        # Override with environment variables if available
        self.pr4_y_map = os.getenv('PR4_Y_MAP_PATH', self.pr4_y_map)
        self.pr4_masks = os.getenv('PR4_MASKS_PATH', self.pr4_masks)
        self.mcxc_catalog = os.getenv('MCXC_CATALOG_PATH', self.mcxc_catalog)
        self.erosita_catalog = os.getenv('EROSITA_CATALOG_PATH', self.erosita_catalog)
        self.manticore_catalog = os.getenv('MANTICORE_CATALOG_PATH', self.manticore_catalog)
        self.planck_y_map = "/cosma7/data/dp004/rttw52/Manticore/new_analysis/clusters/posterior_associations/notebooks/COM_CompMap/COM_CompMap_YSZ_R2.02/milca_ymaps.fits"
        self.planck_y_mask = "/cosma7/data/dp004/rttw52/Manticore/new_analysis/clusters/posterior_associations/notebooks/COM_CompMap/COM_CompMap_YSZ_R2.02/COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits"

    def validate_paths(self) -> dict:
        """Validate that required paths exist"""
        validation_results = {}
        
        for field_name, path in self.__dict__.items():
            path_obj = Path(path)
            validation_results[field_name] = {
                'path': path,
                'exists': path_obj.exists(),
                'readable': path_obj.exists() and os.access(path, os.R_OK)
            }
        
        return validation_results
    
    @classmethod
    def get_default(cls):
        """Get default data paths"""
        return cls()
