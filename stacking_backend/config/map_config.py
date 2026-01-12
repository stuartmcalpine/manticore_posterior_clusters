# stacking_backend/config/map_config.py
from dataclasses import dataclass
from typing import Optional, Union, List, Dict
from enum import Enum

class MapFormat(Enum):
    HEALPIX = "healpix"
    FLAT = "flat"
    FITS_IMAGE = "fits_image"

@dataclass
class MapConfig:
    """Configuration for generic map input"""
    
    # Map file paths
    map_path: str
    mask_path: Optional[str] = None
    
    # Map format and structure
    map_format: MapFormat = MapFormat.HEALPIX
    map_hdu: int = 1  # HDU index in FITS file
    mask_hdu: int = 1
    
    # Column/field names (for multi-column FITS)
    map_column: Optional[str] = None
    mask_columns: Optional[List[str]] = None
    
    # HEALPix specific
    nside: Optional[int] = None
    nested: bool = False
    coord_system: str = "G"
    
    # Mask combination strategy
    mask_combine_method: str = "AND"
    mask_threshold: float = 0.5
    
    # Data preprocessing
    remove_monopole: bool = False
    remove_dipole: bool = False
    calibration_factor: float = 1.0

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.map_format not in MapFormat:
            self.map_format = MapFormat(self.map_format)
        
        if self.mask_combine_method not in ["AND", "OR", "SINGLE"]:
            raise ValueError(f"mask_combine_method must be AND, OR, or SINGLE, got {self.mask_combine_method}")
        
        if not 0 <= self.mask_threshold <= 1:
            raise ValueError(f"mask_threshold must be between 0 and 1, got {self.mask_threshold}")
        
        if self.coord_system not in ["G", "C"]:
            raise ValueError(f"coord_system must be 'G' (Galactic) or 'C' (Celestial), got {self.coord_system}")

    @classmethod
    def for_planck_pr4(cls, y_map_path: str, masks_path: str):
        """Preset for Planck PR4 data"""
        return cls(
            map_path=y_map_path,
            mask_path=masks_path,
            map_format=MapFormat.HEALPIX,
            map_column="FULL",
            mask_columns=["NILC-MASK", "GAL-MASK", "PS-MASK"],
            mask_combine_method="AND",
            nside=2048,
            coord_system="G"
        )

    @classmethod
    def for_planck_comcomp(cls, y_map_path: str, mask_path: str = None):
        """Preset for official Planck COM_CompMap Compton-y (MILCA/NILC) maps."""
        return cls(
            map_path=y_map_path,
            mask_path=mask_path,
            map_format=MapFormat.HEALPIX,
            map_column=None,
            mask_columns=None,
            mask_combine_method="AND",
            nside=2048,
            coord_system="G"
        )

    @classmethod
    def for_simple_healpix(cls, map_path: str, mask_path: Optional[str] = None):
        """Preset for simple HEALPix map (single array)"""
        return cls(
            map_path=map_path,
            mask_path=mask_path,
            map_format=MapFormat.HEALPIX,
            map_column=None,
            mask_columns=None,
            coord_system="G"
        )
    
    @classmethod
    def for_act_or_spt(cls, map_path: str, mask_path: Optional[str] = None,
                       coord_system: str = "C", calibration: float = 1.0):
        """Preset for ACT or SPT maps (typically in Celestial coordinates)"""
        return cls(
            map_path=map_path,
            mask_path=mask_path,
            map_format=MapFormat.HEALPIX,
            map_column=None,
            mask_columns=None,
            coord_system=coord_system,
            calibration_factor=calibration,
            remove_monopole=True
        )

    @classmethod
    def for_planck_cmb_smica(cls, cmb_map_path: str):
        """Preset for Planck CMB temperature maps (SMICA/NILC) for kSZ analysis"""
        return cls(
            map_path=cmb_map_path,
            mask_path=cmb_map_path,
            map_format=MapFormat.HEALPIX,
            map_column="I_STOKES",
            mask_columns=["TMASK"],
            mask_combine_method="SINGLE",
            nside=2048,
            coord_system="G",
            calibration_factor=1e6,
            remove_monopole=False,
            remove_dipole=False,
            nested=True
        )

    @classmethod
    def for_prefiltered_planck(cls, filtered_map_path: str, mask_path: Optional[str] = None,
                              nested: bool = True):
        """
        Preset for pre-filtered Planck maps (e.g., output from preprocess_planck_map.py).
        
        Use this when you've already applied Tanimura filtering, dipole removal, and
        calibration using the standalone preprocessing script.
        
        Parameters
        ----------
        filtered_map_path : str
            Path to pre-filtered HEALPix map
        mask_path : str, optional
            Path to mask file
        nested : bool
            Whether map is in NESTED ordering (default: True)
        """
        return cls(
            map_path=filtered_map_path,
            mask_path=mask_path,
            map_format=MapFormat.HEALPIX,
            map_column=None,  # Pre-filtered maps are simple arrays
            mask_columns=["TMASK"] if mask_path else None,
            mask_combine_method="SINGLE",
            nside=2048,
            coord_system="G",
            calibration_factor=1,
            remove_monopole=False, 
            remove_dipole=False,  # Already removed 
            nested=nested
        )
