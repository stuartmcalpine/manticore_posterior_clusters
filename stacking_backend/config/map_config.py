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

    # ℓ-space Tanimura-style high-pass filter
    ell_filter_type: Optional[str] = None  # None or "tanimura"
    ell_filter_lmin: int = 360             # start of ramp (ℓ₁)
    ell_filter_lmax: int = 720             # end of ramp (ℓ₂)

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
            map_format=MapFormat.HEALPIX,   # Same as PR4 – HEALPix all-sky map
            map_column=None,                # IMPORTANT: Compton-y maps do NOT have columns
            mask_columns=None,              # Mask typically single-map; no columns
            mask_combine_method="AND",      # If you choose to combine multiple masks later
            nside=2048,                     # All official y-maps are NSIDE=2048
            coord_system="G"                # MILCA/NILC maps are in Galactic coordinates
        )

    @classmethod
    def for_simple_healpix(cls, map_path: str, mask_path: Optional[str] = None):
        """Preset for simple HEALPix map (single array)"""
        return cls(
            map_path=map_path,
            mask_path=mask_path,
            map_format=MapFormat.HEALPIX,
            map_column=None,  # Use primary data array
            mask_columns=None,  # Use primary mask array
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
            mask_path=cmb_map_path,          # same file for map + TMASK
            map_format=MapFormat.HEALPIX,
            map_column="I_STOKES",           # temperature column
            mask_columns=["TMASK"],          # use TMASK from same table
            mask_combine_method="SINGLE",
            nside=2048,
            coord_system="G",
            calibration_factor=1e6,          # K -> µK
            remove_monopole=False,
            remove_dipole=False,
            nested=True,
            apply_ell_filter=False,
            ell_filter_lmin=360,        # ≈ 30 arcmin
            ell_filter_lmax=720,        # ≈ 15 arcmin
            lmax=None,                   # default to 3*nside-1
        )

    @classmethod
    def for_planck_217_tanimura(
        cls,
        map_path: str,
        mask_path: Optional[str] = None,
        ell_lmin: int = 360,
        ell_lmax: int = 720,
        nested: bool = True,
        ell_filter_type = "tanimura"
    ):
        """
        Preset for Planck 217 GHz map processed with a Tanimura-style
        ℓ-space high-pass filter for kSZ analysis.
    
        - Uses I_STOKES as the temperature map.
        - Assumes NSIDE=2048, Galactic coordinates.
        - Converts K_CMB → µK_CMB via calibration_factor.
        - Applies a cosine high-pass window W_ell:
            W_ell = 0 for ell < ell_lmin
            W_ell = 1 for ell > ell_lmax
            smooth ramp between ell_lmin and ell_lmax.
        """
        return cls(
            map_path=map_path,
            mask_path=mask_path,
            map_format=MapFormat.HEALPIX,
            map_hdu=1,                 # FREQ-MAP HDU
            map_column="I_STOKES",     # main temperature map
            mask_columns=["TMASK"],
            mask_combine_method="SINGLE",
            nside=2048,
            nested=nested,             # True for HFI_SkyMap_217_2048_R3.01_full.fits
            coord_system="G",          # GALACTIC in the header
            calibration_factor=1e6,    # K → µK
            remove_monopole=False,
            remove_dipole=False,       # you can set True if you want dipole removed
            ell_filter_type=ell_filter_type,
            ell_filter_lmin=ell_lmin,
            ell_filter_lmax=ell_lmax,
        )

