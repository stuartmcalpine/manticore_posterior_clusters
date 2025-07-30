from dataclasses import dataclass
from typing import Tuple

@dataclass
class AnalysisParameters:
    """Default parameters for cluster analysis"""
    
    # Patch extraction parameters
    patch_size_deg: float = 15.0
    npix: int = 256
    
    # Aperture photometry parameters
    inner_r200_factor: float = 1.0
    outer_r200_factor: float = 3.0
    min_coverage: float = 0.9
    
    # Stacking parameters
    max_patches: int = None
    
    # Profile calculation parameters
    n_radial_bins: int = 20
    max_radius_deg: float = None
    
    # Quality control parameters
    min_inner_pixels: int = 10
    min_outer_pixels: int = 50
    
    # Background subtraction parameters
    bg_inner_radius_deg: float = 5.0
    bg_outer_radius_deg: float = 7.0
    min_bg_pixels: int = 100
    
    @classmethod
    def get_default(cls):
        """Get default analysis parameters"""
        return cls()
    
    @classmethod
    def for_mass_scaling(cls):
        """Get parameters optimized for mass scaling analysis"""
        return cls(
            patch_size_deg=20.0,
            npix=256,
            inner_r200_factor=1.0,
            outer_r200_factor=3.0,
            min_coverage=0.8,
            n_radial_bins=30
        )
