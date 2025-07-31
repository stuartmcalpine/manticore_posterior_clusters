# stacking_backend/config/analysis_params.py
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class AnalysisParameters:
    """Centralized configuration for cluster analysis"""
    
    # Patch extraction parameters
    patch_size_deg: float = 15.0
    npix: int = 256
    
    # Aperture photometry parameters
    inner_r200_factor: float = 1.0
    outer_r200_factor: float = 3.0
    min_coverage: float = 0.9
    
    # Stacking parameters
    max_patches: Optional[int] = None
    
    # Profile calculation parameters
    n_radial_bins: int = 20
    max_radius_deg: Optional[float] = None
    
    # Quality control parameters
    min_inner_pixels: int = 10
    min_outer_pixels: int = 50
    
    # Background subtraction parameters
    bg_inner_radius_deg: float = 5.0
    bg_outer_radius_deg: float = 7.0
    min_bg_pixels: int = 100
    
    # Bootstrap parameters
    n_bootstrap: int = 500
    bootstrap_confidence_level: float = 0.68
    
    # Null test parameters
    n_random_pointings: int = 500
    exclusion_radius_factor: float = 3.0
    
    # Coordinate validation parameters
    max_reasonable_r200_deg: float = 10.0
    max_reasonable_redshift: float = 3.0
    min_galactic_latitude: float = 10.0  # Avoid galactic plane
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        self.validate()
    
    def validate(self) -> None:
        """Validate all parameters"""
        
        # Patch parameters
        if self.patch_size_deg <= 0 or self.patch_size_deg > 90:
            raise ValueError(f"patch_size_deg must be in (0, 90], got {self.patch_size_deg}")
        
        if self.npix <= 0:
            raise ValueError(f"npix must be positive, got {self.npix}")
        
        # Aperture parameters
        if self.inner_r200_factor <= 0:
            raise ValueError(f"inner_r200_factor must be positive, got {self.inner_r200_factor}")
        
        if self.outer_r200_factor <= self.inner_r200_factor:
            raise ValueError(f"outer_r200_factor must be > inner_r200_factor")
        
        if not 0 < self.min_coverage <= 1:
            raise ValueError(f"min_coverage must be in (0, 1], got {self.min_coverage}")
        
        # Bootstrap parameters
        if self.n_bootstrap < 10:
            raise ValueError(f"n_bootstrap must be >= 10, got {self.n_bootstrap}")
        
        if not 0 < self.bootstrap_confidence_level < 1:
            raise ValueError(f"bootstrap_confidence_level must be in (0, 1), got {self.bootstrap_confidence_level}")
    
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
            n_radial_bins=30,
            n_bootstrap=1000,
            n_random_pointings=1000
        )
    
    @classmethod
    def for_quick_test(cls):
        """Get parameters for quick testing"""
        return cls(
            patch_size_deg=10.0,
            npix=128,
            n_bootstrap=100,
            n_random_pointings=100
        )
