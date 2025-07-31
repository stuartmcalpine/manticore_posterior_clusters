# stacking_backend/utils/validation.py
import numpy as np
from typing import List, Tuple, Any

class InputValidator:
    """Input validation for analysis parameters and data"""
    
    @staticmethod
    def validate_coord_list(coord_list: List[Tuple]) -> None:
        """Validate coordinate list format with detailed error messages"""
        if coord_list.size == 0:
            raise ValueError("Coordinate list is empty")

        if not isinstance(coord_list, (list, tuple, np.ndarray)):
            raise TypeError(f"coord_list must be list, tuple, or array, got {type(coord_list)}")
        
        for i, coords in enumerate(coord_list):
            try:
                coords = np.array(coords, dtype=float)
            except (ValueError, TypeError):
                raise ValueError(f"Coordinate {i} contains non-numeric values: {coords}")
            
            if len(coords) < 2:
                raise ValueError(f"Coordinate {i} must have at least 2 elements (lon, lat), got {len(coords)}")
            
            # Validate longitude range
            if not (0 <= coords[0] <= 360):
                raise ValueError(f"Coordinate {i} longitude out of range [0, 360]: {coords[0]}")
            
            # Validate latitude range
            if not (-90 <= coords[1] <= 90):
                raise ValueError(f"Coordinate {i} latitude out of range [-90, 90]: {coords[1]}")
            
            if len(coords) >= 3:  # Has R200
                if coords[2] <= 0:
                    raise ValueError(f"Coordinate {i} has non-positive R200: {coords[2]}")
                if coords[2] > 10:  # Sanity check - R200 > 10 degrees is very large
                    print(f"Warning: Coordinate {i} has very large R200: {coords[2]}°")
            
            if len(coords) >= 4:  # Has redshift
                if coords[3] < 0:
                    raise ValueError(f"Coordinate {i} has negative redshift: {coords[3]}")
                if coords[3] > 3:  # Sanity check
                    print(f"Warning: Coordinate {i} has very high redshift: {coords[3]}")
    
    @staticmethod
    def validate_analysis_params(patch_size_deg: float, npix: int, inner_r200_factor: float, 
                               outer_r200_factor: float, min_coverage: float) -> None:
        """Validate analysis parameters with detailed checks"""
        
        # Patch size validation
        if not isinstance(patch_size_deg, (int, float)):
            raise TypeError(f"patch_size_deg must be numeric, got {type(patch_size_deg)}")
        if patch_size_deg <= 0:
            raise ValueError(f"patch_size_deg must be positive, got {patch_size_deg}")
        if patch_size_deg > 90:
            raise ValueError(f"patch_size_deg too large (>{90}°), got {patch_size_deg}")
        
        # Pixel count validation
        if not isinstance(npix, int):
            raise TypeError(f"npix must be integer, got {type(npix)}")
        if npix <= 0:
            raise ValueError(f"npix must be positive, got {npix}")
        if npix > 2048:
            print(f"Warning: npix is very large ({npix}), may cause memory issues")
        
        # R200 factor validation
        if not isinstance(inner_r200_factor, (int, float)):
            raise TypeError(f"inner_r200_factor must be numeric, got {type(inner_r200_factor)}")
        if inner_r200_factor <= 0:
            raise ValueError(f"inner_r200_factor must be positive, got {inner_r200_factor}")
        
        if not isinstance(outer_r200_factor, (int, float)):
            raise TypeError(f"outer_r200_factor must be numeric, got {type(outer_r200_factor)}")
        if outer_r200_factor <= inner_r200_factor:
            raise ValueError(f"outer_r200_factor ({outer_r200_factor}) must be greater than inner_r200_factor ({inner_r200_factor})")
        
        # Coverage validation
        if not isinstance(min_coverage, (int, float)):
            raise TypeError(f"min_coverage must be numeric, got {type(min_coverage)}")
        if not 0 < min_coverage <= 1:
            raise ValueError(f"min_coverage must be between 0 and 1, got {min_coverage}")
    
    @staticmethod
    def validate_results_dict(results_dict: dict) -> None:
        """Validate results dictionary for plotting"""
        if not results_dict:
            raise ValueError("Results dictionary is empty")
        
        if not isinstance(results_dict, dict):
            raise TypeError(f"results_dict must be dictionary, got {type(results_dict)}")
        
        for label, results in results_dict.items():
            if not isinstance(results, dict):
                raise ValueError(f"Results for {label} must be a dictionary, got {type(results)}")
            
            if not results.get('success', False):
                print(f"Warning: Results for {label} indicate failure")
            
            # Check for required keys
            required_keys = ['mean_delta_y', 'error_mean', 'significance', 'n_measurements']
            missing_keys = [key for key in required_keys if key not in results]
            if missing_keys:
                raise ValueError(f"Results for {label} missing required keys: {missing_keys}")
    
    @staticmethod
    def validate_file_path(file_path: str, must_exist: bool = True) -> None:
        """Validate file path"""
        from pathlib import Path
        
        if not isinstance(file_path, (str, Path)):
            raise TypeError(f"file_path must be string or Path, got {type(file_path)}")
        
        path_obj = Path(file_path)
        
        if must_exist and not path_obj.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        if must_exist and not path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
