import numpy as np

class InputValidator:
    """Input validation for analysis parameters and data"""
    
    @staticmethod
    def validate_coord_list(coord_list):
        """Validate coordinate list format"""
        if not coord_list:
            raise ValueError("Coordinate list is empty")
        
        for i, coords in enumerate(coord_list):
            if len(coords) < 2:
                raise ValueError(f"Coordinate {i} must have at least 2 elements (lon, lat)")
            
            if len(coords) >= 4:  # Has redshift
                if coords[3] < 0:
                    raise ValueError(f"Coordinate {i} has negative redshift: {coords[3]}")
            
            if len(coords) >= 3:  # Has R200
                if coords[2] <= 0:
                    raise ValueError(f"Coordinate {i} has non-positive R200: {coords[2]}")
    
    @staticmethod
    def validate_analysis_params(patch_size_deg, npix, inner_r200_factor, outer_r200_factor, min_coverage):
        """Validate analysis parameters"""
        if patch_size_deg <= 0:
            raise ValueError(f"patch_size_deg must be positive, got {patch_size_deg}")
        
        if npix <= 0:
            raise ValueError(f"npix must be positive, got {npix}")
        
        if inner_r200_factor <= 0:
            raise ValueError(f"inner_r200_factor must be positive, got {inner_r200_factor}")
        
        if outer_r200_factor <= inner_r200_factor:
            raise ValueError(f"outer_r200_factor ({outer_r200_factor}) must be greater than inner_r200_factor ({inner_r200_factor})")
        
        if not 0 < min_coverage <= 1:
            raise ValueError(f"min_coverage must be between 0 and 1, got {min_coverage}")
    
    @staticmethod
    def validate_results_dict(results_dict):
        """Validate results dictionary for plotting"""
        if not results_dict:
            raise ValueError("Results dictionary is empty")
        
        for label, results in results_dict.items():
            if not isinstance(results, dict):
                raise ValueError(f"Results for {label} must be a dictionary")
            
            if not results.get('success', False):
                print(f"Warning: Results for {label} indicate failure")
