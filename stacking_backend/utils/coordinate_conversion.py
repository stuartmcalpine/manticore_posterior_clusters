# stacking_backend/utils/coordinate_conversion.py
import numpy as np
from astropy.cosmology import Planck18

def mpc_to_angular_degrees(radius_mpc, redshift):
    """
    Convert physical radius in Mpc to angular radius in degrees
    
    Parameters
    ----------
    radius_mpc : float or array_like
        Physical radius in Mpc
    redshift : float or array_like
        Redshift of the object
        
    Returns
    -------
    radius_deg : float or array_like
        Angular radius in degrees
    """
    # Get angular diameter distance in Mpc
    D_A = Planck18.angular_diameter_distance(redshift).value
    
    # Convert to angular size in radians, then degrees
    radius_rad = radius_mpc / D_A
    radius_deg = np.degrees(radius_rad)
    
    return radius_deg
