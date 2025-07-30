import numpy as np
import healpy as hp
from astropy.io import fits

def load_pr4_data():
    """Load the PR4 NILC y-map and masks"""

    print("🔍 LOADING PR4 NILC Y-MAP AND MASKS")
    print("="*50)

    # Load y-map
    with fits.open("Chandran_compy/PR4_NILC_y_map.fits") as hdul:
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
    with fits.open("Chandran_compy/Masks.fits") as hdul:
        mask_data = hdul[1].data

        print(f"Mask columns: {mask_data.dtype.names}")

        nilc_mask = mask_data['NILC-MASK']
        gal_mask = mask_data['GAL-MASK']
        ps_mask = mask_data['PS-MASK']

    return {
        'y_map': y_map,
        'y_half1': y_half1,
        'y_half2': y_half2,
        'nilc_mask': nilc_mask,
        'gal_mask': gal_mask,
        'ps_mask': ps_mask,
        'nside': y_header['NSIDE'],
        'combined_mask': (nilc_mask > 0.5) & (gal_mask > 0.5) & (ps_mask > 0.5)
    }
