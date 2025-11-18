#!/usr/bin/env python
import numpy as np
import healpy as hp
from astropy.io import fits
from pathlib import Path

# Optional: use CAMB to generate a theoretical CMB Cl if available
try:
    import camb
    HAS_CAMB = True
except ImportError:
    HAS_CAMB = False
    print("⚠️  CAMB not found. This script can still generate the noise Cl.")
    print("    For CMB Cl, either install CAMB or point to an external Cl file.")

# ----------------------------------------------------------------------
# User configuration
# ----------------------------------------------------------------------

# Full 217 GHz map (the one you showed)
MAP_PATH = Path(
    "/cosma7/data/dp004/rttw52/Manticore/new_analysis/clusters/"
    "posterior_associations/notebooks/Planck217GHz/"
    "HFI_SkyMap_217_2048_R3.01_full.fits"
)

# Output files
OUT_CMB_CL = Path("cl_cmb_ell.txt")
OUT_NOISE_CL = Path("cl_noise_ell_217.txt")

# Maximum multipole to compute
LMAX = 4096  # safe for NSIDE=2048

# If you don't have CAMB, you can point to an existing CMB Cl file instead:
# This file should have two columns: ell, Cl_TT (in µK^2).
EXTERNAL_CMB_CL = None  # e.g. Path("planck2018_ttt_cl.txt")

# ----------------------------------------------------------------------
# 1. Compute / load CMB Cl
# ----------------------------------------------------------------------

def get_cmb_cl(lmax: int) -> np.ndarray:
    """
    Return CMB TT power spectrum C_l[0..lmax] in µK^2.

    If EXTERNAL_CMB_CL is set, read it (2 columns: ell, C_l).
    Otherwise use CAMB to generate a Planck18-like ΛCDM TT spectrum.
    """
    if EXTERNAL_CMB_CL is not None:
        print(f"📥 Loading CMB Cl from {EXTERNAL_CMB_CL}")
        arr = np.loadtxt(EXTERNAL_CMB_CL)
        ell_in = arr[:, 0].astype(int)
        cl_in = arr[:, 1]

        cl = np.zeros(lmax + 1)
        valid = ell_in <= lmax
        cl[ell_in[valid]] = cl_in[valid]
        return cl

    if not HAS_CAMB:
        raise RuntimeError(
            "No CAMB installed and EXTERNAL_CMB_CL is None.\n"
            "Please either install CAMB (pip install camb) or provide a Cl file."
        )

    print("🧠 Generating CMB Cl with CAMB (Planck18-like cosmology)...")

    import camb
    pars = camb.CAMBparams()
    # Planck 2018-ish parameters
    pars.set_cosmology(
        H0=67.32,
        ombh2=0.022383,
        omch2=0.12011,
        TCMB=2.7255,
    )
    pars.InitPower.set_params(
        ns=0.96605,
        As=2.1e-9
    )
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    results = camb.get_results(pars)

    # IMPORTANT:
    # By default, get_cmb_power_spectra returns D_l = l(l+1)C_l/(2π)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=False)
    tot = powers["total"]  # shape (lmax+1, 4): [TT, EE, BB, TE]
    dl_tt = tot[: lmax + 1, 0]  # D_l^TT in µK^2

    ell = np.arange(lmax + 1, dtype=float)
    # Convert D_l -> C_l: C_l = D_l * 2π / (l(l+1))
    cl_tt = np.zeros_like(ell)
    mask = ell > 0
    cl_tt[mask] = dl_tt[mask] * 2.0 * np.pi / (ell[mask] * (ell[mask] + 1.0))
    cl_tt[0] = 0.0  # l=0 undefined, set to 0

    print(f"   Generated CMB C_l up to ell={lmax}")
    return cl_tt

# ----------------------------------------------------------------------
# 2. Compute noise Cl from the 217 GHz full map (using II_COV)
# ----------------------------------------------------------------------

def get_noise_cl_from_iicov(fits_path: Path, lmax: int) -> np.ndarray:
    """
    Approximate the noise power spectrum N_ell as white noise using the II_COV
    per-pixel variance column in the 217 GHz full map.

    Steps:
      - Read II_COV (per-pixel variance) from the FREQ-MAP HDU.
      - Convert K^2 -> µK^2 to match CMB Cl (which are in µK^2).
      - Compute mean variance <sigma^2> over finite pixels.
      - White-noise level: N_ell = <sigma^2> * 4π / Npix  (independent of ell).

    Parameters
    ----------
    fits_path : Path
        Path to the Planck 217 GHz full map FITS file
        (e.g. HFI_SkyMap_217_2048_R3.01_full.fits).
    lmax : int
        Maximum multipole to compute.

    Returns
    -------
    nl : np.ndarray
        Noise power spectrum N_ell of length lmax+1 in µK^2.
    """
    fits_path = Path(fits_path)
    if not fits_path.is_file():
        raise FileNotFoundError(f"Map file not found: {fits_path}")

    print(f"📥 Reading map from {fits_path}")
    with fits.open(fits_path) as hdul:
        hdu = hdul[1]  # FREQ-MAP
        data = hdu.data
        header = hdu.header

        nside = header["NSIDE"]
        ordering = header.get("ORDERING", "NESTED")
        tunit = header.get("TUNIT1", "Kcmb")

        print(f"   NSIDE: {nside}, ORDERING: {ordering}, TUNIT1: {tunit}")
        if "II_COV" not in data.names:
            raise RuntimeError("II_COV column not found in FREQ-MAP HDU")

        # II_COV is the variance per pixel, in K^2 (since TUNIT1 is Kcmb)
        ii_cov_K2 = np.array(data["II_COV"], dtype=float)

    finite = np.isfinite(ii_cov_K2)
    sigma2_mean_K2 = np.mean(ii_cov_K2[finite])

    # Convert K^2 -> µK^2  (1 K = 1e6 µK)
    sigma2_mean_uK2 = sigma2_mean_K2 * (1e6 ** 2)

    npix = hp.nside2npix(nside)
    # White-noise Cl level N0 in µK^2
    n0_uK2 = sigma2_mean_uK2 * 4.0 * np.pi / npix

    print(f"   <sigma^2> = {sigma2_mean_uK2:.3e} µK^2, "
          f"N0 (white) = {n0_uK2:.3e} µK^2")

    nl = np.full(lmax + 1, n0_uK2, dtype=float)
    return nl


def main():
    if not MAP_PATH.is_file():
        raise FileNotFoundError(f"Map file not found: {MAP_PATH}")

    # 1) CMB Cl
    cl_cmb = get_cmb_cl(LMAX)
    ell = np.arange(LMAX + 1)
    np.savetxt(
        OUT_CMB_CL,
        np.column_stack([ell, cl_cmb]),
        header="ell   Cl_TT   (µK^2)",
    )
    print(f"✅ Saved CMB Cl to {OUT_CMB_CL.resolve()}")

    # 2) Noise Cl
    cl_noise = get_noise_cl_from_iicov(MAP_PATH, LMAX)
    np.savetxt(
        OUT_NOISE_CL,
        np.column_stack([ell, cl_noise]),
        header="ell   N_ell   (same units as map variance)",
    )
    print(f"✅ Saved noise Cl to {OUT_NOISE_CL.resolve()}")


if __name__ == "__main__":
    main()

