import numpy as np

__all__ = [
    '_safe_covariance',
    '_sym_eig_sqrt_inv', 
    '_matrix_sqrt',
    '_matrix_invsqrt',
    '_mean_matter_density_Msun_per_Mpc3',
    '_lagrangian_radius_from_mass',
    '_dimensionless_covariance_metrics'
]

def _safe_covariance(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Return 3x3 covariance of points (N,3) with small Tikhonov regularization.
    """
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 2:
        raise ValueError("Need at least 2 points of shape (N,3) for covariance")
    C = np.cov(X.T, bias=False)
    # Regularize to keep positive-definite
    return C + eps * np.eye(3)

def _sym_eig_sqrt_inv(S: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Symmetric eigendecomposition -> matrix sqrt or inverse sqrt.
    """
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)  # avoid negatives
    if invert:
        W = np.diag(1.0 / np.sqrt(w))
    else:
        W = np.diag(np.sqrt(w))
    return (V @ W) @ V.T

def _matrix_sqrt(S: np.ndarray) -> np.ndarray:
    return _sym_eig_sqrt_inv(S, invert=False)

def _matrix_invsqrt(S: np.ndarray) -> np.ndarray:
    return _sym_eig_sqrt_inv(S, invert=True)

def _mean_matter_density_Msun_per_Mpc3() -> float:
    """
    Mean comoving matter density rho_m [Msun / Mpc^3] for fixed Omega_m, h.
    """
    Omega_m = 0.306
    h = 0.681
    # Critical density rho_c0 ≈ 2.77536627e11 * h^2 Msun / Mpc^3
    rho_c0 = 2.77536627e11 * (h ** 2)
    return Omega_m * rho_c0

def _lagrangian_radius_from_mass(M200: float, rho_m_Msun_per_Mpc3: float) -> float:
    """
    Lagrangian radius R_L = (3 M / (4π rho_m))^(1/3) in Mpc.
    Use M200 (Msun) for M; rho_m is comoving mean density (Msun/Mpc^3).
    """
    return ((3.0 * M200) / (4.0 * np.pi * rho_m_Msun_per_Mpc3)) ** (1.0 / 3.0)

def _dimensionless_covariance_metrics(X_init_data: np.ndarray,
                                      X_init_ctrl: np.ndarray,
                                      R_L: float) -> dict:
    """
    Compute dimensionless scatter and information gain (nats & bits)
    from covariance matrices normalized by R_L^2.
    """
    # center
    Xi_d = X_init_data - X_init_data.mean(axis=0, keepdims=True)
    Xi_c = X_init_ctrl - X_init_ctrl.mean(axis=0, keepdims=True)

    Sd = _safe_covariance(Xi_d) / (R_L ** 2)
    Sc = _safe_covariance(Xi_c) / (R_L ** 2)

    # dimensionless scatter (RMS)
    s_data = float(np.sqrt(np.trace(Sd)))
    s_ctrl = float(np.sqrt(np.trace(Sc)))
    s_ratio = s_ctrl / s_data if s_data > 0 else np.nan

    # information gain (nats & bits): 0.5 * ln(det(Sc)/det(Sd))
    det_Sd = float(np.linalg.det(Sd))
    det_Sc = float(np.linalg.det(Sc))
    det_Sd = max(det_Sd, 1e-36)
    det_Sc = max(det_Sc, 1e-36)
    info_nats = 0.5 * np.log(det_Sc / det_Sd)
    info_bits = info_nats / np.log(2.0)

    # legacy 3D "volume" via covariance determinant^1/2 (dimensionless)
    vol_data_dimless = float(np.sqrt(det_Sd))
    vol_ctrl_dimless = float(np.sqrt(det_Sc))
    vol_ratio_dimless = vol_ctrl_dimless / vol_data_dimless if vol_data_dimless > 0 else np.nan

    return {
        "s_data": s_data, "s_ctrl": s_ctrl, "s_ratio": s_ratio,
        "info_nats": info_nats, "info_bits": info_bits,
        "vol_ratio_dimless": vol_ratio_dimless
    }
