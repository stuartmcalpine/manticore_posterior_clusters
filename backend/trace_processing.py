import numpy as np
from .math_utils import _safe_covariance

__all__ = [
    '_extract_positions_at_or_after_snapshot',
    '_get_initial_final_positions',
]

def _extract_positions_at_or_after_snapshot(trace: dict, key: str, target_snapshot: int) -> np.ndarray | None:
    """
    Return 3D position from 'key' at earliest snapshot >= target_snapshot.
    """
    snapshots = trace['snapshots']
    positions = trace.get(key, None)
    if positions is None:
        return None
    valid = snapshots[snapshots >= target_snapshot]
    if len(valid) == 0:
        return None
    earliest = int(np.min(valid))
    idx = np.where(snapshots == earliest)[0][0]
    return positions[idx]

def _get_initial_final_positions(traces: list, init_snap: int, final_snap: int = 77) -> tuple[np.ndarray, np.ndarray]:
    """
    Return arrays (N,3) of initial and final positions for provided traces.
    Uses BoundSubhalo/CentreOfMass if present, else SO/200_crit/CentreOfMass.
    For initial: earliest snapshot >= init_snap; for final: exact 'final_snap'.
    """
    init_pts, fin_pts = [], []

    for tr in traces:
        # choose pos key
        if 'BoundSubhalo/CentreOfMass' in tr:
            poskey = 'BoundSubhalo/CentreOfMass'
        elif 'SO/200_crit/CentreOfMass' in tr:
            poskey = 'SO/200_crit/CentreOfMass'
        else:
            continue

        # initial
        p_init = _extract_positions_at_or_after_snapshot(tr, poskey, init_snap)
        # final
        snaps = tr['snapshots']
        fin_idx = np.where(snaps == final_snap)[0]
        if len(fin_idx) == 0 or p_init is None:
            continue
        p_fin = tr[poskey][fin_idx[0]]

        init_pts.append(p_init)
        fin_pts.append(p_fin)

    if len(init_pts) < 2 or len(fin_pts) < 2:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.asarray(init_pts), np.asarray(fin_pts)
