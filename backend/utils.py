# Backward compatibility imports - can be removed after transition period
from .io import *
from .analysis import *
from .math_utils import *
from .trace_processing import *

__all__ = [
    # io.py exports
    'ensure_output_dir',
    'save_clusters_to_hdf5',
    'load_clusters_from_hdf5',
    'save_halo_traces_to_hdf5',
    'load_halo_traces_index',
    'load_specific_halo_traces',
    'load_halo_traces_from_hdf5',
    'get_cluster_trace_info',
    'load_single_cluster_traces',
    'find_clusters_in_window',
    'load_single_cluster_members',
    # analysis.py exports
    'analyze_volume_ratios_batch',
    'find_control_matches_batch',
    'find_control_matches_and_recenter_single',
    'calculate_lagrangian_volume',
    # math_utils.py exports
    '_safe_covariance',
    '_sym_eig_sqrt_inv',
    '_matrix_sqrt',
    '_matrix_invsqrt',
    '_mean_matter_density_Msun_per_Mpc3',
    '_lagrangian_radius_from_mass',
    '_dimensionless_covariance_metrics',
    # trace_processing.py exports
    '_extract_positions_at_or_after_snapshot',
    '_get_initial_final_positions',
    '_apply_matched_final_affine'
]
