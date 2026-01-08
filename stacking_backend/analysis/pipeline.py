import numpy as np
from astropy.cosmology import Planck18
from ..data import GenericMapLoader, PatchExtractor
from ..config.map_config import MapConfig
from ..utils.validation import InputValidator
from ..utils.statistics import StatisticsCalculator
from .stacking import PatchStacker
from .individual_clusters import IndividualClusterAnalyzer
from .validation import NullTestValidator
import threading

class ClusterAnalysisPipeline:
    """Main analysis pipeline with r/r500 scaling"""
    
    def __init__(self, map_config=None):
        """
        Initialize the analysis pipeline
        
        Parameters
        ----------
        map_config : MapConfig, optional
            Configuration for map loading. If None, will use default PR4 configuration
        """
        if map_config is None:
            raise ValueError("MapConfig is required. Please provide a MapConfig object.")
        
        self.map_config = map_config
        self._lock = threading.Lock()
        self._initialize_data()
    
    def _initialize_data(self):
        """Load map data using MapConfig and initialize components with error handling"""
        try:
            # Use GenericMapLoader with the provided MapConfig
            loader = GenericMapLoader(self.map_config)
            data = loader.load_data()
            
            # Initialize PatchExtractor with loaded data
            self.patch_extractor = PatchExtractor(
                y_map=data["map"],
                combined_mask=data.get("combined_mask"),
                nside=data["nside"],
                nested=data.get("nested", False),
                coord_system=data.get("coord_system", "G")
            )
            
            # Initialize analysis components
            self.stacker = PatchStacker(self.patch_extractor)
            self.individual_analyzer = IndividualClusterAnalyzer(self.patch_extractor)
            self.validator = NullTestValidator(self.patch_extractor)
            
            print(f"✅ Pipeline initialized with {self.map_config.map_format.value} format map")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize analysis pipeline: {str(e)}") from e
    
    def run_individual_r500_analysis_with_validation(self, coord_list, inner_r500_factor=1.0, outer_r500_factor=3.0,
                                               patch_size_r500=10.0, npix=256, max_patches=None,
                                               min_coverage=0.9, run_null_tests=True,
                                               n_bootstrap=500, n_random=500, weights=None, weight_vars=None,
                                               velocity_weighting_scheme='tanimura',
                                               subtract_background=False, bg_inner_r500=3.0,
                                               bg_outer_r500=5.0, analysis_mode='tsz'):
        """
        Full analysis pipeline with r/r500 scaling.
        
        Parameters
        ----------
        coord_list : list
            List of cluster coordinates (lon, lat, r500[, z, ...])
        inner_r500_factor : float
            Inner aperture radius in units of R500
        outer_r500_factor : float
            Outer aperture radius in units of R500
        patch_size_r500 : float
            Patch size in units of R500 (patch spans ±patch_size_r500/2)
        npix : int
            Number of pixels per side
        max_patches : int or None
            Maximum number of patches to stack
        min_coverage : float
            Minimum coverage fraction
        run_null_tests : bool
            Whether to run null tests
        n_bootstrap : int
            Number of bootstrap samples
        n_random : int
            Number of random pointings for null tests
        weights : array-like or None
            Optional per-cluster weights (e.g. LOS velocities for kSZ).
        weight_vars : array-like or None
            Optional per-cluster weight variances (e.g. velocity variances from posterior).
        velocity_weighting_scheme : str, default='tanimura'
            Weighting scheme for kSZ stacking. One of:
            - 'tanimura': Original Tanimura et al. (2021) estimator.
              Uses w_i = v_i / σ²_T,i. Treats velocities as perfectly known.
              Use when velocities have negligible uncertainty.
            - 'optimal_posterior': Optimal estimator for velocity posteriors.
              Accounts for both CMB variance and velocity uncertainty.
              Requires weight_vars (velocity variances from posterior).
              This is the minimum-variance unbiased estimator for velocity
              posteriors v̂_i ~ N(v_i, σ²_v,i). Downweights clusters with
              uncertain or small velocities.
        subtract_background : bool, default=False
            Whether to subtract background in stacking. Default is False because
            aperture photometry already performs background subtraction at physical
            scales (1.5-2.5 × R500), which is the standard approach in tSZ literature.
            Set to True only if you need to remove large-scale map systematics.

            IMPORTANT: When True, background is subtracted at physical scales
            (bg_inner_r500 to bg_outer_r500) to ensure consistent zero-points
            after r/r500 rescaling.
        bg_inner_r500 : float, default=3.0
            Inner radius for background annulus in units of R500. Only used when
            subtract_background=True. Default 3.0 × R500 is outside typical cluster
            signal extent (~2 × R500).
        bg_outer_r500 : float, default=5.0
            Outer radius for background annulus in units of R500. Only used when
            subtract_background=True. Provides robust background in noise-dominated
            regime.
        analysis_mode : str, default='tsz'
            'tsz': Full tSZ analysis with aperture photometry, bootstrap, significance
            'ksz': Streamlined kSZ analysis - skip aperture photometry

        Notes
        -----
        Background Subtraction Strategy:
        The pipeline performs background subtraction at two stages:
        1. Aperture photometry (always): Measures signal relative to 1.5-2.5 × R500
           annulus for each cluster. This is standard practice in tSZ studies.
        2. Stacking (optional): Additional large-scale mode removal if
           subtract_background=True. Use only when maps have significant gradients.

        For standard tSZ analysis, use subtract_background=False (default).
        """

        # Input validation
        InputValidator.validate_coord_list(coord_list)
        InputValidator.validate_analysis_params(patch_size_r500, npix, inner_r500_factor, outer_r500_factor, min_coverage)
        
        if analysis_mode not in ['tsz', 'ksz']:
            raise ValueError(f"analysis_mode must be 'tsz' or 'ksz', got '{analysis_mode}'")
        
        profile_only_mode = (analysis_mode == 'ksz')
        
        if weights is not None:
            weights = np.asarray(weights)
            if len(weights) != len(coord_list):
                raise ValueError(
                    f"weights must have same length as coord_list "
                    f"({len(weights)} vs {len(coord_list)})"
                )
        
        if weight_vars is not None:
            weight_vars = np.asarray(weight_vars)
            if len(weight_vars) != len(coord_list):
                raise ValueError(
                    f"weight_vars must have same length as coord_list "
                    f"({len(weight_vars)} vs {len(coord_list)})"
                )
        
        print("🚀 CLUSTER ANALYSIS PIPELINE")
        print("="*70)
        print(f"⚙️  Analysis mode: {analysis_mode.upper()}")
        print(f"⚙️  Scaling mode: R/R500")
        print(f"⚙️  Patch size: ±{patch_size_r500/2:.1f} × R500")
        if weights is not None:
            print(f"⚙️  Velocity weighting scheme: {velocity_weighting_scheme}")
            if weight_vars is not None:
                print(f"⚙️  Using velocity variances from posterior")
        if profile_only_mode:
            print(f"⚙️  Profile-only mode: Skipping aperture photometry, bootstrap, and null tests")
        
        # Step 1: Calculate individual cluster measurements
        print(f"\n🔍 Step 1: Individual cluster processing...")
        individual_results, rejection_stats, quality_stats = self.individual_analyzer.calculate_measurements(
            coord_list=coord_list,
            inner_r500_factor=inner_r500_factor,
            outer_r500_factor=outer_r500_factor,
            patch_size_r500=patch_size_r500,
            npix=npix,
            min_coverage=min_coverage,
            weights=weights,
            weight_vars=weight_vars,
            profile_only_mode=profile_only_mode
        )
        
        if not individual_results:
            raise ValueError('No valid individual measurements obtained')
        
        valid_coords = [result['coords'] for result in individual_results]
        
        # Extract weights corresponding to valid clusters, if provided
        if weights is not None:
            individual_weights = np.array([result.get('weight', np.nan) for result in individual_results])
            if np.any(~np.isfinite(individual_weights)):
                raise RuntimeError("Non-finite weights encountered in individual_results")
        else:
            individual_weights = None
        
        # Extract weight_vars corresponding to valid clusters, if provided
        if weight_vars is not None:
            individual_weight_vars = np.array([result.get('weight_var', np.nan) for result in individual_results])
            if np.any(~np.isfinite(individual_weight_vars)):
                raise RuntimeError("Non-finite weight_vars encountered in individual_results")
        else:
            individual_weight_vars = None
        
        # For kSZ mode, skip steps 2-4 (bootstrap, null tests, significance)
        bootstrap_results = None
        null_results = None
        significance_results = None
        individual_delta_y = None
        individual_errors = None
        measurement_values = None
        measurement_errors = None
        measurement_label = None
        
        if not profile_only_mode:
            # tSZ mode: full analysis pipeline
            
            # Extract measurements and errors
            individual_delta_y = np.array([result['delta_y'] for result in individual_results])
            individual_errors = np.array([result['delta_y_error'] for result in individual_results])
            
            # Define the estimator used for bootstrap & significance
            if individual_weights is None:
                measurement_values = individual_delta_y
                measurement_errors = individual_errors
                measurement_label = "delta_y"
            else:
                measurement_values = individual_delta_y * individual_weights
                measurement_errors = individual_errors * np.abs(individual_weights)
                measurement_label = "weighted_delta"
            
            # Step 2: Robust Bootstrap with proper error propagation
            print(f"\n🔄 Step 2: Robust bootstrap error estimation ({n_bootstrap} samples)...")
            bootstrap_results = self._robust_bootstrap_analysis(
                individual_results, measurement_values, measurement_errors,
                n_bootstrap
            )
            
            # Step 3: Null tests (if requested)
            if run_null_tests:
                print(f"\n🎯 Step 3: Null tests with random pointings ({n_random} samples)...")
                null_results = self.validator.run_null_tests(
                    n_random_pointings=n_random,
                    coord_list=valid_coords,
                    inner_r500_factor=inner_r500_factor,
                    outer_r500_factor=outer_r500_factor,
                    patch_size_r500=patch_size_r500,
                    npix=npix,
                    min_coverage=min_coverage,
                    weights=individual_weights
                )
            
            # Step 4: Calculate corrected significance
            print(f"\n📈 Step 4: Computing corrected detection significance...")
            significance_results = self._calculate_corrected_significance(
                measurement_values, measurement_errors, bootstrap_results, null_results
            )
        else:
            # kSZ mode: skip bootstrap and null tests
            print(f"\n⏭️  Steps 2-4: Skipped (profile-only mode)")
        
        # Step N: Create stacked patch
        step_num = 2 if profile_only_mode else 5
        print(f"\n📚 Step {step_num}: Stacking patches in R/R500 mode...")
        stacked_patch, stacking_info, stack_rejection_stats = self.stacker.stack_patches(
            coord_list=valid_coords,
            patch_size_r500=patch_size_r500,
            npix=npix,
            min_coverage=min_coverage,
            max_patches=max_patches,
            weights=individual_weights,
            weight_vars=individual_weight_vars,
            velocity_weighting_scheme=velocity_weighting_scheme,
            subtract_background=subtract_background,
            bg_inner_r500=bg_inner_r500,
            bg_outer_r500=bg_outer_r500
        )
        if stacked_patch is None:
            raise ValueError('No valid patches for stacking')
        
        # Compile results
        results = self._compile_results(
            individual_results, individual_delta_y, individual_errors,
            individual_weights, individual_weight_vars, velocity_weighting_scheme,
            measurement_values, measurement_errors, measurement_label,
            bootstrap_results, significance_results, stacked_patch, stacking_info,
            rejection_stats, quality_stats, coord_list, patch_size_r500, npix,
            inner_r500_factor, outer_r500_factor, null_results,
            profile_only_mode
        )
        
        self._print_summary(results, profile_only_mode)
        
        return results

    def _robust_bootstrap_analysis(self, individual_results, values, errors, n_bootstrap):
        """Robust bootstrap that properly combines sample and measurement variance"""
        
        n_clusters = len(individual_results)
        bootstrap_means = []
        bootstrap_combined_errors = []
        
        for i in range(n_bootstrap):
            # Resample clusters with replacement
            bootstrap_indices = np.random.choice(n_clusters, size=n_clusters, replace=True)
            
            # Get resampled measurements and errors
            bootstrap_values = values[bootstrap_indices]
            bootstrap_errors = errors[bootstrap_indices]
            
            # Calculate bootstrap mean
            bootstrap_mean = np.mean(bootstrap_values)
            bootstrap_means.append(bootstrap_mean)
            
            # Calculate combined error for this bootstrap sample
            measurement_var = np.mean(bootstrap_errors**2)
            sample_var = np.var(bootstrap_values)
            combined_error = np.sqrt(measurement_var/n_clusters + sample_var/n_clusters)
            bootstrap_combined_errors.append(combined_error)
        
        bootstrap_means = np.array(bootstrap_means)
        bootstrap_combined_errors = np.array(bootstrap_combined_errors)
        
        # Calculate final statistics
        mean_estimate = np.mean(bootstrap_means)
        
        # Sample variance from bootstrap distribution
        sample_variance = np.var(bootstrap_means)
        
        # Average measurement variance
        measurement_variance = np.mean(errors**2) / n_clusters
        
        # Total variance combining both sources
        total_variance = sample_variance + measurement_variance
        total_error = np.sqrt(total_variance)
        
        # Calculate confidence intervals
        bootstrap_means_sorted = np.sort(bootstrap_means)
        ci_16 = bootstrap_means_sorted[int(0.16 * n_bootstrap)]
        ci_84 = bootstrap_means_sorted[int(0.84 * n_bootstrap)]
        ci_2p5 = bootstrap_means_sorted[int(0.025 * n_bootstrap)]
        ci_97p5 = bootstrap_means_sorted[int(0.975 * n_bootstrap)]
        
        return {
            'bootstrap_mean': mean_estimate,
            'sample_variance': sample_variance,
            'measurement_variance': measurement_variance,
            'total_variance': total_variance,
            'total_error': total_error,
            'bootstrap_samples': bootstrap_means,
            'confidence_interval_68': (ci_16, ci_84),
            'confidence_interval_95': (ci_2p5, ci_97p5),
            'n_bootstrap': n_bootstrap
        }
    
    def _calculate_corrected_significance(self, values, errors, bootstrap_results, null_results):
        """Calculate corrected significance with proper error propagation and null correction"""
        
        # Get signal estimate
        signal = bootstrap_results['bootstrap_mean']
        
        # Correct for null bias if significant
        null_corrected_signal = signal
        null_bias = 0
        if null_results is not None:
            null_bias = null_results['random_mean']
            null_std = null_results['random_std']
            
            if np.abs(null_bias) > null_std / 3:
                null_corrected_signal = signal - null_bias
                print(f"   📊 Applying null bias correction: {null_bias:.2e}")
            else:
                print(f"   ✅ Null bias negligible: {null_bias:.2e} ± {null_std:.2e}")
        
        # Calculate various significance metrics
        simple_significance = signal / bootstrap_results['total_error']
        null_corrected_significance = null_corrected_signal / bootstrap_results['total_error']
        
        null_relative_significance = None
        if null_results is not None:
            null_std = null_results['random_std']
            if null_std > 0:
                null_relative_significance = (signal - null_bias) / null_std
        
        conservative_error = bootstrap_results['total_error']
        if null_results is not None and null_results['random_std'] > conservative_error:
            conservative_error = null_results['random_std']
        conservative_significance = null_corrected_signal / conservative_error
        
        return {
            'signal': signal,
            'null_bias': null_bias,
            'corrected_signal': null_corrected_signal,
            'total_error': bootstrap_results['total_error'],
            'simple_significance': simple_significance,
            'null_corrected_significance': null_corrected_significance,
            'null_relative_significance': null_relative_significance,
            'conservative_significance': conservative_significance,
            'primary_significance': null_corrected_significance
        }
    
    def _compile_results(self, individual_results, individual_delta_y, individual_errors,
                        individual_weights, individual_weight_vars, velocity_weighting_scheme,
                        measurement_values, measurement_errors, measurement_label,
                        bootstrap_results, significance_results, stacked_patch, stacking_info,
                        rejection_stats, quality_stats, coord_list, patch_size_r500, npix,
                        inner_r500_factor, outer_r500_factor, null_results,
                        profile_only_mode):
        """Compile results"""
        
        # Calculate R500 statistics
        r500_values = [result['r500_deg'] for result in individual_results]
        r500_stats = StatisticsCalculator.calculate_r500_statistics(r500_values)
        
        if not profile_only_mode:
            inner_radii = [result['inner_radius_deg'] for result in individual_results]
            outer_radii = [result['outer_radius_deg'] for result in individual_results]
        else:
            inner_radii = None
            outer_radii = None
        
        weighted_mode = individual_weights is not None
        
        # Base results (always included)
        results = {
            'success': True,
            'analysis_mode': 'ksz' if profile_only_mode else 'tsz',
            
            # Stacked patch data
            'stacked_patch': stacked_patch,
            'stacking_info': stacking_info,
            
            # R500 statistics
            'r500_values': r500_values,
            'r500_median': r500_stats['median'],
            'r500_range': r500_stats['range'],
            
            # Sample information
            'n_measurements': len(individual_results),
            'n_input_coords': len(coord_list),
            'n_rejected': len(coord_list) - len(individual_results),
            'rejection_stats': rejection_stats,
            
            # Analysis parameters
            'patch_size_r500': patch_size_r500,
            'npix': npix,
            'inner_r500_factor': inner_r500_factor,
            'outer_r500_factor': outer_r500_factor,
            
            'weighted_mode': weighted_mode,
            'velocity_weighting_scheme': velocity_weighting_scheme if weighted_mode else None,
        }
        
        # Add tSZ-specific results if in tSZ mode
        if not profile_only_mode:
            results.update({
                # Main measurements
                'mean_delta_y': significance_results['corrected_signal'],
                'error_mean': significance_results['total_error'],
                'significance': significance_results['primary_significance'],
                
                # Significance metrics
                'significance_metrics': significance_results,
                
                # Error decomposition
                'error_decomposition': {
                    'sample_variance': bootstrap_results['sample_variance'],
                    'measurement_variance': bootstrap_results['measurement_variance'],
                    'total_variance': bootstrap_results['total_variance'],
                    'sample_std': np.sqrt(bootstrap_results['sample_variance']),
                    'measurement_std': np.sqrt(bootstrap_results['measurement_variance']),
                    'total_std': bootstrap_results['total_error']
                },
                
                # Bootstrap results
                'bootstrap_results': bootstrap_results,
                
                # Validation results
                'null_results': null_results,
                
                # Individual cluster results
                'individual_results': individual_results,
                'individual_measurements': individual_delta_y.tolist(),
                'individual_errors': individual_errors.tolist(),
                
                # Weighted estimator bookkeeping
                'weights': individual_weights.tolist() if individual_weights is not None else None,
                'weight_vars': individual_weight_vars.tolist() if individual_weight_vars is not None else None,
                'estimator_values': measurement_values.tolist(),
                'estimator_errors': measurement_errors.tolist(),
                'estimator_label': measurement_label,
                
                # Quality metrics
                'quality_stats': quality_stats,
                
                'inner_radii_range': (np.min(inner_radii), np.max(inner_radii)),
                'outer_radii_range': (np.min(outer_radii), np.max(outer_radii)),
                
                # Error type flag
                'error_type': 'robust_combined',
                'analysis_version': '2.0_corrected',
            })
        else:
            # kSZ mode: minimal results
            results.update({
                'individual_results': individual_results,
                'weights': individual_weights.tolist() if individual_weights is not None else None,
                'weight_vars': individual_weight_vars.tolist() if individual_weight_vars is not None else None,
                'quality_stats': None,
            })
        
        return results

    def _print_summary(self, results, profile_only_mode):
        """Print analysis summary"""
        print(f"\n🎯 ANALYSIS RESULTS:")
        print("="*50)
        
        if not profile_only_mode:
            # tSZ mode: full summary
            sig_metrics = results['significance_metrics']
            error_decomp = results['error_decomposition']
            
            print(f"📊 Signal Detection (estimator: {results['estimator_label']}):")
            print(f"   Raw signal: {sig_metrics['signal']:.2e}")
            if sig_metrics['null_bias'] != 0:
                print(f"   Null bias: {sig_metrics['null_bias']:.2e}")
            print(f"   Corrected signal: {sig_metrics['corrected_signal']:.2e}")
            
            print(f"\n📊 Error Budget:")
            print(f"   Sample std: {error_decomp['sample_std']:.2e} ({error_decomp['sample_variance']/error_decomp['total_variance']*100:.1f}%)")
            print(f"   Measurement std: {error_decomp['measurement_std']:.2e} ({error_decomp['measurement_variance']/error_decomp['total_variance']*100:.1f}%)")
            print(f"   Total error: {error_decomp['total_std']:.2e}")
            
            print(f"\n📊 Detection Significance:")
            print(f"   Primary (corrected): {sig_metrics['primary_significance']:.2f}σ")
            print(f"   Simple (uncorrected): {sig_metrics['simple_significance']:.2f}σ")
            if sig_metrics['null_relative_significance'] is not None:
                print(f"   Relative to null: {sig_metrics['null_relative_significance']:.2f}σ")
            print(f"   Conservative: {sig_metrics['conservative_significance']:.2f}σ")
            
            if results['bootstrap_results']['confidence_interval_68']:
                ci_68 = results['bootstrap_results']['confidence_interval_68']
                ci_95 = results['bootstrap_results']['confidence_interval_95']
                print(f"\n📊 Confidence Intervals:")
                print(f"   68% CI: [{ci_68[0]:.2e}, {ci_68[1]:.2e}]")
                print(f"   95% CI: [{ci_95[0]:.2e}, {ci_95[1]:.2e}]")
            
            if results['quality_stats']:
                qs = results['quality_stats']
                print(f"\n📊 Measurement Quality:")
                print(f"   Mean S/N per cluster: {qs['mean_snr']:.2f}")
                print(f"   High S/N fraction: {qs['high_snr_fraction']*100:.1f}%")
        else:
            # kSZ mode: minimal summary
            print(f"📊 Profile-Only Mode (kSZ):")
            print(f"   Stacked patch computed successfully")
        
        print(f"\n📏 Sample Statistics:")
        print(f"   Valid clusters: {results['n_measurements']}/{results['n_input_coords']}")
        print(f"   Median R₅₀₀: {results['r500_median']:.3f}°")
        
        print(f"\n⚙️  Analysis Configuration:")
        print(f"   Analysis mode: {results['analysis_mode'].upper()}")
        print(f"   Scaling mode: R/R500")
        print(f"   Patch size: ±{results['patch_size_r500']/2:.1f} × R500")
        print(f"   Stacking coordinate system: {results['stacking_info'].get('coord_system', 'r500')}")
        if results['weighted_mode']:
            print(f"   Weighted mode: True (using weights in stacking)")
        else:
            print(f"   Weighted mode: False (unweighted analysis)")
        
        print(f"\n🎉 Analysis complete!")
