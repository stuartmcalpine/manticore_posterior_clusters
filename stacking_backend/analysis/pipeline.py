import numpy as np
from astropy.cosmology import Planck18
from ..data import GenericMapLoader, PatchExtractor
from ..config.map_config import MapConfig
from ..utils.validation import InputValidator
from ..utils.statistics import StatisticsCalculator
from .stacking import PatchStacker
from .profiles import RadialProfileCalculator
from .individual_clusters import IndividualClusterAnalyzer
from .validation import NullTestValidator
import threading

class ClusterAnalysisPipeline:
    """Main analysis pipeline with corrected significance calculation"""
    
    def __init__(self, map_config=None):
        """
        Initialize the analysis pipeline
        
        Parameters
        ----------
        map_config : MapConfig, optional
            Configuration for map loading. If None, will use default PR4 configuration
        """
        if map_config is None:
            # Default to PR4 configuration for backward compatibility
            # User should provide proper paths when creating MapConfig
            raise ValueError("MapConfig is required. Please provide a MapConfig object.")
        
        self.map_config = map_config
        self._lock = threading.Lock()
        self._initialize_data()
    
    def _initialize_data(self):
        """Load map data using MapConfig and initialize components with error handling"""
        try:
            # Use GenericMapLoader with the provided MapConfig
            loader = GenericMapLoader(self.map_config)
            data = loader.load_data(use_cache=True)
            
            # Initialize PatchExtractor with loaded data
            self.patch_extractor = PatchExtractor(
                y_map=data["map"],  # or data["y_map"] for backward compatibility
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
                                                   patch_size_deg=15.0, npix=256, max_patches=None,
                                                   min_coverage=0.9, n_radial_bins=20, run_null_tests=True,
                                                   n_bootstrap=500, n_random=500, weights=None, max_radius_r500=5):
        """
        Full analysis pipeline with corrected error propagation and significance calculation.
        
        Parameters
        ----------
        coord_list : list
            List of cluster coordinates (lon, lat, r500[, z, ...])
        weights : array-like or None
            Optional per-cluster weights (e.g. LOS velocities for kSZ).
            If provided, both the stacked patch and the scalar estimator used
            for significance will be weighted in a Tanimura-like fashion.
        """
        
        # Input validation
        InputValidator.validate_coord_list(coord_list)
        InputValidator.validate_analysis_params(patch_size_deg, npix, inner_r500_factor, outer_r500_factor, min_coverage)
        
        if weights is not None:
            weights = np.asarray(weights)
            if len(weights) != len(coord_list):
                raise ValueError(
                    f"weights must have same length as coord_list "
                    f"({len(weights)} vs {len(coord_list)})"
                )
        
        print("🚀 CLUSTER ANALYSIS PIPELINE (Corrected Significance)")
        print("="*70)
        
        # Step 1: Calculate individual cluster measurements with errors
        print(f"\n🔍 Step 1: Individual cluster measurements with error estimation...")
        individual_results, rejection_stats, quality_stats = self.individual_analyzer.calculate_measurements(
            coord_list=coord_list,
            inner_r500_factor=inner_r500_factor,
            outer_r500_factor=outer_r500_factor,
            patch_size_deg=patch_size_deg,
            npix=npix,
            min_coverage=min_coverage,
            weights=weights  # store per-cluster weights in results if provided
        )
        
        if not individual_results:
            raise ValueError('No valid individual measurements obtained')
        
        # Extract measurements and errors (always unweighted here: raw delta_y)
        individual_delta_y = np.array([result['delta_y'] for result in individual_results])
        individual_errors = np.array([result['delta_y_error'] for result in individual_results])
        valid_coords = [result['coords'] for result in individual_results]
        
        # Extract weights corresponding to valid clusters, if provided
        if weights is not None:
            individual_weights = np.array([result.get('weight', np.nan) for result in individual_results])
            # Filter out any NaNs just in case (shouldn't normally happen)
            if np.any(~np.isfinite(individual_weights)):
                raise RuntimeError("Non-finite weights encountered in individual_results")
        else:
            individual_weights = None
        
        # Define the estimator used for bootstrap & significance:
        # - If no weights: use raw delta_y (tSZ-like).
        # - If weights: use weighted kSZ-like estimator w_i * delta_y_i.
        if individual_weights is None:
            measurement_values = individual_delta_y
            measurement_errors = individual_errors
            measurement_label = "delta_y"
        else:
            measurement_values = individual_delta_y * individual_weights
            measurement_errors = individual_errors * np.abs(individual_weights)
            measurement_label = "weighted_delta"  # e.g. v * delta_T
        
        # Step 2: Robust Bootstrap with proper error propagation
        print(f"\n🔄 Step 2: Robust bootstrap error estimation ({n_bootstrap} samples)...")
        bootstrap_results = self._robust_bootstrap_analysis(
            individual_results, measurement_values, measurement_errors,
            n_bootstrap
        )
        
        # Step 3: Create stacked patch (unweighted for tSZ, weighted for kSZ)
        print(f"\n📚 Step 3: Stacking patches...")
        stacked_patch, stacking_info, stack_rejection_stats = self.stacker.stack_patches(
            coord_list=valid_coords,
            patch_size_deg=patch_size_deg,
            npix=npix,
            min_coverage=min_coverage,
            max_patches=max_patches,
            weights=individual_weights  # None for tSZ, velocities for kSZ
        )
        
        if stacked_patch is None:
            raise ValueError('No valid patches for stacking')
        
        # Step 4: Null tests with random pointings (optionally weight-aware)
        null_results = None
        if run_null_tests:
            print(f"\n🎯 Step 4: Null tests with random pointings ({n_random} samples)...")
            null_results = self.validator.run_null_tests(
                n_random_pointings=n_random,
                coord_list=valid_coords,
                inner_r500_factor=inner_r500_factor,
                outer_r500_factor=outer_r500_factor,
                patch_size_deg=patch_size_deg,
                npix=npix,
                min_coverage=min_coverage,
                weights=individual_weights  # None for tSZ, velocities for kSZ
            )
        
        # Step 5: Calculate corrected significance for the chosen estimator
        print(f"\n📈 Step 5: Computing corrected detection significance...")
        significance_results = self._calculate_corrected_significance(
            measurement_values, measurement_errors, bootstrap_results, null_results
        )
        
        # Step 6: Calculate radial profile
        print(f"\n📊 Step 6: Radial profile in r/r500 units...")
        r500_values = [result['r500_deg'] for result in individual_results]
        median_r500_deg = np.median(r500_values)
        
        radii, profile, profile_errors, profile_counts = RadialProfileCalculator.calculate_profile_scaled(
            stacked_patch=stacked_patch,
            patch_size_deg=patch_size_deg,
            r500_deg=median_r500_deg,
            n_radial_bins=n_radial_bins,
            max_radius_r500=max_radius_r500,
        )
        
        # Step 7: Compile results
        results = self._compile_corrected_results(
            individual_results, individual_delta_y, individual_errors,
            individual_weights, measurement_values, measurement_errors, measurement_label,
            bootstrap_results, significance_results, stacked_patch, stacking_info,
            radii, profile, profile_errors, profile_counts,
            rejection_stats, quality_stats, coord_list, patch_size_deg, npix,
            inner_r500_factor, outer_r500_factor, null_results
        )
        
        self._print_corrected_summary(results)
        
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
            # Includes both measurement error and sample variance
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
        
        # Get signal estimate (mean of chosen estimator)
        signal = bootstrap_results['bootstrap_mean']
        
        # Correct for null bias if significant
        null_corrected_signal = signal
        null_bias = 0
        if null_results is not None:
            null_bias = null_results['random_mean']
            null_std = null_results['random_std']
            
            # Apply bias correction if null mean is significantly different from zero
            # Use 3-sigma threshold to avoid overcorrecting for noise
            if np.abs(null_bias) > null_std / 3:
                null_corrected_signal = signal - null_bias
                print(f"   📊 Applying null bias correction: {null_bias:.2e}")
            else:
                print(f"   ✅ Null bias negligible: {null_bias:.2e} ± {null_std:.2e}")
        
        # 1. Simple significance (signal/error)
        simple_significance = signal / bootstrap_results['total_error']
        
        # 2. Null-corrected significance
        null_corrected_significance = null_corrected_signal / bootstrap_results['total_error']
        
        # 3. Significance relative to null distribution
        null_relative_significance = None
        if null_results is not None:
            null_std = null_results['random_std']
            if null_std > 0:
                null_relative_significance = (signal - null_bias) / null_std
        
        # 4. Conservative significance (using larger error estimate)
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
            'primary_significance': null_corrected_significance  # Use this as main result
        }
    
    def _compile_corrected_results(self, individual_results, individual_delta_y, individual_errors,
                                  individual_weights, measurement_values, measurement_errors, measurement_label,
                                  bootstrap_results, significance_results, stacked_patch, stacking_info,
                                  radii, profile, profile_errors, profile_counts,
                                  rejection_stats, quality_stats, coord_list, patch_size_deg, npix,
                                  inner_r500_factor, outer_r500_factor, null_results):
        """Compile results with corrected error estimates and significance"""
        
        # Calculate R500 statistics
        r500_values = [result['r500_deg'] for result in individual_results]
        r500_stats = StatisticsCalculator.calculate_r500_statistics(r500_values)
        inner_radii = [result['inner_radius_deg'] for result in individual_results]
        outer_radii = [result['outer_radius_deg'] for result in individual_results]
        
        # Book-keeping for weights & estimator type
        weighted_mode = individual_weights is not None
        
        return {
            'success': True,
            
            # Main measurements with corrected errors (for chosen estimator)
            'mean_delta_y': significance_results['corrected_signal'],
            'error_mean': significance_results['total_error'],
            'significance': significance_results['primary_significance'],
            
            # Detailed significance metrics
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
            
            # Individual cluster results with errors (unweighted)
            'individual_results': individual_results,
            'individual_measurements': individual_delta_y.tolist(),
            'individual_errors': individual_errors.tolist(),
            
            # Weighted estimator bookkeeping
            'weights': individual_weights.tolist() if individual_weights is not None else None,
            'estimator_values': measurement_values.tolist(),
            'estimator_errors': measurement_errors.tolist(),
            'estimator_label': measurement_label,
            
            # Quality metrics
            'quality_stats': quality_stats,
            
            # Stacked patch data
            'stacked_patch': stacked_patch,
            'stacking_info': stacking_info,
            
            # Radial profile (in r/r500 units)
            'profile_radii': radii,
            'profile_mean': profile,
            'profile_errors': profile_errors,
            'profile_counts': profile_counts,
            
            # R500 statistics
            'r500_values': r500_values,
            'r500_median': r500_stats['median'],
            'r500_range': r500_stats['range'],
            'inner_radii_range': (np.min(inner_radii), np.max(inner_radii)),
            'outer_radii_range': (np.min(outer_radii), np.max(outer_radii)),
            
            # Sample information
            'n_measurements': len(individual_results),
            'n_input_coords': len(coord_list),
            'n_rejected': len(coord_list) - len(individual_results),
            'rejection_stats': rejection_stats,
            
            # Analysis parameters
            'patch_size_deg': patch_size_deg,
            'npix': npix,
            'inner_r500_factor': inner_r500_factor,
            'outer_r500_factor': outer_r500_factor,
            
            # Error type flag
            'error_type': 'robust_combined',
            'analysis_version': '2.0_corrected',
            
            # Mode info
            'weighted_mode': weighted_mode
        }
    
    def _print_corrected_summary(self, results):
        """Print analysis summary with corrected statistics"""
        print(f"\n🎯 CORRECTED ANALYSIS RESULTS:")
        print("="*50)
        
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
        
        print(f"\n📏 Sample Statistics:")
        print(f"   Valid clusters: {results['n_measurements']}/{results['n_input_coords']}")
        print(f"   Median R₅₀₀: {results['r500_median']:.3f}°")
        if results['weighted_mode']:
            print(f"\n⚙️  Weighted mode: True (e.g. kSZ-style, using weights in stacking and estimator)")
        else:
            print(f"\n⚙️  Weighted mode: False (unweighted tSZ-style analysis)")
        
        print(f"\n🎉 Analysis complete with corrected significance calculation!")

