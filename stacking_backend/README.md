# Stacking Backend

A comprehensive Python package for analyzing galaxy clusters through the thermal and kinematic Sunyaev-Zel'dovich (tSZ/kSZ) effects using Planck data.

## Overview

The `stacking_backend` package performs statistical analysis of galaxy clusters by:
- **tSZ mode**: Measuring the Compton-y parameter signal in Planck maps with full aperture photometry, bootstrap error estimation, and null tests
- **kSZ mode**: Velocity-weighted stacking to detect the kinematic SZ effect with optimal estimators for velocity posteriors

The code implements r/r500 scaling to stack clusters of different sizes at consistent physical scales, following the methodology of Tanimura et al. (2021) with extensions for velocity posterior uncertainties.

## Key Features

- 🔭 **Dual Analysis Modes**: Full tSZ photometry or streamlined kSZ velocity-weighted stacking
- 📐 **Physical Scaling**: r/r500 rescaling ensures consistent physical scales across clusters
- 📊 **Optimal Estimators**: Three velocity-weighting schemes including a custom minimum-variance estimator for velocity posteriors
- ✅ **Robust Statistics**: Bootstrap error estimation with proper variance decomposition
- 🎯 **Validation Framework**: Null tests with mask-bias correction for tSZ
- 🔧 **Flexible Configuration**: Extensive parameter control for different science cases

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Analysis Modes](#analysis-modes)
4. [Velocity Weighting Estimators](#velocity-weighting-estimators)
5. [Background Subtraction Strategy](#background-subtraction-strategy)
6. [Configuration Parameters](#configuration-parameters)
7. [Output Products](#output-products)
8. [Examples](#examples)
9. [Scientific Background](#scientific-background)

---

## Installation

```bash
# Clone the repository
cd stacking_backend

# Install dependencies
pip install numpy scipy matplotlib healpy astropy h5py
```

### Dependencies
- `numpy` - Numerical operations
- `scipy` - Interpolation and statistics
- `matplotlib` - Plotting
- `healpy` - HEALPix map operations
- `astropy` - Coordinate transformations
- `h5py` - Catalog reading

---

## Quick Start

### tSZ Analysis (Standard)

```python
from stacking_backend import ClusterAnalysisPipeline
from stacking_backend.config import MapConfig

# Configure map paths
map_config = MapConfig(
    map_path="path/to/planck_y_map.fits",
    mask_path="path/to/masks.fits",
    map_format="planck_pr4"
)

# Initialize pipeline
pipeline = ClusterAnalysisPipeline(map_config)

# Define cluster coordinates: [lon_gal, lat_gal, R500_deg, redshift]
clusters = [
    [120.5, 45.2, 0.15, 0.05],
    [230.1, -30.4, 0.18, 0.08],
    # ... more clusters
]

# Run tSZ analysis
results = pipeline.run_individual_r500_analysis_with_validation(
    coord_list=clusters,
    inner_r500_factor=1.0,      # Inner aperture at 1 × R500
    outer_r500_factor=3.0,      # Outer aperture at 3 × R500
    analysis_mode='tsz',        # Full tSZ analysis
    run_null_tests=True,        # Validate detection
    n_bootstrap=500             # Error estimation samples
)

print(f"Detection: {results['significance']:.1f}σ")
print(f"Signal: {results['mean_delta_y']:.2e} ± {results['error_mean']:.2e}")
```

### kSZ Analysis (Velocity-Weighted)

```python
# Load cluster velocities from your reconstruction
velocities = [...]           # LOS velocities in km/s
velocity_vars = [...]        # Velocity variances from posterior

# Run kSZ analysis with optimal posterior estimator
results = pipeline.run_individual_r500_analysis_with_validation(
    coord_list=clusters,
    weights=velocities,
    weight_vars=velocity_vars,
    velocity_weighting_scheme='optimal_posterior',  # Use optimal estimator
    analysis_mode='ksz',                            # Skip aperture photometry
    patch_size_r500=10.0,
    npix=256
)

# Extract stacked patch and compute radial profile
from stacking_backend.analysis import RadialProfileCalculator

radii, profile, errors, counts = RadialProfileCalculator.calculate_profile_from_results(
    results, n_radial_bins=20
)
```

---

## Analysis Modes

### `analysis_mode='tsz'` (Default)

**Full thermal SZ analysis with aperture photometry**

**Pipeline Steps:**
1. Individual cluster measurements (aperture photometry at each cluster's R500)
2. Bootstrap error estimation (500 resamples)
3. Null tests with random sky positions
4. Significance calculation with null bias correction
5. Patch stacking in r/r500 coordinates

**Output:** Complete statistical analysis with significance, error decomposition, null test results

**Use when:**
- Measuring tSZ signal strength
- Deriving Y500-M500 scaling relations
- Need rigorous statistical validation

### `analysis_mode='ksz'` (Profile-Only)

**Streamlined kinematic SZ analysis for velocity-weighted stacking**

**Pipeline Steps:**
1. Individual cluster processing (skip aperture photometry)
2. Velocity-weighted stacking with chosen estimator
3. Radial profile calculation

**Output:** Stacked patch in r/r500 coordinates for profile analysis

**Use when:**
- Detecting kSZ signal with velocity weighting
- Have velocity posteriors from reconstruction
- Perform null tests via velocity shuffling in notebooks

**Note:** Null tests for kSZ should be done externally by shuffling cluster-velocity pairings and re-running the pipeline.

---

## Velocity Weighting Estimators

Three estimators are available for kSZ analysis via the `velocity_weighting_scheme` parameter:

### 1. `'simple'` (Baseline)

**Simple velocity-weighted mean without inverse-variance weighting**

```
Stacked(r) = Σᵢ [Tᵢ(r) · vᵢ] / Σᵢ [|vᵢ|]
```

**Properties:**
- Equal weighting for all clusters regardless of CMB noise level
- No inverse-variance weighting by patch variance
- Simplest estimator for velocity-weighted stacking

**When to use:**
- As a baseline for comparison with other estimators
- When CMB variance is approximately uniform across all patches
- When you want to avoid any assumptions about noise properties

**Requirements:**
- `weights`: Velocity values
- `weight_vars`: Not required

### 2. `'tanimura'`

**Original Tanimura et al. (2021) estimator**

```
Stacked(r) = Σᵢ [Tᵢ(r) · vᵢ / σ²ᵀ,ᵢ] / Σᵢ [|vᵢ| / σ²ᵀ,ᵢ]
```

**Properties:**
- Inverse-variance weighted by CMB noise σ²ᵀ,ᵢ
- Treats velocities as perfectly known (no velocity uncertainty)
- Standard approach when velocity errors are negligible

**When to use:**
- Velocities from spectroscopic measurements with small uncertainties
- Simple velocity estimators without posterior distributions
- Benchmark comparisons with literature

**Requirements:**
- `weights`: Velocity values
- `weight_vars`: Not required

### 3. `'optimal_posterior'` (Recommended)

**Minimum-variance unbiased estimator for velocity posteriors**

```
Stacked(r) = Σᵢ [Tᵢ(r) / (σ²ᵀ,ᵢ · v̂ᵢ)] / Σᵢ [1 / (σ²ᵀ,ᵢ · v̂ᵢ²)]
```

**Properties:**
- Accounts for both CMB variance AND velocity uncertainty
- Downweights clusters with uncertain or small velocities
- Minimum-variance unbiased estimator for velocity posteriors

**When to use:**
- Have velocity posteriors v̂ᵢ ~ N(vᵢ, σ²ᵥ,ᵢ) from reconstruction
- Want to optimally combine information from measurements and velocity uncertainties
- Maximize signal-to-noise in stacked kSZ profile

**Requirements:**
- `weights`: Velocity posterior means
- `weight_vars`: Velocity posterior variances (required)

---

## Derivation: Optimal Posterior Estimator

### Problem Setup

For each cluster i, we have:
- **CMB observation**: Tᵢ(r) = A(r) × vᵢ + noiseᵢ(r)
  - A(r) is the universal kSZ profile we want to estimate
  - noiseᵢ ~ N(0, σ²ᵀ,ᵢ)
- **Velocity posterior**: v̂ᵢ ~ N(vᵢ, σ²ᵥ,ᵢ)
  - v̂ᵢ is the posterior mean from your velocity reconstruction
  - σ²ᵥ,ᵢ is the posterior variance

**Goal:** Estimate A(r) by optimally combining measurements, accounting for velocity uncertainties.

### Derivation

**Step 1: Expected value of the product**

We form the measurement: yᵢ(r) = Tᵢ(r) × v̂ᵢ

Since Tᵢ and v̂ᵢ are independent (CMB is independent of velocity reconstruction):

```
E[yᵢ] = E[Tᵢ] × E[v̂ᵢ] = (A × vᵢ) × vᵢ = A × vᵢ²
```

**Step 2: Variance of the product**

For independent random variables:

```
var(Tᵢ × v̂ᵢ) = var(Tᵢ)·var(v̂ᵢ) + var(Tᵢ)·E[v̂ᵢ]² + var(v̂ᵢ)·E[Tᵢ]²
```

Assuming small kSZ signals where E[Tᵢ]² << σ²ᵀ,ᵢ:

```
var(Tᵢ × v̂ᵢ) ≈ σ²ᵀ,ᵢ · σ²ᵥ,ᵢ + σ²ᵀ,ᵢ · vᵢ²
              = σ²ᵀ,ᵢ · (vᵢ² + σ²ᵥ,ᵢ)
```

Since E[v̂ᵢ²] = vᵢ² + σ²ᵥ,ᵢ, we can approximate:

```
var(Tᵢ × v̂ᵢ) ≈ σ²ᵀ,ᵢ · v̂ᵢ²
```

**Step 3: Optimal inverse-variance weighting**

The optimal weight for minimum variance is:

```
wᵢ = 1 / var(Tᵢ × v̂ᵢ) = 1 / (σ²ᵀ,ᵢ · v̂ᵢ²)
```

**Step 4: Final estimator**

Forming the weighted combination:

```
Â(r) = Σᵢ [wᵢ · Tᵢ(r) · v̂ᵢ] / Σᵢ [wᵢ]
     = Σᵢ [Tᵢ(r) · v̂ᵢ / (σ²ᵀ,ᵢ · v̂ᵢ²)] / Σᵢ [1 / (σ²ᵀ,ᵢ · v̂ᵢ²)]
     = Σᵢ [Tᵢ(r) / (σ²ᵀ,ᵢ · v̂ᵢ)] / Σᵢ [1 / (σ²ᵀ,ᵢ · v̂ᵢ²)]
```

### Properties

This estimator automatically:
- ✅ **Downweights high CMB noise**: Clusters with large σ²ᵀ,ᵢ get less weight
- ✅ **Downweights small velocities**: Clusters with small |v̂ᵢ| get less weight (appears in numerator and denominator)
- ✅ **Downweights uncertain velocities**: Velocity uncertainty manifests through v̂ᵢ² in the denominator
- ✅ **Minimum variance**: This is the optimal unbiased estimator for the given problem

### Comparison of Estimators

| Feature | Simple | Tanimura | Optimal Posterior |
|---------|--------|----------|-------------------|
| **Accounts for CMB variance** | ❌ No | ✅ Yes | ✅ Yes |
| **Accounts for velocity variance** | ❌ No | ❌ No | ✅ Yes |
| **Downweights small velocities** | ✅ Linear (1/v) | ✅ Linear (1/v) | ✅ Stronger (1/v²) |
| **Use case** | Baseline / uniform noise | Point-estimate velocities | Velocity posteriors |
| **Expected S/N** | Baseline | Good | **Better** |

**Empirical test:** The optimal_posterior estimator should yield a stronger kSZ detection (higher signal in stacked patch) because it's the minimum-variance estimator. The simple estimator serves as a useful baseline for comparison.

---

## Background Subtraction Strategy

The pipeline performs background subtraction at **two stages**, with different purposes:

### Stage 1: Aperture Photometry (Always - tSZ mode only)

**Location:** Individual cluster measurements
**Radii:** 1.5 - 2.5 × R500
**Purpose:** Measure cluster signal relative to local environment
**Mode:** Automatic in tSZ mode

This is the **primary background subtraction** and follows standard practice in tSZ literature.

### Stage 2: Stacking (Optional)

**Location:** Before rescaling patches
**Default:** `subtract_background=False` (disabled)
**Purpose:** Remove large-scale map systematics or gradients

**When enabled (`subtract_background=True`):**
- Radii: Configurable in r/r500 units via `bg_inner_r500` and `bg_outer_r500`
- Default: 3.0 - 5.0 × R500
- Applied at **physical scales** per cluster (ensures consistent zero-points after r/r500 rescaling)

### Recommended Usage

**For standard tSZ analysis:**
```python
subtract_background=False  # Rely on aperture photometry (default)
```

**Only use `subtract_background=True` when:**
- Working with maps containing significant DC offsets
- Large-scale gradients are present
- Need to remove systematic map features

**For kSZ analysis:**
```python
subtract_background=False  # Always - no background subtraction in stacking
```

kSZ relies on velocity weighting to cancel CMB and other isotropic backgrounds.

---

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coord_list` | list | required | Cluster coordinates [[lon, lat, R500, z], ...] |
| `analysis_mode` | str | `'tsz'` | Analysis mode: `'tsz'` or `'ksz'` |
| `patch_size_r500` | float | `10.0` | Patch size in R500 units (spans ±patch_size_r500/2) |
| `npix` | int | `256` | Pixels per side of patch |
| `min_coverage` | float | `0.9` | Minimum mask coverage fraction |

### tSZ-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inner_r500_factor` | float | `1.0` | Inner aperture radius in R500 units |
| `outer_r500_factor` | float | `3.0` | Outer aperture radius in R500 units |
| `run_null_tests` | bool | `True` | Run validation with random pointings |
| `n_bootstrap` | int | `500` | Bootstrap resamples for error estimation |
| `n_random` | int | `500` | Random pointings for null tests |

### kSZ-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | array | `None` | Velocity values (km/s) |
| `weight_vars` | array | `None` | Velocity variances (required for optimal_posterior) |
| `velocity_weighting_scheme` | str | `'tanimura'` | Estimator: `'simple'`, `'tanimura'`, or `'optimal_posterior'` |

### Background Subtraction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subtract_background` | bool | `False` | Enable stacking background subtraction |
| `bg_inner_r500` | float | `3.0` | Background annulus inner radius (R500 units) |
| `bg_outer_r500` | float | `5.0` | Background annulus outer radius (R500 units) |

---

## Output Products

### tSZ Mode Results Dictionary

```python
{
    # Detection statistics
    'mean_delta_y': float,              # Corrected signal estimate
    'error_mean': float,                # Total error (bootstrap)
    'significance': float,              # Detection significance (σ)

    # Significance breakdown
    'significance_metrics': {
        'signal': float,                # Raw signal
        'null_bias': float,             # Null test bias
        'corrected_signal': float,      # Bias-corrected signal
        'simple_significance': float,   # Uncorrected σ
        'null_corrected_significance': float,
        'conservative_significance': float
    },

    # Error decomposition
    'error_decomposition': {
        'sample_variance': float,       # Cluster-to-cluster variance
        'measurement_variance': float,  # Per-cluster measurement uncertainty
        'total_variance': float,        # Combined variance
        'sample_std': float,
        'measurement_std': float,
        'total_std': float
    },

    # Bootstrap results
    'bootstrap_results': {
        'bootstrap_mean': float,
        'bootstrap_samples': array,     # Full distribution
        'confidence_interval_68': tuple,
        'confidence_interval_95': tuple,
        'n_bootstrap': int
    },

    # Null test results
    'null_results': {
        'random_measurements': list,
        'random_mean': float,
        'random_std': float,
        'n_valid_random': int
    },

    # Individual cluster data
    'individual_results': list,         # Per-cluster measurements
    'individual_measurements': list,    # Delta_y values
    'individual_errors': list,          # Measurement errors

    # Stacked products
    'stacked_patch': ndarray,           # 2D stacked map in r/r500
    'stacking_info': dict,              # Stacking metadata

    # Sample statistics
    'n_measurements': int,              # Valid clusters analyzed
    'n_rejected': int,                  # Rejected clusters
    'rejection_stats': dict,            # Rejection reasons
    'r500_median': float,               # Median cluster size

    # Configuration
    'analysis_mode': str,               # 'tsz'
    'patch_size_r500': float,
    'npix': int,
    'inner_r500_factor': float,
    'outer_r500_factor': float
}
```

### kSZ Mode Results Dictionary

```python
{
    # Stacked products (primary output)
    'stacked_patch': ndarray,           # 2D velocity-weighted stack in r/r500
    'stacking_info': {
        'n_stacked': int,
        'weighted': True,
        'velocity_weighting_scheme': str,
        'mean_weight': float,           # Mean velocity
        'std_weight': float,            # Velocity std
        'mean_variance': float          # Mean CMB variance
    },

    # Individual cluster data
    'individual_results': list,         # Basic per-cluster info
    'weights': list,                    # Velocities used
    'weight_vars': list,                # Velocity variances used

    # Sample statistics
    'n_measurements': int,
    'n_rejected': int,
    'rejection_stats': dict,
    'r500_median': float,

    # Configuration
    'analysis_mode': str,               # 'ksz'
    'weighted_mode': True,
    'velocity_weighting_scheme': str,   # Estimator used
    'patch_size_r500': float,
    'npix': int
}
```

---

## Examples

### Example 1: Basic tSZ Detection

```python
from stacking_backend import ClusterAnalysisPipeline
from stacking_backend.config import MapConfig

# Setup
map_config = MapConfig(
    map_path="planck_pr4_y_map.fits",
    mask_path="planck_pr4_masks.fits",
    map_format="planck_pr4"
)
pipeline = ClusterAnalysisPipeline(map_config)

# Cluster sample
clusters = load_mcxc_clusters()  # [lon, lat, R500, z]

# Run analysis
results = pipeline.run_individual_r500_analysis_with_validation(
    coord_list=clusters,
    analysis_mode='tsz'
)

# Print results
print(f"✅ Detection: {results['significance']:.1f}σ")
print(f"Signal: {results['mean_delta_y']:.2e} ± {results['error_mean']:.2e}")
print(f"Sample variance contributes {results['error_decomposition']['sample_variance']/results['error_decomposition']['total_variance']*100:.1f}%")
```

### Example 2: Mass-Binned Scaling Relation

```python
# Bin clusters by mass
mass_bins = [
    (1e14, 2e14),
    (2e14, 5e14),
    (5e14, 1e15)
]

results_by_mass = {}

for m_min, m_max in mass_bins:
    # Select clusters in mass bin
    mask = (masses >= m_min) & (masses < m_max)
    bin_clusters = clusters[mask]

    # Measure tSZ signal
    results = pipeline.run_individual_r500_analysis_with_validation(
        coord_list=bin_clusters,
        analysis_mode='tsz',
        run_null_tests=True
    )

    results_by_mass[(m_min, m_max)] = results

    print(f"Mass bin [{m_min:.1e}, {m_max:.1e}]:")
    print(f"  Signal: {results['mean_delta_y']:.2e} ± {results['error_mean']:.2e}")
    print(f"  Significance: {results['significance']:.1f}σ")

# Derive Y500-M500 scaling relation
# (combine results_by_mass with Y500 integration)
```

### Example 3: kSZ with Velocity Posteriors

```python
# Load velocity reconstruction
velocity_samples = load_velocity_mcmc()  # Shape: (n_clusters, n_mcmc_samples)

# Compute posterior statistics
velocity_means = np.mean(velocity_samples, axis=1)
velocity_vars = np.var(velocity_samples, axis=1)

# Run optimal posterior estimator
results = pipeline.run_individual_r500_analysis_with_validation(
    coord_list=clusters,
    weights=velocity_means,
    weight_vars=velocity_vars,
    velocity_weighting_scheme='optimal_posterior',
    analysis_mode='ksz',
    patch_size_r500=10.0,
    npix=256
)

# Calculate radial profile
from stacking_backend.analysis import RadialProfileCalculator

radii, profile, errors, counts = RadialProfileCalculator.calculate_profile(
    stacked_patch=results['stacked_patch'],
    patch_size_r500=10.0,
    n_radial_bins=20,
    max_radius_r500=5.0
)

# Plot
import matplotlib.pyplot as plt
plt.errorbar(radii, profile, yerr=errors, fmt='o-')
plt.xlabel('r / R500')
plt.ylabel('kSZ Temperature [μK]')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.show()
```

### Example 4: Comparison of Velocity Estimators

```python
# Compare all three velocity weighting schemes

schemes = ['simple', 'tanimura', 'optimal_posterior']
results_comparison = {}

for scheme in schemes:
    # For simple and Tanimura, weight_vars not needed
    kwargs = {'weights': velocity_means}
    if scheme == 'optimal_posterior':
        kwargs['weight_vars'] = velocity_vars

    results = pipeline.run_individual_r500_analysis_with_validation(
        coord_list=clusters,
        velocity_weighting_scheme=scheme,
        analysis_mode='ksz',
        **kwargs
    )

    results_comparison[scheme] = results

    # Measure central pixel amplitude
    center = results['npix'] // 2
    central_amplitude = results['stacked_patch'][center, center]

    print(f"{scheme}:")
    print(f"  Central amplitude: {central_amplitude:.2e} μK")
    print(f"  Stacked {results['n_measurements']} clusters")

# Optimal posterior should show strongest signal
# Tanimura should improve over simple by accounting for CMB variance
```

### Example 5: kSZ Null Tests via Velocity Shuffling

```python
# Proper null test for kSZ: shuffle cluster-velocity pairings

n_shuffles = 100
null_amplitudes = []

for i in range(n_shuffles):
    # Shuffle velocities
    shuffled_velocities = np.random.permutation(velocity_means)
    shuffled_vars = np.random.permutation(velocity_vars)

    # Run pipeline with shuffled velocities
    results = pipeline.run_individual_r500_analysis_with_validation(
        coord_list=clusters,
        weights=shuffled_velocities,
        weight_vars=shuffled_vars,
        velocity_weighting_scheme='optimal_posterior',
        analysis_mode='ksz'
    )

    # Extract signal
    center = results['npix'] // 2
    null_amplitudes.append(results['stacked_patch'][center, center])

null_amplitudes = np.array(null_amplitudes)

# Compare real signal to null distribution
real_amplitude = results_real['stacked_patch'][center, center]
null_mean = np.mean(null_amplitudes)
null_std = np.std(null_amplitudes)

significance = (real_amplitude - null_mean) / null_std
print(f"kSZ Detection: {significance:.1f}σ relative to shuffled null")
```

---

## Scientific Background

### The Sunyaev-Zel'dovich Effects

**Thermal SZ (tSZ):**
- CMB photons inverse-Compton scattered by hot electrons in cluster ICM
- Temperature change: ΔT/T ∝ y = ∫ Pₑ dl (Compton-y parameter)
- Independent of redshift → powerful cosmological probe
- Used for cluster detection, mass calibration, pressure profiles

**Kinematic SZ (kSZ):**
- Doppler shift from cluster bulk motion
- Temperature change: ΔT/T ∝ τₑ × (v_pec / c)
- Proportional to line-of-sight velocity
- Requires velocity information to stack (otherwise cancels)
- Probes large-scale flows, dark energy, modified gravity

### r/r500 Scaling

Clusters span a wide range of sizes (R500 from ~0.1° to ~0.5°). Direct stacking in angular coordinates would blur the signal. Instead, we:

1. **Rescale** each cluster by its characteristic radius R500
2. **Stack** in units of r/r500 (physical coordinates)
3. All clusters aligned at same **physical scale** relative to their size

This preserves the universal profile shape and maximizes signal-to-noise.

### Variance Estimation for Weighting

Following Tanimura et al. (2021), we calculate variance σ²ᵀ,ᵢ over the full patch extent (~10 × θ₅₀₀):

- Pre-filtered maps have large-scale CMB removed (ℓ < 720)
- Residual CMB variance ~(40 μK)² dominates
- kSZ signal ~2-5 μK contributes <2% to variance
- Signal contamination is negligible

This variance provides robust inverse-variance weights for stacking.

### Statistical Framework

**Bootstrap error estimation:**
- Cluster-level resampling (with replacement)
- Properly combines sample variance and measurement variance
- Accounts for cluster-to-cluster scatter

**Null tests (tSZ):**
- Random sky positions with matched masking properties
- Tests for spurious signals in the map itself
- Provides bias correction if needed

**Significance calculation:**
- Corrects for null bias if significant
- Conservative estimates using max(bootstrap error, null std)
- Multiple significance metrics for robustness

---

## Performance Notes

- **Memory:** ~2-4 GB for NSIDE=2048 HEALPix maps
- **Threading:** Patch extraction is thread-safe
- **Typical runtime:**
  - tSZ analysis (100 clusters, 500 bootstrap): ~5-10 minutes
  - kSZ stacking (1000 clusters): ~2-3 minutes
- **Caching:** Map data cached to avoid repeated loading

---

## References

**Key Papers:**

1. **Tanimura et al. (2021)** - "Direct detection of the kinetic Sunyaev-Zel'dovich effect in galaxy clusters"
   *Astronomy & Astrophysics*, 645, A112
   https://doi.org/10.1051/0004-6361/202038846
   *Original velocity-weighted kSZ estimator*

2. **Tanimura et al. (2022)** - "Convolutional neural network-reconstructed velocity for kinetic SZ detection"
   *Astronomy & Astrophysics*, 662, A48
   https://doi.org/10.1051/0004-6361/202243046
   *Improved velocity reconstruction and 4.9σ kSZ detection*

3. **Planck Collaboration (2013)** - "Planck intermediate results. V. Pressure profiles of galaxy clusters"
   *Astronomy & Astrophysics*, 550, A131
   *Universal pressure profile and aperture photometry methods*

**Methodology:**
- r/r500 scaling: Tanimura et al. (2020, 2021)
- CMB filtering: ℓ < 720 cutoff (Tanimura et al. 2021)
- Aperture photometry: Standard practice in tSZ literature
- Null tests with mask-bias correction: This package

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{stacking_backend,
  title = {Stacking Backend: tSZ/kSZ Analysis Pipeline},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/[your-repo]/stacking_backend}
}
```

And cite the key methodological papers:
- Tanimura et al. (2021) for the velocity-weighted kSZ methodology
- Your velocity reconstruction paper (when using optimal_posterior estimator)

---

## License

[Your chosen license]

---

## Contact

For questions, bug reports, or feature requests:
- GitHub Issues: [your-repo]/issues
- Email: [your-email]

---

## Changelog

### Version 2.1 (2025)
- ✅ Added `simple` velocity weighting estimator (velocity-weighted mean without inverse-variance weighting)

### Version 2.0 (2025)
- ✅ Changed `subtract_background` default to `False`
- ✅ Converted background parameters to r/r500 units
- ✅ Added `optimal_posterior` velocity weighting estimator
- ✅ Removed invalid velocity estimators (product, velocity_snr, velocity_snr_direct)
- ✅ Added validation preventing weights with tSZ mode
- ✅ Simplified null tests (unweighted tSZ only)
- ✅ Comprehensive documentation of estimator derivations

### Version 1.0
- Initial release with tSZ and kSZ support
- r/r500 scaling implementation
- Bootstrap error estimation
- Null test framework
