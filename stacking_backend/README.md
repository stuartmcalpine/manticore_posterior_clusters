# Stacking Backend

A comprehensive Python package for analyzing galaxy clusters through the thermal Sunyaev-Zel'dovich (tSZ) effect using Planck PR4 data.

## Overview

The `stacking_backend` package performs statistical analysis of galaxy clusters by measuring their Compton-y parameter signal in Planck satellite maps. It implements a complete pipeline from data loading through stacking analysis to mass-scaling relations, with robust error estimation and validation.

## Key Features

- 🔭 **tSZ Signal Extraction**: Measure Compton-y parameter around galaxy clusters
- 📚 **Stacking Analysis**: Combine multiple clusters to boost signal-to-noise
- 📊 **Mass Scaling Relations**: Derive Y₅₀₀-M₅₀₀ relationships for cosmology
- ✅ **Validation Framework**: Null tests and bootstrap error estimation
- 🎨 **Visualization Tools**: Comprehensive plotting for patches, profiles, and scaling
- 🔧 **Configurable Pipeline**: Flexible parameters for different analysis scenarios

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/stacking_backend.git
cd stacking_backend

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `healpy`
- `astropy`
- `h5py` (for catalog reading)

## Quick Start

```python
from stacking_backend import ClusterAnalysisPipeline
from stacking_backend.config import DataPaths

# Initialize pipeline with data paths
data_paths = DataPaths(
    pr4_y_map="path/to/PR4_NILC_y_map.fits",
    pr4_masks="path/to/Masks.fits"
)
pipeline = ClusterAnalysisPipeline(data_paths)

# Load cluster coordinates (lon, lat, R500, redshift)
coord_list = [
    [120.5, 45.2, 0.15, 0.05],
    [230.1, -30.4, 0.18, 0.08],
    # ... more clusters
]

# Run analysis with validation
results = pipeline.run_individual_r200_analysis_with_validation(
    coord_list=coord_list,
    inner_r200_factor=1.0,  # Inner aperture at 1×R₅₀₀
    outer_r200_factor=3.0,  # Outer aperture at 3×R₅₀₀
    patch_size_deg=15.0,     # Patch size in degrees
    npix=256,                # Pixels per patch
    n_bootstrap=500,         # Bootstrap samples
    run_null_tests=True      # Enable validation
)

print(f"Detection significance: {results['significance']:.1f}σ")
print(f"Mean Δy: {results['mean_delta_y']:.2e}")
```

## Core Components

### 1. Data Module (`stacking_backend/data/`)

Handles Planck data loading and patch extraction:

- **`pr4_loader.py`**: Loads PR4 NILC y-maps and masks (galactic, point source, NILC)
- **`patch_extractor.py`**: Thread-safe extraction of patches from HEALPix maps
- **`coordinate_utils.py`**: Coordinate transformations (RA/Dec ↔ Galactic ↔ HEALPix)

### 2. Analysis Module (`stacking_backend/analysis/`)

Core analysis algorithms:

- **`pipeline.py`**: Main analysis orchestrator with validation
- **`photometry.py`**: Aperture photometry and Y₅₀₀ integration
- **`stacking.py`**: Patch stacking with background subtraction
- **`profiles.py`**: Radial profile calculation
- **`individual_clusters.py`**: Per-cluster measurements with individual R₅₀₀
- **`validation.py`**: Null tests with mask-bias correction

### 3. Plotting Module (`stacking_backend/plotting/`)

Visualization tools:

- **`basic_plots.py`**: Individual patches and profiles
- **`summary_plots.py`**: Multi-panel analysis summaries
- **`mass_scaling.py`**: Y₅₀₀-M₅₀₀ scaling relations

### 4. Configuration Module (`stacking_backend/config/`)

- **`analysis_params.py`**: Analysis parameter presets
- **`paths.py`**: Data path management with environment variables

## Analysis Pipeline

### Step 1: Physical Coordinate Conversion
Converts observed coordinates to physical frame with E(z) corrections for proper scaling.

### Step 2: Individual Cluster Measurements
```python
# Each cluster measured with its own R₅₀₀
- Extract patch around cluster
- Apply masks (galactic, point sources)
- Calculate aperture photometry (inner vs outer annulus)
- Measure Y₅₀₀ integration
- Track quality metrics
```

### Step 3: Bootstrap Error Estimation
Performs cluster-level resampling (N=500 default) to estimate sample variance and derive robust error bars.

### Step 4: Patch Stacking
```python
# Stack valid patches for improved S/N
- Background subtraction using outer annuli
- Weighted mean combination
- Track contributing pixels per location
```

### Step 5: Null Test Validation
Compares cluster signals to random sky positions with matched masking properties to validate detection significance.

### Step 6: Radial Profile Generation
Computes azimuthally averaged profiles in radial bins from stacked patch.

## Output Products

The pipeline returns a comprehensive results dictionary:

```python
{
    'success': bool,                    # Analysis success flag
    'mean_delta_y': float,              # Mean y-parameter difference
    'error_mean': float,                # Bootstrap error estimate
    'significance': float,              # Detection significance (σ)
    'bootstrap_results': dict,          # Full bootstrap statistics
    'null_results': dict,               # Null test validation
    'individual_results': list,         # Per-cluster measurements
    'stacked_patch': np.array,          # 2D stacked y-map
    'profile_radii': np.array,          # Radial bin centers
    'profile_mean': np.array,           # Radial profile values
    'r200_median': float,               # Median R₂₀₀ of sample
    'n_measurements': int,              # Valid clusters analyzed
    'rejection_stats': dict,            # Rejection reason counts
}
```

## Configuration Options

### Analysis Parameters

```python
from stacking_backend.config import AnalysisParameters

# Default configuration
params = AnalysisParameters()

# Quick test configuration
params = AnalysisParameters.for_quick_test()

# Mass scaling optimized
params = AnalysisParameters.for_mass_scaling()

# Custom configuration
params = AnalysisParameters(
    patch_size_deg=20.0,
    npix=512,
    inner_r200_factor=0.5,
    outer_r200_factor=2.0,
    n_bootstrap=1000,
    n_random_pointings=1000,
    min_coverage=0.8
)
```

### Environment Variables

Configure data paths via environment:

```bash
export PR4_Y_MAP_PATH="/path/to/PR4_NILC_y_map.fits"
export PR4_MASKS_PATH="/path/to/Masks.fits"
export MCXC_CATALOG_PATH="/path/to/mcxc_clusters.hdf5"
```

## Visualization Examples

### Plot Individual Patch
```python
from stacking_backend.plotting import BasicPlotter

plotter = BasicPlotter(pipeline.patch_extractor)
fig, axes, patch, mask = plotter.plot_patch(
    ra=150.0, dec=2.5,
    patch_size_deg=10.0,
    npix=256
)
```

### Analysis Summary
```python
from stacking_backend.plotting import SummaryPlotter

fig, axes = SummaryPlotter.plot_analysis_summary(
    results,
    title="Cluster Analysis Results"
)
```

### Mass Scaling Relations
```python
from stacking_backend.plotting import MassScalingPlotter

# Results from multiple mass bins
fig, ax, fit_results = MassScalingPlotter.plot_y_mass_scaling(
    results_dict,
    show_theory=True,
    fit_scaling=True
)
```

## Scientific Applications

### 1. Cosmological Parameter Constraints
Use Y₅₀₀-M₅₀₀ scaling relations to calibrate cluster mass proxies for cosmology.

### 2. Cluster Physics Studies
Analyze radial profiles to study:
- Gas pressure profiles
- ICM thermodynamics
- AGN feedback effects

### 3. Survey Cross-Validation
Compare tSZ detections with:
- X-ray catalogs (eROSITA, XMM)
- Optical cluster finders
- Weak lensing masses

### 4. Systematic Error Assessment
- Null tests validate measurement pipeline
- Bootstrap quantifies sample variance
- Mask-bias correction ensures unbiased random samples

## Data Requirements

### Input Data Format
- **Planck Maps**: HEALPix format FITS files (NSIDE=2048)
- **Coordinates**: List of [lon, lat, R₅₀₀, redshift]
  - lon/lat in Galactic degrees
  - R₅₀₀ in degrees
  - redshift (optional but recommended)

### Catalog Support
Built-in readers for:
- MCXC (Meta-Catalog of X-ray Clusters)
- eROSITA cluster catalogs
- Custom HDF5/FITS catalogs

## Performance Considerations

- **Memory**: ~2-4 GB for NSIDE=2048 maps
- **Threading**: Thread-safe patch extraction
- **Caching**: Automatic data caching to avoid reloading
- **Optimization**: Vectorized operations with NumPy

## Validation Framework

The package includes comprehensive validation:

1. **Input Validation**: Checks coordinate ranges, parameter bounds
2. **Quality Control**: Tracks rejection reasons, coverage statistics  
3. **Null Tests**: Random sky positions with matched masking
4. **Bootstrap Errors**: Cluster-level resampling for robust uncertainties
5. **Significance Testing**: Proper statistical significance calculation
