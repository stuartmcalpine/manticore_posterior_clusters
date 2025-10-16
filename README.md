# Manticore Posterior Clustering Pipeline

A scientific computing pipeline for analyzing dark matter halo clustering in cosmological simulations. This tool identifies stable halo associations across multiple MCMC simulation samples and traces their evolution through cosmic time.

## Overview

The pipeline analyzes constrained cosmological simulations to identify robust halo associations that persist across multiple MCMC realizations. It then traces these associations backwards in time and compares them with control simulations to quantify the information gain from observational constraints.

## Modes of Operation

### Mode 1: Halo Clustering
Identifies stable dark matter halo clusters across multiple MCMC simulation samples using DBSCAN clustering.

**What it does:**
- Loads halo catalogs from multiple MCMC samples at z=0
- Applies mass and radius cuts
- Performs DBSCAN clustering to find stable associations
- Computes cluster statistics (mass, position, shape parameters)
- Saves results to HDF5

### Mode 2: Cluster Tracing (MPI-enabled)
Traces identified clusters backwards through time using merger tree data.

**What it does:**
- Loads clusters from Mode 1
- Traces each cluster member back to earlier snapshots
- Tracks halo properties through time
- Supports parallel processing via MPI
- Saves full trajectory data to HDF5

### Mode 3: Control Analysis
Analyzes control simulations with random observer positions to establish statistical baselines.

**What it does:**
- Loads control simulations
- Samples multiple random observer positions
- Performs clustering analysis similar to Mode 1
- Provides null hypothesis comparison data

### Mode 4: Control Tracing (MPI-enabled)
Traces control simulation halos backwards in time for comparison with constrained simulations.

**What it does:**
- Loads control simulation halos
- Traces them back through time like Mode 2
- Provides comparison dataset for statistical analysis
- Supports parallel processing via MPI

## Requirements

```bash
# Core dependencies
numpy
scipy
scikit-learn
h5py
toml
mpi4py  # For parallel modes (2 and 4)

# Project-specific
pymanticore  # Custom package for SWIFT simulation analysis
```

## Usage

### Basic Commands

```bash
# Mode 1: Identify clusters (single process)
python run_modes.py 1

# Mode 2: Trace clusters (parallel with 16 processes)
python run_modes.py 2 --mpi-np 16

# Mode 3: Analyze control simulations (single process)
python run_modes.py 3

# Mode 4: Trace control halos (parallel with 16 processes)
python run_modes.py 4 --mpi-np 16
```

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `mode` | Operation mode (1, 2, 3, or 4) | Required |
| `--config` | Path to configuration file | `config.toml` |
| `--output` | Output directory for results | `output` |
| `--mpi-np` | Number of MPI processes (modes 2 & 4) | 1 |

### Examples

```bash
# Use custom config and output directory
python run_modes.py 1 --config my_config.toml --output results/

# Run Mode 2 with 32 MPI processes
python run_modes.py 2 --mpi-np 32 --output results/

# Sequential run of all modes
python run_modes.py 1
python run_modes.py 2 --mpi-np 16
python run_modes.py 3
python run_modes.py 4 --mpi-np 16
```

## Configuration

Edit `config.toml` to customize the analysis:

```toml
[global]
basedir = "/path/to/constrained/simulations"
observer_coords = [500.0, 500.0, 500.0]  # Mpc
output_dir = "output"
boxsize = 1000.0  # Mpc

[mode1]
mcmc_start = 0          # First MCMC sample
mcmc_end = 79           # Last MCMC sample
m200_mass_cut = 0.5e14  # Minimum halo mass (M_sun)
radius_cut = 400.0      # Maximum distance from observer (Mpc)
eps = 1.75              # DBSCAN epsilon parameter (Mpc)
min_samples = 9         # DBSCAN minimum cluster size

[mode2]
target_snapshot = 10    # Earliest snapshot to trace back to
min_cluster_size = 40   # Minimum cluster size to trace

[mode3]
basedir = "/path/to/control/simulations"
mcmc_start = 0
mcmc_end = 9
num_samplings = 8       # Random observer samples per simulation
# ... other parameters

[mode4]
basedir = "/path/to/control/simulations"
# ... control simulation parameters
```

## Typical Workflow

1. **Identify Clusters**
   ```bash
   python run_modes.py 1
   ```
   Output: `output/clusters_eps_1p75_min_samples_9.h5`

2. **Trace Cluster Evolution**
   ```bash
   python run_modes.py 2 --mpi-np 16
   ```
   Output: `output/halo_traces_eps_1p75_min_samples_9.h5`

3. **Generate Control Comparison** (optional)
   ```bash
   python run_modes.py 3
   python run_modes.py 4 --mpi-np 16
   ```
   Output: `output/random_control_clusters.h5`, `output/control_halo_traces_*.h5`

## Output Files

All outputs are HDF5 files containing:

- **Clusters file** (Mode 1): Cluster membership, statistics, halo properties
- **Traces file** (Mode 2): Time evolution of cluster members
- **Control files** (Modes 3 & 4): Comparison data from unconstrained simulations

### Key Metrics Computed

- Lagrangian volumes (convex hull of initial positions)
- Covariance-based scatter ratios
- Information gain (bits) between constrained and control
- Mass distributions and dispersions
- Shape parameters (axis ratios, asphericity, prolateness)

## Parallel Execution

Modes 2 and 4 support MPI parallelization:

```bash
# Using mpirun directly
mpirun -np 32 python -c "from backend.mode2_trace import run_mode2; run_mode2()"

# Or using the convenience script
python run_modes.py 2 --mpi-np 32
```

The work is automatically distributed across MPI ranks by MCMC sample ID.

## Analysis Example

After running all modes, you can analyze the results:

```python
from backend.io import load_clusters_from_hdf5, load_single_cluster_traces
from backend.analysis import analyze_volume_ratios_batch

# Load clusters
clusters, metadata = load_clusters_from_hdf5("output", "clusters_eps_1p75_min_samples_9.h5")

# Analyze volume ratios
masses, volume_ratios, s_ratios, info_bits, cluster_ids, distances = \
    analyze_volume_ratios_batch(clusters, config, trace_filename)

# Results show information gain from observational constraints
print(f"Mean information gain: {info_bits.mean():.2f} bits")
```

## Notes

- Mode 1 must be run before Mode 2 (Mode 2 requires Mode 1 output)
- Modes 3 and 4 are optional but provide important statistical comparisons
- Large simulations benefit significantly from MPI parallelization in Modes 2 and 4
- Typical runtime: Mode 1 (~minutes), Mode 2 (~hours with MPI), Mode 3 (~minutes), Mode 4 (~hours with MPI)

## Citation

If you use this pipeline in your research, please cite the Manticore project papers.

## License

[Specify your license here]
