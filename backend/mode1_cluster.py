from backend.mode1a_raw_cluster import run_mode1a
from backend.mode1b_postprocess import run_mode1b

def run_mode1(config_path="config.toml", output_dir="output", eps=None, min_samples=None,
              mass_outlier_threshold=None, use_mass_distance=None):
    """
    Mode 1: Complete clustering pipeline (convenience wrapper)

    Runs Mode 1a (raw DBSCAN) followed by Mode 1b (post-processing corrections)
    in sequence. This is equivalent to running:
        python run_modes.py 1a
        python run_modes.py 1b
    """
    print("=" * 60)
    print("Mode 1: Complete Clustering Pipeline")
    print("=" * 60)
    print("\nThis runs Mode 1a (raw DBSCAN) followed by Mode 1b (post-processing)")
    print()

    # Run Mode 1a
    print("STEP 1: Running Mode 1a (Raw DBSCAN)...")
    print("-" * 60)
    run_mode1a(config_path=config_path, output_dir=output_dir, eps=eps, min_samples=min_samples)

    print("\n" + "=" * 60)
    print()

    # Run Mode 1b
    print("STEP 2: Running Mode 1b (Post-processing)...")
    print("-" * 60)
    run_mode1b(config_path=config_path, output_dir=output_dir,
               input_filename=None,  # Will auto-detect from mode1a output
               mass_outlier_threshold=mass_outlier_threshold,
               use_mass_distance=use_mass_distance)

    print("\n" + "=" * 60)
    print("Mode 1 complete!")
    print("=" * 60)

if __name__ == '__main__':
    run_mode1()
