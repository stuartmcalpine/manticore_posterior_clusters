#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run manticore posterior clustering modes')
    parser.add_argument('mode', choices=['1a', '1b', '2', '3', '4'], help='Mode to run (1a, 1b, 2, 3, or 4)')
    parser.add_argument('--config', default='config.toml', help='Path to config file (default: config.toml)')
    parser.add_argument('--output', default='output', help='Output directory (default: output)')
    parser.add_argument('--mpi-np', type=int, default=1, help='Number of MPI processes for mode 2 and 4 (default: 1)')
    parser.add_argument('--eps', type=float, default=None, help='DBSCAN eps parameter (overrides config file, used in mode 3)')
    parser.add_argument('--min-samples', type=int, default=None, help='DBSCAN/HDBSCAN min_samples parameter (overrides config file)')
    parser.add_argument('--min-cluster-size', type=int, default=None, help='HDBSCAN min_cluster_size parameter (overrides config file, used in mode 1a)')
    parser.add_argument('--input-filename', type=str, default=None, help='Input filename for mode 1b (raw DBSCAN output from mode 1a)')
    parser.add_argument('--mass-outlier-threshold', type=float, default=None, help='Mass outlier threshold for mode 1b (overrides config file)')

    args = parser.parse_args()
    
    # Add backend to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    if args.mode == '1a':
        print(f"Running Mode 1a (HDBSCAN Clustering) with config={args.config}, output={args.output}")
        if args.min_cluster_size is not None:
            print(f"  Overriding min_cluster_size={args.min_cluster_size}")
        if args.min_samples is not None:
            print(f"  Overriding min_samples={args.min_samples}")
        from backend.mode1a_raw_cluster import run_mode1a
        run_mode1a(config_path=args.config, output_dir=args.output,
                   min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)

    elif args.mode == '1b':
        print(f"Running Mode 1b (Post-processing) with config={args.config}, output={args.output}")
        if args.input_filename is not None:
            print(f"  Using input file: {args.input_filename}")
        if args.mass_outlier_threshold is not None:
            print(f"  Overriding mass_outlier_threshold={args.mass_outlier_threshold}")
        from backend.mode1b_postprocess import run_mode1b
        run_mode1b(config_path=args.config, output_dir=args.output,
                   input_filename=args.input_filename, mass_outlier_threshold=args.mass_outlier_threshold)

    elif args.mode == '2':
        print(f"Running Mode 2 with config={args.config}, output={args.output}, mpi_np={args.mpi_np}")
        
        if args.mpi_np == 1:
            # Run without MPI
            from backend.mode2_trace import run_mode2
            run_mode2(config_path=args.config, output_dir=args.output)
        else:
            # Run with MPI
            cmd = [
                'mpirun', '-np', str(args.mpi_np), 
                'python', '-c',
                f"import sys; sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}'); "
                f"from backend.mode2_trace import run_mode2; "
                f"run_mode2(config_path='{args.config}', output_dir='{args.output}')"
            ]
            result = subprocess.run(cmd)
            sys.exit(result.returncode)
            
    elif args.mode == '3':
        print(f"Running Mode 3 with config={args.config}, output={args.output}")
        if args.eps is not None:
            print(f"  Overriding eps={args.eps}")
        if args.min_samples is not None:
            print(f"  Overriding min_samples={args.min_samples}")
        from backend.mode3_null import run_mode3
        run_mode3(config_path=args.config, output_dir=args.output, eps=args.eps, min_samples=args.min_samples)
        
    elif args.mode == '4':
        print(f"Running Mode 4 with config={args.config}, output={args.output}, mpi_np={args.mpi_np}")
        
        if args.mpi_np == 1:
            # Run without MPI
            from backend.mode4_control_trace import run_mode4
            run_mode4(config_path=args.config, output_dir=args.output)
        else:
            # Run with MPI
            cmd = [
                'mpirun', '-np', str(args.mpi_np), 
                'python', '-c',
                f"import sys; sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}'); "
                f"from backend.mode4_control_trace import run_mode4; "
                f"run_mode4(config_path='{args.config}', output_dir='{args.output}')"
            ]
            result = subprocess.run(cmd)
            sys.exit(result.returncode)

if __name__ == '__main__':
    main()
