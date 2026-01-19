#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run manticore posterior clustering modes')
    parser.add_argument('mode', choices=['1', '2', '3', '4'], help='Mode to run (1, 2, 3, or 4)')
    parser.add_argument('--config', default='config.toml', help='Path to config file (default: config.toml)')
    parser.add_argument('--output', default='output', help='Output directory (default: output)')
    parser.add_argument('--mpi-np', type=int, default=1, help='Number of MPI processes for mode 2 and 4 (default: 1)')
    parser.add_argument('--algorithm', choices=['hdbscan', 'dbscan'], default=None, help='Clustering algorithm (overrides config file, used in mode 1 and 3)')
    parser.add_argument('--min-samples', type=int, default=None, help='HDBSCAN min_samples parameter (overrides config file)')
    parser.add_argument('--min-cluster-size', type=int, default=None, help='HDBSCAN min_cluster_size parameter (overrides config file, used in mode 1 and 3)')
    parser.add_argument('--eps', type=float, default=None, help='DBSCAN eps parameter in Mpc (overrides config file, used in mode 1 and 3)')

    args = parser.parse_args()

    # Add backend to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.mode == '1':
        algo_str = args.algorithm.upper() if args.algorithm else "config default"
        print(f"Running Mode 1 (Clustering) with config={args.config}, output={args.output}, algorithm={algo_str}")
        if args.algorithm is not None:
            print(f"  Overriding clustering_algorithm={args.algorithm}")
        if args.min_cluster_size is not None:
            print(f"  Overriding min_cluster_size={args.min_cluster_size}")
        if args.min_samples is not None:
            print(f"  Overriding min_samples={args.min_samples}")
        if args.eps is not None:
            print(f"  Overriding eps={args.eps}")
        from backend.mode1_cluster import run_mode1
        run_mode1(config_path=args.config, output_dir=args.output,
                  min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                  algorithm=args.algorithm, eps=args.eps)

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
        algo_str = args.algorithm.upper() if args.algorithm else "config default"
        print(f"Running Mode 3 (Random Control Clustering) with config={args.config}, output={args.output}, algorithm={algo_str}")
        if args.algorithm is not None:
            print(f"  Overriding clustering_algorithm={args.algorithm}")
        if args.min_cluster_size is not None:
            print(f"  Overriding min_cluster_size={args.min_cluster_size}")
        if args.min_samples is not None:
            print(f"  Overriding min_samples={args.min_samples}")
        if args.eps is not None:
            print(f"  Overriding eps={args.eps}")
        from backend.mode3_null import run_mode3
        run_mode3(config_path=args.config, output_dir=args.output,
                  min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                  algorithm=args.algorithm, eps=args.eps)

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
