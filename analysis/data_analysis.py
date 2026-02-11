#!/usr/bin/env python3
"""
Data analysis script for G2 dataset.
Generates statistical visualizations and saves them to plots/ directory.
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
RUNS_DIR = PROJECT_ROOT / "models" / "link_models"
DATA_PATH = PROJECT_ROOT / "samples" / "link_data" / "g2_test.npz"

# Import compression functions
sys.path.insert(0, str(PROJECT_ROOT))
from geometry.compression import form_to_vec, metric_to_vec, vec_to_metric

# Ensure plots directory exists
PLOTS_DIR.mkdir(exist_ok=True)


def find_most_recent_run():
    """
    Find the most recent run number that exists for both 3form and metric.
    
    Returns
    -------
    int or None
        Run number if found, None otherwise
    """
    if not RUNS_DIR.exists():
        return None
    
    # Find all 3form and metric run files
    form_runs = set()
    metric_runs = set()
    
    for file in RUNS_DIR.glob("3form_run*.keras"):
        try:
            run_num = int(file.stem.replace("3form_run", ""))
            form_runs.add(run_num)
        except ValueError:
            continue
    
    for file in RUNS_DIR.glob("metric_run*.keras"):
        try:
            run_num = int(file.stem.replace("metric_run", ""))
            metric_runs.add(run_num)
        except ValueError:
            continue
    
    # Find common runs
    common_runs = form_runs & metric_runs
    
    if not common_runs:
        return None
    
    return max(common_runs)


def plot_volume_comparison(data, save_path):
    """Plot CY volume vs G2 volume."""
    # Riemannian metrics are stored as 21 Cholesky components (6x6 matrix)
    # G2 metrics are stored as 28 Cholesky components (7x7 matrix)
    
    # Reconstruct full matrices if needed
    riem_metrics = data["riemannian_metrics"]
    g2_metrics_data = data["g2_metrics"]
    
    if riem_metrics.ndim == 2 and riem_metrics.shape[1] == 21:
        # Reconstruct 6x6 symmetric matrices from 21 Cholesky components
        riem_matrices = vec_to_metric(riem_metrics)
        cy_vol = np.sqrt(np.linalg.det(riem_matrices))
    else:
        cy_vol = np.sqrt(np.linalg.det(riem_metrics))
    
    if g2_metrics_data.ndim == 2 and g2_metrics_data.shape[1] == 28:
        # Reconstruct 7x7 symmetric matrices from 28 Cholesky components
        g2_matrices = vec_to_metric(g2_metrics_data)
        g2_vol = np.sqrt(np.linalg.det(g2_matrices))
    else:
        g2_vol = np.sqrt(np.linalg.det(g2_metrics_data))
    
    # Calculate Pearson correlation coefficient
    pmcc = np.corrcoef(cy_vol, g2_vol)[0, 1]
    
    plt.figure(figsize=(8, 6))
    plt.plot(cy_vol, g2_vol, 'o', markersize=2, label=f'PMCC = {pmcc:.4f}')
    plt.xlabel("Vol CY")
    plt.ylabel("Vol " + r"$G_2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved volume comparison plot to {save_path.name}")


def plot_3form_histograms(Y, save_path, bins=100, logy=False):
    """
    Plot histograms for all 35 components of the 3-form.
    
    Parameters
    ----------
    Y : ndarray, shape (N, 35)
        3-form components
    save_path : Path
        Where to save the plot
    bins : int
        Number of histogram bins
    logy : bool
        Whether to use log scale for y-axis
    """
    assert Y.shape[1] == 35

    nrows, ncols = 5, 7  # 5*7 = 35
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12))
    axes = axes.flatten()
    indices = list(itertools.combinations(range(7), 3))
    
    for k in range(35):
        ax = axes[k]
        ax.hist(Y[:, k], bins=bins, alpha=0.7)
        ax.set_title(r"$\varphi$" + f"[{indices[k][0]+1},{indices[k][1]+1},{indices[k][2]+1}]", 
                     fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 3-form histogram plot to {save_path.name}")


def plot_metric_histograms(Y, save_path, bins=100, logy=False):
    """
    Plot histograms for all 28 components of the G2 metric.
    
    Parameters
    ----------
    Y : ndarray, shape (N, 28)
        Metric components
    save_path : Path
        Where to save the plot
    bins : int
        Number of histogram bins
    logy : bool
        Whether to use log scale for y-axis
    """
    assert Y.shape[1] == 28

    nrows, ncols = 4, 7  # 4*7 = 28
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12))
    axes = axes.flatten()
    indices = [(i, j) for i in range(1, 8) for j in range(i, 8)]

    for k in range(28):
        ax = axes[k]
        ax.hist(Y[:, k], bins=bins, alpha=0.7)
        ax.set_title(r"$g_{\varphi}$" + f"[{indices[k][0]},{indices[k][1]}]", 
                     fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metric histogram plot to {save_path.name}")


def print_data_statistics(data):
    """Print basic statistics about the dataset."""
    print("Dataset Statistics:")
    print("-" * 80)
    print(f"Number of samples: {len(data['base_points'])}")
    print()
    print("Data arrays:")
    for key in sorted(data.keys()):
        print(f"  {key:25s} shape: {data[key].shape}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze G2 dataset and generate plots.')
    parser.add_argument('--run_number', type=int, default=None,
                        help='Run number to include in plot filenames (default: most recent run with both 3form and metric)')
    parser.add_argument('--n-points', type=int, default=None,
                        help='Number of points to randomly sample from test set (default: use all points)')
    args = parser.parse_args()
    
    # Determine run number
    run_number = args.run_number
    if run_number is None:
        run_number = find_most_recent_run()
        if run_number is None:
            print("ERROR: No common run number found for both 3form and metric models.")
            print(f"Please train models using run_g2.py or specify --run_number explicitly.")
            sys.exit(1)
        print(f"Using most recent run number: {run_number}")
    else:
        print(f"Using specified run number: {run_number}")
    
    print("=" * 80)
    print("G2 Dataset Analysis")
    print("=" * 80)
    print(f"Data file: {DATA_PATH}")
    print(f"Output directory: {PLOTS_DIR}")
    print(f"Run number: {run_number}")
    print()
    
    # Load data
    print("Loading data...")
    data_raw = np.load(DATA_PATH)
    
    # Random sampling if requested
    total_points = len(data_raw['base_points'])
    if args.n_points is not None and args.n_points < total_points:
        print(f"Randomly sampling {args.n_points} points from {total_points} total points...")
        indices = np.random.choice(total_points, size=args.n_points, replace=False)
        
        # Create new dict with sampled data
        data = {}
        for key in data_raw.files:
            # Only sample arrays that have the same number of samples
            # (skip metadata like cy_run_number)
            if len(data_raw[key]) == total_points:
                data[key] = data_raw[key][indices]
            else:
                data[key] = data_raw[key]
        print(f"Sampled {args.n_points} points for analysis")
    else:
        if args.n_points is not None:
            print(f"Requested {args.n_points} points, but dataset only has {total_points}. Using all points.")
        # Convert to dict for consistent access pattern
        data = {key: data_raw[key] for key in data_raw.files}
    
    print_data_statistics(data)
    
    # 1. Volume comparison plot
    print("Generating volume comparison plot...")
    volume_plot_path = PLOTS_DIR / f"volume_comparison_run{run_number}.png"
    plot_volume_comparison(data, volume_plot_path)
    print()
    
    # 2. 3-form component histograms
    print("Generating 3-form component histograms...")
    # Check if phis are in full tensor form or component form
    phis = data["phis"]
    if phis.ndim == 2 and phis.shape[1] == 35:
        # Already in component form
        Y_3form = phis
    else:
        # Full tensor form - extract components
        Y_3form = form_to_vec(phis)
    
    threeform_plot_path = PLOTS_DIR / f"3form_histograms_run{run_number}.png"
    plot_3form_histograms(Y_3form, threeform_plot_path)
    print()
    
    # 3. Metric component histograms
    print("Generating metric component histograms...")
    # Check if metrics are in full matrix form or component form
    g2_metrics = data["g2_metrics"]
    if g2_metrics.ndim == 2 and g2_metrics.shape[1] == 28:
        # Already in component form
        Y_metric = g2_metrics
    else:
        # Full matrix form - extract components
        Y_metric = metric_to_vec(g2_metrics)
    
    metric_plot_path = PLOTS_DIR / f"metric_histograms_run{run_number}.png"
    plot_metric_histograms(Y_metric, metric_plot_path)
    print()
    
    # Print component statistics
    print("Component Statistics:")
    print("-" * 80)
    print("3-form components:")
    print(f"  Min:    {Y_3form.min():.6f}")
    print(f"  Max:    {Y_3form.max():.6f}")
    print(f"  Mean:   {Y_3form.mean():.6f}")
    print(f"  Std:    {Y_3form.std():.6f}")
    print()
    print("Metric components:")
    print(f"  Min:    {Y_metric.min():.6f}")
    print(f"  Max:    {Y_metric.max():.6f}")
    print(f"  Mean:   {Y_metric.mean():.6f}")
    print(f"  Std:    {Y_metric.std():.6f}")
    print()
    
    print("=" * 80)
    print("Analysis complete!")
    print(f"All plots saved to: {PLOTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
