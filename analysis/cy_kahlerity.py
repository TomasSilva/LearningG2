#!/usr/bin/env python3
"""
Check Kählerity of CY metric output.

Uses the numerical exterior derivative algorithm from https://arxiv.org/abs/2510.00999
to verify that dω = 0 for the Kähler form learned by the CY metric model.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import glob
import re
import yaml
import pickle

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import geometry functions
from geometry.geometry import kahler_form_real_matrix, find_max_dQ_coords
from geometry.numerical_exterior_derivative import (
    sample_numerical_kform_neighborhood_val,
    numerical_d,
    quintic_solver
)

# Import analysis utilities
from analysis.utils import (
    get_most_recent_cy_run_number,
    load_cy_model,
    print_statistics
)


def compute_domega_norm(fmodel, BASIS, points, epsilon=1e-5, desc="Computing dω"):
    """
    Compute ||dω|| for a set of points.
    
    Parameters
    ----------
    fmodel : MultFSModel
        CY metric model
    BASIS : dict
        Basis for the CY manifold
    points : ndarray
        Array of points to evaluate
    epsilon : float
        Epsilon for numerical derivative
    desc : str
        Description for progress bar
        
    Returns
    -------
    list
        List of ||dω|| values for each point
    """
    # Define sampler function
    def sampler_2form(p):
        """Sample Kähler form at point p."""
        metric = np.array(fmodel(p)[0])
        return kahler_form_real_matrix(metric)
    
    # Define functions that use BASIS
    def find_max_dQ_with_basis(point_cc):
        """Wrapper to use find_max_dQ_coords with BASIS."""
        return find_max_dQ_coords(point_cc, BASIS)
    
    vals = []
    for i in tqdm(range(len(points)), desc=desc):
        point = points[i]
        
        # Sample neighborhood
        dic2_form = sample_numerical_kform_neighborhood_val(
            sampler_2form, 
            point, 
            epsilon=epsilon,
            find_max_dQ_coords_fn=find_max_dQ_with_basis,
            quintic_solver_fn=quintic_solver
        )
        
        # Compute numerical exterior derivative
        numerical_exterior_d = numerical_d(dic2_form, epsilon=epsilon)
        
        # Compute norm
        vals.append(np.linalg.norm(numerical_exterior_d))
    
    return vals


def plot_domega_norms(vals, dataset_name, save_path=None):
    """
    Plot ||dω|| values.
    
    Parameters
    ----------
    vals : list
        List of ||dω|| norms
    dataset_name : str
        Name of dataset (for title)
    save_path : Path, optional
        If provided, save plot to this path
    """
    plt.figure(figsize=(10, 6))
    plt.plot(vals, 'o-', markersize=3, alpha=0.7)
    plt.xlabel("Data Index")
    plt.ylabel(r"$\|\mathsf{d}\omega\|$")
    
    # Add statistics
    vals_array = np.array(vals)
    plt.axhline(np.mean(vals_array), color='r', linestyle='--', 
                label=f'Mean: {np.mean(vals_array):.2e}')
    plt.axhline(np.median(vals_array), color='g', linestyle='--', 
                label=f'Median: {np.median(vals_array):.2e}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Check Kählerity of CY metric by computing dω'
    )
    parser.add_argument('--cy-run-number', type=int, default=None,
                       help='CY model run number to use (default: most recent)')
    parser.add_argument('--data-dir', type=str,
                       default='./samples/cy_data',
                       help='Directory containing dataset.npz and basis.pickle')
    parser.add_argument('--n-train', type=int, default=0,
                       help='Number of training points to check')
    parser.add_argument('--n-val', type=int, default=None,
                       help='Number of validation points to check (default: all)')
    parser.add_argument('--epsilon', type=float, default=1e-5,
                       help='Epsilon for numerical derivative')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Directory to save output plots')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine CY run number
    if args.cy_run_number is None:
        cy_run_number = get_most_recent_cy_run_number(script_dir / 'models' / 'cy_models')
        if cy_run_number is None:
            print("Error: No CY models found in ./models/cy_models/")
            print("Please train a CY model first using run_cy.py")
            sys.exit(1)
        print(f"Auto-detected most recent CY model: run {cy_run_number}")
    else:
        cy_run_number = args.cy_run_number
        print(f"Using specified CY model run: {cy_run_number}")
    
    print("=" * 80)
    print("CY Metric Kählerity Check")
    print("=" * 80)
    print(f"CY run number: {cy_run_number}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epsilon: {args.epsilon}")
    print()
    
    # Load model and data
    print("Loading model and data...")
    fmodel, BASIS, data = load_cy_model(cy_run_number, args.data_dir, script_dir)
    print()
    
    # Check training data
    if args.n_train > 0:
        print(f"Checking Kählerity on {args.n_train} training points...")
        n_train = min(args.n_train, len(data['X_train']))
        vals_train = compute_domega_norm(
            fmodel, BASIS, data['X_train'][:n_train], 
            epsilon=args.epsilon, desc="Training set"
        )
        print_statistics("Training Set", vals_train)
        plot_domega_norms(
            vals_train, "Training Set",
            save_path=output_dir / "cy_kahlerity_train.png"
        )
        print()
    
    # Check validation data
    if args.n_val is None or args.n_val > 0:
        n_val = len(data['X_val']) if args.n_val is None else min(args.n_val, len(data['X_val']))
        print(f"Checking Kählerity on {n_val} validation points...")
        vals_val = compute_domega_norm(
            fmodel, BASIS, data['X_val'][:n_val],
            epsilon=args.epsilon, desc="Validation set"
        )
        print_statistics("Validation Set", vals_val)
        plot_domega_norms(
            vals_val, "Validation Set",
            save_path=output_dir / "cy_kahlerity_val.png"
        )
        print()
    
    print("=" * 80)
    print("Kählerity check complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
