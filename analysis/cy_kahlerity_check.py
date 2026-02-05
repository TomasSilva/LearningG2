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

# Setup cymetric imports
_parent_dir = PROJECT_ROOT.parent
_cymetric_dir = _parent_dir / "cymetric"

if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_cymetric_dir) not in sys.path:
    sys.path.insert(0, str(_cymetric_dir))

# Create alias to fix cymetric internal imports
try:
    import cymetric
    if hasattr(cymetric, 'cymetric'):
        sys.modules['cymetric'] = cymetric.cymetric
    
    from cymetric.pointgen.pointgen import PointGenerator
    from cymetric.models.helper import prepare_basis
    from cymetric.models.models import MultFSModel
    CYMETRIC_AVAILABLE = True
except ImportError:
    CYMETRIC_AVAILABLE = False
    print("Warning: cymetric package not available. Some functionality may be limited.")


def get_most_recent_cy_run_number(model_dir='./models/cy_models'):
    """Find the most recent run number for CY models.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing CY model files
        
    Returns
    -------
    int or None
        Most recent run number, or None if no runs exist
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return None
    
    # Find all existing model files
    pattern = str(model_dir / "cy_metric_model_run*.keras")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return None
    
    # Extract run numbers from filenames
    run_numbers = []
    for filepath in existing_files:
        filename = Path(filepath).stem
        match = re.search(r'cy_metric_model_run(\d+)', filename)
        if match:
            run_numbers.append(int(match.group(1)))
    
    return max(run_numbers) if run_numbers else None


def load_cy_model_and_data(model_path, data_dir, config_path):
    """
    Load CY metric model and associated data.
    
    Parameters
    ----------
    model_path : Path
        Path to trained CY metric model (.keras file)
    data_dir : Path
        Directory containing dataset.npz and basis.pickle
    config_path : Path
        Path to model configuration YAML file
        
    Returns
    -------
    fmodel : MultFSModel
        Loaded CY metric model
    data : dict
        Loaded dataset
    BASIS : dict
        Prepared basis for the CY manifold
    """
    if not CYMETRIC_AVAILABLE:
        raise ImportError("cymetric package is required but not available")
    
    # Load data
    data = np.load(data_dir / 'dataset.npz')
    BASIS = np.load(data_dir / 'basis.pickle', allow_pickle=True)
    BASIS = prepare_basis(BASIS)
    
    print(f"Loaded data from {data_dir}")
    
    # Load model configuration
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    alpha = config['alpha']
    print(f"Loaded config from {config_path}")
    
    # Load trained model
    loaded_nn = tf.keras.models.load_model(str(model_path))
    fmodel = MultFSModel(loaded_nn, BASIS, alpha=alpha)
    
    print(f"Loaded model from {model_path}")
    
    return fmodel, data, BASIS


def compute_domega_norm(fmodel, BASIS, points, epsilon=1e-12, desc="Computing dω"):
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
    plt.title(f"Kählerity Check: {dataset_name}")
    plt.grid(True, alpha=0.3)
    
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
    else:
        plt.show()
    
    plt.close()


def print_statistics(vals, dataset_name):
    """Print statistics for ||dω|| values."""
    vals_array = np.array(vals)
    print(f"\n{dataset_name} Statistics:")
    print(f"  Mean ||dω||:   {np.mean(vals_array):.6e}")
    print(f"  Median ||dω||: {np.median(vals_array):.6e}")
    print(f"  Std ||dω||:    {np.std(vals_array):.6e}")
    print(f"  Min ||dω||:    {np.min(vals_array):.6e}")
    print(f"  Max ||dω||:    {np.max(vals_array):.6e}")


def main():
    parser = argparse.ArgumentParser(
        description='Check Kählerity of CY metric by computing dω'
    )
    parser.add_argument('--cy-run-number', type=int, default=None,
                       help='CY model run number to use (default: most recent)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained CY metric model (overrides --cy-run-number)')
    parser.add_argument('--data-dir', type=str,
                       default='./samples/cy_data',
                       help='Directory containing dataset.npz and basis.pickle')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model configuration file (overrides --cy-run-number)')
    parser.add_argument('--n-train', type=int, default=100,
                       help='Number of training points to check')
    parser.add_argument('--n-val', type=int, default=100,
                       help='Number of validation points to check')
    parser.add_argument('--epsilon', type=float, default=1e-12,
                       help='Epsilon for numerical derivative')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Directory to save output plots')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent.parent
    
    # Determine CY run number and paths
    if args.model is None and args.config is None:
        # Need to determine run number
        if args.cy_run_number is None:
            # Auto-detect most recent
            cy_run_number = get_most_recent_cy_run_number(script_dir / 'models' / 'cy_models')
            if cy_run_number is None:
                print("Error: No CY models found in ./models/cy_models/")
                print("Please train a CY model first using run_cy.py")
                sys.exit(1)
            print(f"Auto-detected most recent CY model: run {cy_run_number}")
        else:
            cy_run_number = args.cy_run_number
            print(f"Using specified CY model run: {cy_run_number}")
        
        # Construct model and config paths
        model_path = script_dir / f'models/cy_models/cy_metric_model_run{cy_run_number}.keras'
        config_path = script_dir / f'models/cy_models/cy_metric_model_run{cy_run_number}_config.yaml'
    else:
        # Use explicitly provided paths (backward compatibility)
        model_path = (script_dir / args.model).resolve() if args.model else script_dir / 'models/cy_models/cy_metric_model.keras'
        config_path = (script_dir / args.config).resolve() if args.config else script_dir / 'models/cy_models/cy_metric_model_config.yaml'
        
        # Try to extract run number from path
        match = re.search(r'cy_metric_model_run(\d+)', str(model_path))
        if match:
            cy_run_number = int(match.group(1))
            print(f"Detected CY model run number from path: {cy_run_number}")
        else:
            cy_run_number = None
    
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CY Metric Kählerity Check")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Epsilon: {args.epsilon}")
    print()
    
    # Load model and data
    print("Loading model and data...")
    fmodel, data, BASIS = load_cy_model_and_data(model_path, data_dir, config_path)
    print()
    
    # Check training data
    if args.n_train > 0:
        print(f"Checking Kählerity on {args.n_train} training points...")
        n_train = min(args.n_train, len(data['X_train']))
        vals_train = compute_domega_norm(
            fmodel, BASIS, data['X_train'][:n_train], 
            epsilon=args.epsilon, desc="Training set"
        )
        print_statistics(vals_train, "Training Set")
        plot_domega_norms(
            vals_train, "Training Set",
            save_path=output_dir / "cy_kahlerity_train.png"
        )
        print()
    
    # Check validation data
    if args.n_val > 0:
        print(f"Checking Kählerity on {args.n_val} validation points...")
        n_val = min(args.n_val, len(data['X_val']))
        vals_val = compute_domega_norm(
            fmodel, BASIS, data['X_val'][:n_val],
            epsilon=args.epsilon, desc="Validation set"
        )
        print_statistics(vals_val, "Validation Set")
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
