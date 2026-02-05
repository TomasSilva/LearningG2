#!/usr/bin/env python3
"""
Sampling the G2 structures on the Link manifold.

This script generates G2 structure data by:
1. Loading a trained CY metric model
2. Sampling points from the CY space
3. Computing the induced G2 structures on the link
4. Saving the data for ML training
"""

import sys
import pathlib
import numpy as np
import os
import yaml
import pickle
import itertools
import glob
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import tensorflow as tf

# Setup paths
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import geometry compression functions
from geometry.compression import metric_to_vec

# Setup path for cymetric package
_parent_dir = SCRIPT_DIR.parent
_cymetric_dir = _parent_dir / "cymetric"

if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_cymetric_dir) not in sys.path:
    sys.path.insert(0, str(_cymetric_dir))

# Create alias to fix cymetric internal imports
import cymetric
if hasattr(cymetric, 'cymetric'):
    sys.modules['cymetric'] = cymetric.cymetric

# Import functions
from geometry.geometry import (
    kahler_form_real_matrix,
    holomorphic_volume_real_imag,
    riemannian_metric_real_matrix,
    compute_gG2,
    find_max_dQ_coords
)
from geometry.wedge import wedge

# Import cymetric functions
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.helper import prepare_basis
from cymetric.models.models import MultFSModel

# Import utilities
from analysis.utils import get_most_recent_cy_run_number


def oriented_3form_components(T):
    """
    Extract the 35 oriented components of a 3-form tensor.
    
    Parameters
    ----------
    T : ndarray, shape (7, 7, 7)
        Fully antisymmetric 3-form tensor
        
    Returns
    -------
    ndarray, shape (35,)
        Oriented components
    """
    T = np.asarray(T)
    assert T.shape == (7, 7, 7)
    triples = list(itertools.combinations(range(7), 3))
    vals = np.array([T[i, j, k] for (i, j, k) in triples], dtype=T.dtype)
    return vals


def oriented_4form_components(T):
    """
    Extract the 35 oriented components of a (7,7,7,7) tensor T,
    corresponding to indices (i,j,k,l) with i<j<k<l.

    Returns
    -------
    comps : np.ndarray, shape (35,)
        Oriented components ordered lexicographically.
    """
    if T.shape != (7, 7, 7, 7):
        raise ValueError("Input tensor must have shape (7,7,7,7)")

    indices = list(itertools.combinations(range(7), 4))
    vals = np.array([T[i, j, k, l] for (i, j, k, l) in indices], dtype=T.dtype)
    return vals


def upper_triangular_part_6x6(A, include_diagonal=True):
    """
    Extract the upper triangular part of a 6x6 matrix.

    Parameters
    ----------
    A : array_like, shape (6, 6)
        Input matrix.
    include_diagonal : bool
        Whether to include the diagonal entries.

    Returns
    -------
    v : ndarray, shape (21,) if include_diagonal else (15,)
        Upper triangular entries in row-major order.
    """
    A = np.asarray(A)
    assert A.shape == (6, 6), "Input must be a 6x6 matrix"

    if include_diagonal:
        idx = np.triu_indices(6)
    else:
        idx = np.triu_indices(6, k=1)

    return A[idx]


def split_npz(data, train=0.9, val=0.05, test=0.05, seed=42):
    """
    Split npz data into train/val/test sets.
    
    Parameters
    ----------
    data : dict-like
        Loaded npz data
    train : float
        Training fraction
    val : float
        Validation fraction
    test : float
        Test fraction
    seed : int
        Random seed
        
    Returns
    -------
    train_data, val_data, test_data : dict
        Split datasets
    """
    assert abs(train + val + test - 1.0) < 1e-8

    # Number of samples (assume all arrays share axis 0)
    N = data[data.files[0]].shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_train = int(train * N)
    n_val = int(val * N)

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    train_data = {}
    val_data = {}
    test_data = {}

    for k in data.files:
        arr = data[k]
        train_data[k] = arr[idx_train]
        val_data[k] = arr[idx_val]
        test_data[k] = arr[idx_test]

    return train_data, val_data, test_data


def sampler_g2_package_R7(p, fmodel, BASIS, rotation=0):
    """
    Sample a G2 structure on R^7 from a point in the CY base.
    
    Parameters
    ----------
    p : ndarray, shape (10,)
        Point in R^10 representation of CY
    fmodel : MultFSModel
        Trained CY metric model
    BASIS : dict
        Basis functions
    rotation : float
        Phase rotation to apply
        
    Returns
    -------
    tuple
        (base_point, link_pt, applied_rotation, phi, psi, 
         riemannian_metric, g2_metric, drop_max, drop_one, eta)
    """
    base_point = p
    applied_rotation = rotation
    
    point_cc = p[0:5] + 1.j * p[5:]
    drop_max = int(find_max_dQ_coords(point_cc, BASIS))
    drop_one = int(np.argmin(np.abs(point_cc - 1)))

    model_out = np.array(fmodel(np.expand_dims(p, axis=0))[0])
    
    riemannian_metric = riemannian_metric_real_matrix(model_out)
    
    w = kahler_form_real_matrix(model_out)
    
    w_R7 = np.pad(w, ((0, 1), (0, 1)), mode='constant')

    holomorphic_volume_form = 1 / (5 * point_cc[drop_max]**4)

    ReOmega, ImOmega = holomorphic_volume_real_imag(holomorphic_volume_form)

    ReOmega_R7 = np.pad(ReOmega, ((0, 1), (0, 1), (0, 1)), mode='constant')
    ImOmega_R7 = np.pad(ImOmega, ((0, 1), (0, 1), (0, 1)), mode='constant')

    u_coords = [i for i in range(5) if i != drop_max and i != drop_one]

    eta = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    
    point_cc = np.exp(1.j * rotation) * point_cc
    
    u_count = 0
    for i in u_coords:
        factor = point_cc[i].conjugate() - (
            point_cc[drop_max].conjugate() * (point_cc[i]**4 / point_cc[drop_max]**4)
        )
        factor = factor / np.linalg.norm(point_cc)**2
        
        eta[u_count] = factor.imag
        eta[u_count + 3] = factor.real
        
        u_count += 1
    
    link_pt = np.concatenate([
        (point_cc / np.linalg.norm(point_cc)).real,
        (point_cc / np.linalg.norm(point_cc)).imag
    ])
    
    g2 = wedge(w_R7, eta) + ReOmega_R7
    star_g2 = (1/2) * wedge(w_R7, w_R7) + wedge(eta, ImOmega_R7)
    
    g2_metric = compute_gG2(g2)
    
    return (
        base_point,
        link_pt,
        applied_rotation,
        oriented_3form_components(g2),
        oriented_4form_components(star_g2),
        upper_triangular_part_6x6(riemannian_metric),
        metric_to_vec(g2_metric),
        drop_max,
        drop_one,
        eta
    )


def main():
    """Main sampling script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample G2 structures from CY metric')
    parser.add_argument('--cy-data-dir', type=str,
                       default='./samples/cy_data',
                       help='Directory containing CY training data')
    parser.add_argument('--cy-run-number', type=int, default=None,
                       help='CY model run number to use (default: most recent)')
    parser.add_argument('--cy-model', type=str, default=None,
                       help='Path to trained CY metric model (overrides --cy-run-number)')
    parser.add_argument('--cy-config', type=str, default=None,
                       help='Path to CY model configuration file (overrides --cy-run-number)')
    parser.add_argument('--output-dir', type=str,
                       default='./samples/link_data',
                       help='Directory to save G2 dataset')
    parser.add_argument('--n-points', type=int, default=None,
                       help='Number of base points to sample (default: all training points)')
    parser.add_argument('--n-rotations', type=int, default=4,
                       help='Number of random rotations per base point')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("G2 Structure Sampling")
    print("=" * 80)
    
    # Determine CY run number
    if args.cy_model is None and args.cy_config is None:
        # Need to determine run number
        if args.cy_run_number is None:
            # Auto-detect most recent
            cy_run_number = get_most_recent_cy_run_number('./models/cy_models')
            if cy_run_number is None:
                print("Error: No CY models found in ./models/cy_models/")
                print("Please train a CY model first using run_cy.py")
                sys.exit(1)
            print(f"Auto-detected most recent CY model: run {cy_run_number}")
        else:
            cy_run_number = args.cy_run_number
            print(f"Using specified CY model run: {cy_run_number}")
        
        # Construct model and config paths
        cy_model_path = f'./models/cy_models/cy_metric_model_run{cy_run_number}.keras'
        cy_config_path = f'./models/cy_models/cy_metric_model_run{cy_run_number}_config.yaml'
    else:
        # Use explicitly provided paths (backward compatibility)
        cy_model_path = args.cy_model or './models/cy_models/cy_metric_model.keras'
        cy_config_path = args.cy_config or './models/cy_models/cy_metric_model_config.yaml'
        
        # Try to extract run number from path
        match = re.search(r'cy_metric_model_run(\d+)', cy_model_path)
        if match:
            cy_run_number = int(match.group(1))
            print(f"Detected CY model run number from path: {cy_run_number}")
        else:
            cy_run_number = None
            print("Warning: Could not detect run number from model path")
    
    # Load CY data and basis
    cy_data_dir = pathlib.Path(args.cy_data_dir)
    data = np.load(cy_data_dir / 'dataset.npz')
    BASIS = np.load(cy_data_dir / 'basis.pickle', allow_pickle=True)
    BASIS = prepare_basis(BASIS)
    
    print(f"Loaded CY data from {cy_data_dir}")
    
    # Load model configuration
    config_path = pathlib.Path(cy_config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded CY model config from {config_path}")
    
    # Extract model architecture parameters from config
    n_layers = config['n_layers']
    n_hidden = config['n_hidden']
    activation = config['activation']
    alpha = config['alpha']
    n_fold = config['n_fold']
    n_in = config['n_in']
    n_out = config['n_out']
    
    # Load trained model
    loaded_nn = tf.keras.models.load_model(cy_model_path)
    fmodel = MultFSModel(loaded_nn, BASIS, alpha=alpha)
    
    print(f"Loaded CY model from {cy_model_path}")
    print(f"Model architecture: {n_layers} layers x {n_hidden} units, activation={activation}")
    
    def compute_sample(point, rotation):
        return sampler_g2_package_R7(point, fmodel, BASIS, rotation=rotation)

    X = data["X_train"]
    
    # Use specified number of points or all training points
    if args.n_points is not None:
        n_base_points = min(args.n_points, len(X))
        X = X[:n_base_points]
    else:
        n_base_points = len(X)
    
    print(f"Using {n_base_points} base points")

    # Create tasks: one with no rotation, n_rotations with random rotations
    tasks = []
    for pt in X:
        tasks.append((pt, 0.0))
        for _ in range(args.n_rotations):
            tasks.append((pt, np.random.uniform(0, 2 * np.pi)))
    
    total_samples = n_base_points * (1 + args.n_rotations)
    print(f"Sampling {total_samples} G2 structures ({n_base_points} base points × {1 + args.n_rotations} rotations)...")
    
    # Parallel sampling
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(compute_sample)(pt, rot)
        for pt, rot in tqdm(tasks, desc="Sampling G2 points")
    )

    # Unpack results
    base_points = np.stack([r[0] for r in results])
    link_points = np.stack([r[1] for r in results])
    rotations = np.stack([r[2] for r in results])
    phis = np.stack([r[3] for r in results])
    psis = np.stack([r[4] for r in results])
    riemannian_metrics = np.stack([r[5] for r in results])
    g2_metrics = np.stack([r[6] for r in results])
    drop_maxs = np.stack([r[7] for r in results])
    drop_ones = np.stack([r[8] for r in results])
    etas = np.stack([r[9] for r in results])

    # Save full dataset
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = output_dir / "g2_dataset.npz"
    
    # Create metadata dictionary
    metadata_dict = {
        'base_points': base_points,
        'link_points': link_points,
        'rotations': rotations,
        'phis': phis,
        'psis': psis,
        'riemannian_metrics': riemannian_metrics,
        'g2_metrics': g2_metrics,
        'drop_maxs': drop_maxs,
        'drop_ones': drop_ones,
        'etas': etas
    }
    
    # Add cy_run_number if available
    if cy_run_number is not None:
        # Store as scalar array so it's preserved in npz
        metadata_dict['cy_run_number'] = np.array([cy_run_number])
    
    np.savez_compressed(dataset_path, **metadata_dict)

    print(f"\nSaved G2 dataset to {dataset_path}")
    if cy_run_number is not None:
        print(f"Dataset metadata: cy_run_number = {cy_run_number}")
    
    # Split into train/val/test
    data = np.load(dataset_path)
    train_data, val_data, test_data = split_npz(data, train=0.9, val=0.05, test=0.05)

    np.savez_compressed(output_dir / "g2_train.npz", **train_data)
    np.savez_compressed(output_dir / "g2_val.npz", **val_data)
    np.savez_compressed(output_dir / "g2_test.npz", **test_data)
    
    print(f"Split dataset into train/val/test in {output_dir}")
    print("=" * 80)
    print("Sampling complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
