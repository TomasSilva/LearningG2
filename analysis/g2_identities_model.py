#!/usr/bin/env python3
"""
Check G2 identities on LEARNED G2 model predictions.

This script verifies the following G2 identities using trained model predictions:
1. φ ∧ ψ = 7·Vol(K_f, g_φ)
2. dψ = 0
3. dφ = ω²

where φ and metric are predicted by trained neural networks.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import yaml
import glob
import re

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import geometry functions
from geometry.geometry import (
    kahler_form_real_matrix, 
    holomorphic_volume_real_imag,
    compute_gG2,
    find_max_dQ_coords
)
from geometry.wedge import wedge
from geometry.numerical_exterior_derivative import (
    sample_numerical_g2_neighborhood_val,
    numerical_d_g2,
    quintic_solver
)
from geometry.compression import form_to_vec, vec_to_form, metric_to_vec, vec_to_metric
from geometry.numerical_star_R7 import Hodge_Dual

# Import analysis utilities
from analysis.utils import (
    get_most_recent_g2_run_number,
    get_most_recent_cy_run_number,
    load_g2_models,
    load_cy_model,
    load_g2_data,
    print_statistics
)


def compute_model_mse(g2_data, g2_models):
    """Compute MSE for 3form and metric models on test data."""
    print("\nEvaluating G2 models on test set...")
    
    # Prepare input: link_points (10) + etas (7) + patch indices (2) = 19 dimensions
    link_points = g2_data['link_points']
    etas = g2_data['etas']
    drop_maxs = g2_data['drop_maxs'].reshape(-1, 1)
    drop_ones = g2_data['drop_ones'].reshape(-1, 1)
    X = np.concatenate([link_points, etas, drop_maxs, drop_ones], axis=1)
    
    results = {}
    
    # Evaluate 3form model
    Y_true_3form = g2_data['phis']
    Y_pred_3form = g2_models['3form'].predict(X, verbose=0)
    mse_3form = np.mean((Y_true_3form - Y_pred_3form)**2)
    results['3form'] = mse_3form
    print(f"  3form model MSE: {mse_3form:.6e}")
    
    # Evaluate metric model
    Y_true_metric = g2_data['g2_metrics']
    Y_pred_metric = g2_models['metric'].predict(X, verbose=0)
    mse_metric = np.mean((Y_true_metric - Y_pred_metric)**2)
    results['metric'] = mse_metric
    print(f"  Metric model MSE: {mse_metric:.6e}")
    
    return results


def compute_hodge_star_psi(phi, metric):
    """
    Compute ψ = ⋆_{g_G2} φ using the Hodge star operator.
    
    For G2 geometry, the metric for the Hodge star should be g_φ (the metric induced by φ),
    but we can also use the predicted G2 metric directly.
    
    Parameters
    ----------
    phi : ndarray, shape (35,) or (7,7,7)
        3-form φ (as vector or tensor)
    metric : ndarray, shape (49,) or (7,7)
        G2 metric tensor (as vector or matrix)
        
    Returns
    -------
    ndarray, shape (7,7,7,7)
        4-form ψ = ⋆φ
    """
    # Convert to tensor forms if needed
    if phi.ndim == 1:
        phi = vec_to_form(phi, n=7, k=3)
    if metric.ndim == 1:
        metric = vec_to_metric(metric)
    
    # Compute Hodge star: ψ = ⋆_{g} φ
    psi = Hodge_Dual(phi, metric)
    
    return psi


def check_phi_wedge_psi(data, g2_models, n_points=100):
    """Check φ ∧ ψ = 7·Vol(g_φ) using model predictions."""
    link_points = data["link_points"]
    etas = data["etas"]
    drop_maxs = data["drop_maxs"]
    drop_ones = data["drop_ones"]
    
    # Random sampling if n_points < dataset size
    total_points = len(link_points)
    if n_points < total_points:
        indices = np.random.choice(total_points, size=n_points, replace=False)
        print(f"\nChecking φ ∧ ψ = 7·Vol(g_φ) on {n_points} randomly sampled points (using model predictions)...")
    else:
        indices = np.arange(total_points)
        n_points = total_points
        print(f"\nChecking φ ∧ ψ = 7·Vol(g_φ) on all {n_points} points (using model predictions)...")
    
    vals = []
    for idx in tqdm(indices, desc="φ∧ψ check (model)"):
        # Prepare model input: link_point (10) + eta (7) + patch indices (2) = 19
        link_pt = link_points[idx]
        eta = etas[idx]
        drop_max = drop_maxs[idx]
        drop_one = drop_ones[idx]
        
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        
        # Predict φ and G2 metric
        phi_vec = g2_models['3form'].predict(X_input, verbose=0)[0]
        metric_vec = g2_models['metric'].predict(X_input, verbose=0)[0]
        
        phi = vec_to_form(phi_vec, n=7, k=3)
        metric = vec_to_metric(metric_vec)
        
        # Compute ψ = ⋆_{g_G2} φ using the Hodge star operator
        psi = Hodge_Dual(phi, metric)
        
        prod = wedge(phi, psi)[0, 1, 2, 3, 4, 5, 6]
        vol = np.sqrt(np.linalg.det(metric))
        
        vals.append(prod / vol)
    
    return np.array(vals)


def check_d_psi_and_d_phi(data, g2_models, fmodel, BASIS, n_points=100, 
                          epsilon=1e-12, global_rotation_epsilon=1e-12):
    """Check dψ = 0 and dφ = ω² using model predictions in neighborhoods."""
    
    link_points = data['link_points']
    base_points = data['base_points']
    etas = data['etas']
    drop_maxs = data['drop_maxs']
    drop_ones = data['drop_ones']
    rotations = data['rotations']
    
    # Random sampling if n_points < dataset size
    total_points = len(base_points)
    if n_points < total_points:
        indices = np.random.choice(total_points, size=n_points, replace=False)
        print(f"\nChecking dψ = 0 and dφ = ω² on {n_points} randomly sampled points (using models)...")
    else:
        indices = np.arange(total_points)
        n_points = total_points
        print(f"\nChecking dψ = 0 and dφ = ω² on all {n_points} points (using models)...")
    
    vals_dpsi = []
    vals_dphi = []
    vals_ratio = []
    
    def compute_link_features(p, rotation=0):
        """Compute link_point, eta, drop_max, drop_one from base_point."""
        point_cc = p[0:5] + 1.j * p[5:]
        drop_max = int(find_max_dQ_coords(point_cc, BASIS))
        drop_one = int(np.argmin(np.abs(point_cc - 1)))
        
        # Apply rotation
        point_cc = np.exp(1.j * rotation) * point_cc
        
        # Compute eta
        u_coords = [i for i in range(5) if i != drop_max and i != drop_one]
        eta = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
        
        u_count = 0
        for i in u_coords:
            factor = point_cc[i].conjugate() - (
                point_cc[drop_max].conjugate() * (point_cc[i]**4 / point_cc[drop_max]**4)
            )
            factor = factor / np.linalg.norm(point_cc)**2
            
            eta[u_count] = factor.imag
            eta[u_count + 3] = factor.real
            u_count += 1
        
        # Compute link_point (normalized)
        link_pt = np.concatenate([
            (point_cc / np.linalg.norm(point_cc)).real,
            (point_cc / np.linalg.norm(point_cc)).imag
        ])
        
        return link_pt, eta, drop_max, drop_one
    
    def predict_phi_at_point(p, rotation=0):
        """Predict φ at a point using trained 3form model."""
        # p is base_point (10d), compute link features
        link_pt, eta, drop_max, drop_one = compute_link_features(p, rotation)
        
        # Model input: link_point (10) + eta (7) + patch indices (2) = 19
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        
        # Predict φ
        phi_vec = g2_models['3form'].predict(X_input, verbose=0)[0]
        phi = vec_to_form(phi_vec, n=7, k=3)
        return phi
    
    def predict_psi_at_point(p, rotation=0):
        """Predict ψ at a point using trained models and Hodge star."""
        # p is base_point (10d), compute link features
        link_pt, eta, drop_max, drop_one = compute_link_features(p, rotation)
        
        # Model input: link_point (10) + eta (7) + patch indices (2) = 19
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        
        # Predict φ and G2 metric
        phi_vec = g2_models['3form'].predict(X_input, verbose=0)[0]
        metric_vec = g2_models['metric'].predict(X_input, verbose=0)[0]
        
        phi = vec_to_form(phi_vec, n=7, k=3)
        metric = vec_to_metric(metric_vec)
        
        # Compute ψ = ⋆_{g_G2} φ using the Hodge star operator
        psi = Hodge_Dual(phi, metric)
        return psi
    
    find_max_fn = lambda point_cc: find_max_dQ_coords(point_cc, BASIS)
    
    for idx in tqdm(indices, desc="dψ and dφ check (model)"):
        base_point = base_points[idx]
        rotation = rotations[idx]
        
        # Check dψ using model predictions in neighborhood
        dic_psi = sample_numerical_g2_neighborhood_val(
            lambda p: predict_psi_at_point(p, rotation), base_point, epsilon,
            find_max_dQ_coords_fn=find_max_fn,
            global_rotation_epsilon=global_rotation_epsilon
        )
        d_psi = numerical_d_g2(dic_psi, epsilon)
        vals_dpsi.append(np.linalg.norm(d_psi))
        
        # Check dφ using model predictions in neighborhood
        dic_phi = sample_numerical_g2_neighborhood_val(
            lambda p: predict_phi_at_point(p, rotation), base_point, epsilon,
            find_max_dQ_coords_fn=find_max_fn,
            global_rotation_epsilon=global_rotation_epsilon
        )
        d_phi = numerical_d_g2(dic_phi, epsilon)
        vals_dphi.append(np.linalg.norm(d_phi))
        
        # Compute ω² for comparison
        cy_metric = np.array(fmodel(np.expand_dims(base_point, axis=0))[0])
        w = kahler_form_real_matrix(cy_metric)
        w_R7 = np.pad(w, ((0, 1), (0, 1)), mode='constant')
        w2 = wedge(w_R7, w_R7)
        
        vals_ratio.append(np.linalg.norm(d_phi) / np.linalg.norm(w2))
    
    return np.array(vals_dpsi), np.array(vals_dphi), np.array(vals_ratio)


def plot_phi_wedge_psi(vals, run_number, output_dir):
    """Plot φ∧ψ/Vol check results."""
    plt.figure(figsize=(10, 6))
    plt.plot(vals, marker='.', linestyle='-', alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\frac{\varphi\wedge\psi}{\sqrt{\det(g_{\varphi})}}$")
    plt.axhline(y=7, linestyle=':', linewidth=2, color='red',
                label=r"$\frac{\varphi\wedge\psi}{\sqrt{\det(g_{\varphi})}}=7$")
    plt.ylim(6.5, 7.5)
    plt.title(f"G2 Identity Check (MODEL): φ∧ψ = 7·Vol (Run {run_number})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / f"g2_phi_wedge_psi_model_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_dpsi(vals_dpsi, run_number, output_dir):
    """Plot ||dψ|| distribution."""
    q_low, q_high = np.percentile(vals_dpsi, [1, 99])
    vals_filtered = vals_dpsi[(vals_dpsi >= q_low) & (vals_dpsi <= q_high)]
    
    # Scatter plot
    plt.figure(figsize=(7, 5))
    plt.plot(vals_dpsi, marker='.', linestyle='None', alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\|\mathrm{d}\psi\|$")
    plt.title(f"||dψ|| per Sample (MODEL, Run {run_number})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"g2_dpsi_model_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    
    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel(r"$\|\mathrm{d}\psi\|$")
    plt.ylabel("Count")
    plt.title(f"||dψ|| Distribution (MODEL, 1-99 percentile, Run {run_number})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"g2_dpsi_model_run{run_number}_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_dphi(vals_dphi, run_number, output_dir):
    """Plot ||dφ|| distribution."""
    q_low, q_high = np.percentile(vals_dphi, [1, 99])
    vals_filtered = vals_dphi[(vals_dphi >= q_low) & (vals_dphi <= q_high)]
    
    # Scatter plot
    plt.figure(figsize=(7, 5))
    plt.plot(vals_dphi, marker='.', linestyle='None', alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\|\mathrm{d}\varphi\|$")
    plt.title(f"||dφ|| per Sample (MODEL, Run {run_number})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_model_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    
    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel(r"$\|\mathrm{d}\varphi\|$")
    plt.ylabel("Count")
    plt.title(f"||dφ|| Distribution (MODEL, 1-99 percentile, Run {run_number})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_model_run{run_number}_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_ratio(vals_ratio, run_number, output_dir):
    """Plot ||dφ|| / ||ω²|| ratio distribution."""
    q_low, q_high = np.percentile(vals_ratio, [1, 99])
    vals_filtered = vals_ratio[(vals_ratio >= q_low) & (vals_ratio <= q_high)]
    
    # Scatter plot
    plt.figure(figsize=(7, 5))
    plt.plot(vals_ratio, marker='.', linestyle='None', alpha=0.6)
    plt.axhline(y=1.0, linestyle='--', color='red', alpha=0.7, label='Ideal ratio = 1')
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\|\mathrm{d}\varphi\| / \|\omega^2\|$")
    plt.title(f"dφ = ω² Check (MODEL): Ratio (Run {run_number})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_omega_ratio_model_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    
    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=1.0, linestyle='--', color='red', alpha=0.7, label='Ideal ratio = 1')
    plt.xlabel(r"$\|\mathrm{d}\varphi\| / \|\omega^2\|$")
    plt.ylabel("Count")
    plt.title(f"Ratio Distribution (MODEL, 1-99 percentile, Run {run_number})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_omega_ratio_model_run{run_number}_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Check G2 identities on LEARNED model predictions'
    )
    parser.add_argument('--g2-run-number', type=int, default=None,
                       help='G2 model run number to use (default: most recent)')
    parser.add_argument('--cy-run-number', type=int, default=None,
                       help='CY model run number to use (default: auto-detect from dataset)')
    parser.add_argument('--g2-data', type=str, default='./samples/link_data/g2_test.npz',
                       help='Path to G2 test dataset')
    parser.add_argument('--cy-data-dir', type=str, default='./samples/cy_data',
                       help='Directory containing CY data')
    parser.add_argument('--n-points', type=int, default=None,
                       help='Number of points for all identity checks (default: all, or random sample if specified)')
    parser.add_argument('--epsilon', type=float, default=1e-12,
                       help='Epsilon for numerical derivative')
    parser.add_argument('--rotation-epsilon', type=float, default=1e-12,
                       help='Epsilon for global phase rotation')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Directory to save output plots')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent.parent
    g2_data_path = (script_dir / args.g2_data).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("G2 Identities Check (LEARNED MODELS)")
    print("=" * 80)
    
    # Determine G2 run number
    if args.g2_run_number is None:
        g2_run_number = get_most_recent_g2_run_number(script_dir / 'models' / 'link_models')
        if g2_run_number is None:
            print("Error: No G2 models found in ./models/link_models/")
            print("Please train G2 models first using run_g2.py")
            sys.exit(1)
        print(f"Auto-detected most recent G2 model: run {g2_run_number}")
    else:
        g2_run_number = args.g2_run_number
        print(f"Using specified G2 model run: {g2_run_number}")
    
    # Load G2 data
    g2_data, dataset_cy_run = load_g2_data(g2_data_path)
    
    # Load G2 models
    g2_models = load_g2_models(g2_run_number, script_dir)
    if g2_models is None:
        print("Error: Could not load G2 models")
        sys.exit(1)
    
    # Determine CY run number
    if args.cy_run_number is None:
        if dataset_cy_run is not None:
            cy_run_number = dataset_cy_run
        else:
            cy_run_number = get_most_recent_cy_run_number(script_dir / 'models' / 'cy_models')
            if cy_run_number is None:
                print("Error: No CY models found and no cy_run_number in dataset")
                sys.exit(1)
            print(f"Auto-detected most recent CY model: run {cy_run_number}")
    else:
        cy_run_number = args.cy_run_number
        print(f"Using specified CY model run: {cy_run_number}")
        
        if dataset_cy_run is not None and cy_run_number != dataset_cy_run:
            print("=" * 80)
            print("WARNING: CY run number mismatch!")
            print(f"  Dataset was generated with CY run {dataset_cy_run}")
            print(f"  But you specified CY run {cy_run_number}")
            print("=" * 80)
    
    print()
    
    # Check 0: Compute MSE on test set
    mse_results = compute_model_mse(g2_data, g2_models)
    
    # Load CY model for derivative checks (needed for ω)
    fmodel, BASIS, cy_data = load_cy_model(cy_run_number, args.cy_data_dir, script_dir)
    
    # Check 1: φ ∧ ψ = 7·Vol (using model predictions)
    n_points = len(g2_data['phis']) if args.n_points is None else args.n_points
    vals_phi_psi = check_phi_wedge_psi(g2_data, g2_models, n_points)
    print_statistics("φ∧ψ/Vol (model predictions)", vals_phi_psi)
    plot_phi_wedge_psi(vals_phi_psi, g2_run_number, output_dir)
    
    # Check 2 & 3: dψ = 0 and dφ = ω² (using model predictions in neighborhoods)
    vals_dpsi, vals_dphi, vals_ratio = check_d_psi_and_d_phi(
        g2_data, g2_models, fmodel, BASIS, n_points,
        args.epsilon, args.rotation_epsilon
    )
    
    print_statistics("||dψ|| (model predictions)", vals_dpsi)
    print_statistics("||dφ|| (model predictions)", vals_dphi)
    print_statistics("||dφ||/||ω²|| (model predictions)", vals_ratio)
    
    plot_dpsi(vals_dpsi, g2_run_number, output_dir)
    plot_dphi(vals_dphi, g2_run_number, output_dir)
    plot_ratio(vals_ratio, g2_run_number, output_dir)
    
    print()
    print("=" * 80)
    print("G2 Identities Check (MODELS) Complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
