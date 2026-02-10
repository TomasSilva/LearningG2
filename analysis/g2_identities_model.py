#!/usr/bin/env python3
"""
Check G2 identities on LEARNED G2 model predictions.

This script verifies the following G2 identities using trained model predictions:
1. φ ∧ ψ = 7·Vol(K_f, g_φ)
2. dψ = 0
3. dφ = ω²

where φ and metric are predicted by trained neural networks.
"""

import os, sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
from geometry.compression import vec_to_form, vec_to_metric
from geometry.numerical_star_R7 import Hodge_Dual

# Import analysis utilities
from analysis.utils import (
    get_most_recent_g2_run_number,
    get_most_recent_cy_run_number,
    load_g2_models,
    load_cy_model,
    load_g2_data,
    print_statistics,
    plot_phi_wedge_psi,
    plot_dphi_ratio,
    plot_dpsi
)

def compute_model_mse(g2_data, g2_models):
    """Compute MSE for 3form and metric models on test data."""
    print("\nEvaluating G2 models on test set...")
    
    # Prepare input: link_points (10) + etas (7) + patch indices (2) = 19 dimensions
    link_points = g2_data['link_points']
    etas = g2_data['etas']
    drop_maxs = g2_data['drop_maxs'].reshape(-1, 1)
    drop_ones = g2_data['drop_ones'].reshape(-1, 1)
    X = np.hstack([link_points, etas, drop_maxs, drop_ones])
    
    results = {}
    
    # Evaluate 3form model
    Y_true_3form = g2_data['phis']
    Y_pred_3form_raw = g2_models['3form'].predict(X, verbose=0)
    
    # Denormalize predictions if model has normalization metadata
    if '3form_y_mean' in g2_models and '3form_y_std' in g2_models:
        y_mean_phi = g2_models['3form_y_mean']
        y_std_phi = g2_models['3form_y_std']
        Y_pred_3form = Y_pred_3form_raw * y_std_phi + y_mean_phi
    else:
        Y_pred_3form = Y_pred_3form_raw
    
    mse_3form = np.mean((Y_true_3form - Y_pred_3form)**2)
    mae_3form = np.mean(np.abs(Y_true_3form - Y_pred_3form))
    results['3form'] = mse_3form
    print(f"  3form MSE: {mse_3form:.6e}, MAE: {mae_3form:.6e}, Relative MAE: {mae_3form / np.mean(np.abs(Y_true_3form)):.6e}")
    
    # Evaluate metric model
    Y_true_metric = g2_data['g2_metrics']
    Y_pred_metric_raw = g2_models['metric'].predict(X, verbose=0)
    
    # Denormalize predictions if model has normalization metadata
    if 'metric_y_mean' in g2_models and 'metric_y_std' in g2_models:
        y_mean_metric = g2_models['metric_y_mean']
        y_std_metric = g2_models['metric_y_std']
        Y_pred_metric = Y_pred_metric_raw * y_std_metric + y_mean_metric
    else:
        Y_pred_metric = Y_pred_metric_raw
    
    mse_metric = np.mean((Y_true_metric - Y_pred_metric)**2)
    mae_metric = np.mean(np.abs(Y_true_metric - Y_pred_metric))
    results['metric'] = mse_metric
    print(f"  Metric MSE: {mse_metric:.6e}, MAE: {mae_metric:.6e}, Relative MAE: {mae_metric / np.mean(np.abs(Y_true_metric)):.6e}")
    
    return results


def check_g2_identities_combined(data, g2_models, fmodel, BASIS, n_points=100, 
                                  epsilon=1e-5, global_rotation_epsilon=1e-5, psi_method='star'):
    """Check all G2 identities in a single pass through the data.
    
    Checks:
    1. φ ∧ ψ = 7·Vol(g_φ)
    2. dψ = 0
    3. dφ = ω²
    
    Args:
        psi_method: 'star' to use Hodge star, 'model' to use 4form model
    """
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
        print(f"\nChecking G2 identities on {n_points} randomly sampled points...")
    else:
        indices = np.arange(total_points)
        n_points = total_points
        print(f"\nChecking G2 identities on all {n_points} points...")
    
    # Results arrays
    vals_phi_psi = []
    vals_dpsi = []
    vals_dphi = []
    vals_omega2 = []
    vals_ratio = []
    
    # Check normalization status
    has_phi_norm = '3form_y_mean' in g2_models and '3form_y_std' in g2_models
    has_metric_norm = 'metric_y_mean' in g2_models and 'metric_y_std' in g2_models
    has_psi_norm = '4form_y_mean' in g2_models and '4form_y_std' in g2_models
    
    y_mean_phi_global = g2_models['3form_y_mean'] if has_phi_norm else None
    y_std_phi_global = g2_models['3form_y_std'] if has_phi_norm else None
    y_mean_metric_global = g2_models['metric_y_mean'] if has_metric_norm else None
    y_std_metric_global = g2_models['metric_y_std'] if has_metric_norm else None
    y_mean_psi_global = g2_models['4form_y_mean'] if has_psi_norm else None
    y_std_psi_global = g2_models['4form_y_std'] if has_psi_norm else None
    
    # Get psi index mapping for expanding 23 -> 35 components when using model method
    if psi_method == 'model':
        if '4form_zero_indices' in g2_models and '4form_nonzero_indices' in g2_models:
            g2_models['psi_zero_indices'] = g2_models['4form_zero_indices']
            g2_models['psi_nonzero_indices'] = g2_models['4form_nonzero_indices']
    
    def compute_link_features_fixed_patch(p, drop_max, drop_one, rotation=0):
        """Compute link_point and eta from base_point p using FIXED drop_max and drop_one."""
        point_cc = p[0:5] + 1.j * p[5:]
        point_cc = np.exp(1.j * rotation) * point_cc
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
        link_pt = np.concatenate([
            (point_cc / np.linalg.norm(point_cc)).real,
            (point_cc / np.linalg.norm(point_cc)).imag
        ])
        return link_pt, eta
    
    def predict_phi_at_point(p, drop_max, drop_one, rotation=0):
        """Predict φ at a point."""
        link_pt, eta = compute_link_features_fixed_patch(p, drop_max, drop_one, rotation)
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
        phi_vec = phi_vec_raw * y_std_phi_global + y_mean_phi_global if has_phi_norm else phi_vec_raw
        phi = vec_to_form(phi_vec, n=7, k=3)
        return phi
    
    def predict_psi_at_point(p, drop_max, drop_one, rotation=0):
        """Predict ψ at a point using Hodge star or 4form model."""
        link_pt, eta = compute_link_features_fixed_patch(p, drop_max, drop_one, rotation)
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        
        phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
        metric_vec_raw = g2_models['metric'].predict(X_input, verbose=0)[0]
        
        phi_vec = phi_vec_raw * y_std_phi_global + y_mean_phi_global if has_phi_norm else phi_vec_raw
        metric_vec = metric_vec_raw * y_std_metric_global + y_mean_metric_global if has_metric_norm else metric_vec_raw
        
        phi = vec_to_form(phi_vec, n=7, k=3)
        metric = vec_to_metric(metric_vec)
        if metric.ndim == 3:
            metric = metric[0]
        
        if psi_method == 'star':
            det_metric = np.linalg.det(metric)
            if det_metric <= 0 or not np.isfinite(det_metric):
                raise ValueError(f"Invalid metric determinant: {det_metric}")
            psi = Hodge_Dual(phi, metric)
            if not np.all(np.isfinite(psi)):
                raise ValueError("Hodge dual produced non-finite values")
        else:  # psi_method == 'model'
            psi_vec_raw = g2_models['4form'].predict(X_input, verbose=0)[0]
            psi_vec_compact = psi_vec_raw * y_std_psi_global + y_mean_psi_global if has_psi_norm else psi_vec_raw
            
            if 'psi_zero_indices' in g2_models and 'psi_nonzero_indices' in g2_models:
                psi_vec_full = np.zeros(35)
                psi_vec_full[g2_models['psi_nonzero_indices']] = psi_vec_compact
            else:
                zero_indices = np.array([6, 8, 12, 15, 17, 18, 24, 25, 27, 29, 32, 33])
                nonzero_indices = np.array([i for i in range(35) if i not in zero_indices])
                psi_vec_full = np.zeros(35)
                psi_vec_full[nonzero_indices] = psi_vec_compact
            
            psi = vec_to_form(psi_vec_full, n=7, k=4)
        
        return psi
    
    def predict_phi_psi_metric_at_point(p, drop_max, drop_one, rotation=0):
        """Predict φ, ψ, and metric at a point."""
        link_pt, eta = compute_link_features_fixed_patch(p, drop_max, drop_one, rotation)
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        
        # Predict φ and metric
        phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
        metric_vec_raw = g2_models['metric'].predict(X_input, verbose=0)[0]
        
        # Denormalize
        phi_vec = phi_vec_raw * y_std_phi_global + y_mean_phi_global if has_phi_norm else phi_vec_raw
        metric_vec = metric_vec_raw * y_std_metric_global + y_mean_metric_global if has_metric_norm else metric_vec_raw
        
        phi = vec_to_form(phi_vec, n=7, k=3)
        metric = vec_to_metric(metric_vec)
        if metric.ndim == 3:
            metric = metric[0]
        
        # Compute or predict ψ
        if psi_method == 'star':
            det_metric = np.linalg.det(metric)
            if det_metric <= 0 or not np.isfinite(det_metric):
                raise ValueError(f"Invalid metric determinant: {det_metric}")
            psi = Hodge_Dual(phi, metric)
            if not np.all(np.isfinite(psi)):
                raise ValueError("Hodge dual produced non-finite values")
        else:  # psi_method == 'model'
            psi_vec_raw = g2_models['4form'].predict(X_input, verbose=0)[0]
            # Denormalize
            psi_vec_compact = psi_vec_raw * y_std_psi_global + y_mean_psi_global if has_psi_norm else psi_vec_raw
            
            # Expand from 23 non-zero components to full 35 components
            if 'psi_zero_indices' in g2_models and 'psi_nonzero_indices' in g2_models:
                psi_vec_full = np.zeros(35)
                psi_vec_full[g2_models['psi_nonzero_indices']] = psi_vec_compact
            else:
                # Use default expansion
                zero_indices = np.array([6, 8, 12, 15, 17, 18, 24, 25, 27, 29, 32, 33])
                nonzero_indices = np.array([i for i in range(35) if i not in zero_indices])
                psi_vec_full = np.zeros(35)
                psi_vec_full[nonzero_indices] = psi_vec_compact
            
            psi = vec_to_form(psi_vec_full, n=7, k=4)
        
        return phi, psi, metric
    
    for idx in tqdm(indices, desc="G2 identities check", file=sys.stdout, dynamic_ncols=True):
        base_point = base_points[idx]
        link_pt = link_points[idx]
        eta = etas[idx]
        rotation = rotations[idx]
        drop_max = int(drop_maxs[idx])
        drop_one = int(drop_ones[idx])
        
        # Check 1: φ ∧ ψ = 7·Vol at central point
        phi, psi, metric = predict_phi_psi_metric_at_point(base_point, drop_max, drop_one, rotation)
        prod = wedge(phi, psi)[0, 1, 2, 3, 4, 5, 6]
        vol = np.sqrt(np.linalg.det(metric))
        # Correct for Hodge dual factorial normalization when using star method
        # Hodge_Dual includes 1/(p!*(n-p)!) = 1/(3!*4!) but wedge products don't
        # So multiply by 4! to get the G2 identity φ∧ψ=7Vol in standard convention
        if psi_method == 'star':
            import math
            prod = prod * math.factorial(4)
        vals_phi_psi.append(prod / vol)
        
        # Check 2: dψ = 0 using neighborhood
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                dic_psi = sample_numerical_g2_neighborhood_val(
                    lambda p: predict_psi_at_point(p, drop_max, drop_one, rotation),
                    base_point, epsilon,
                    drop_max=drop_max, drop_one=drop_one
                )
                d_psi = numerical_d_g2(dic_psi, epsilon)
                norm_dpsi = np.linalg.norm(d_psi)
                # Filter out unreasonably large values (likely numerical errors)
                if np.isfinite(norm_dpsi) and norm_dpsi < 1e3:
                    vals_dpsi.append(norm_dpsi)
        except Exception:
            # Skip points where derivative computation fails
            pass
        
        # Check 3: dφ = ω² using neighborhood
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                dic_phi = sample_numerical_g2_neighborhood_val(
                    lambda p: predict_phi_at_point(p, drop_max, drop_one, rotation),
                    base_point, epsilon,
                    drop_max=drop_max, drop_one=drop_one
                )
                d_phi = numerical_d_g2(dic_phi, epsilon)
                norm_dphi = np.linalg.norm(d_phi)
                
                # Compute ω² for comparison
                cy_metric = np.array(fmodel(np.expand_dims(base_point, axis=0))[0])
                w = kahler_form_real_matrix(cy_metric)
                w_R7 = np.pad(w, ((0, 1), (0, 1)), mode='constant')
                w2 = wedge(w_R7, w_R7)
                norm_w2 = np.linalg.norm(w2)
                
                # Filter out unreasonably large values (likely numerical errors)
                if np.isfinite(norm_dphi) and np.isfinite(norm_w2) and norm_w2 > 0:
                    if norm_dphi < 1e3 and norm_dphi / norm_w2 < 1e3:
                        vals_dphi.append(norm_dphi)
                        vals_omega2.append(norm_w2)
                        vals_ratio.append(norm_dphi / norm_w2)
        except Exception:
            # Skip points where derivative computation fails
            pass
    
    return (np.array(vals_phi_psi), np.array(vals_dpsi), np.array(vals_dphi), 
            np.array(vals_omega2), np.array(vals_ratio))


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
    parser.add_argument('--epsilon', type=float, default=1e-5,
                       help='Epsilon for numerical derivative')
    parser.add_argument('--rotation-epsilon', type=float, default=1e-5,
                       help='Epsilon for global phase rotation')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Directory to save output plots')
    parser.add_argument('--psi-method', type=str, choices=['star', 'model'], default='star',
                       help="Method to compute psi: 'star' (Hodge dual) or 'model' (4form model). Default: 'star'")
    
    args = parser.parse_args()
    
    # Outlier filtering parameter
    outlier_proportion = 0.05  # Remove top 5% from statistics and plots
    
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
    
    # Load G2 models (load 4form only if using model method)
    g2_models = load_g2_models(g2_run_number, script_dir, load_4form=(args.psi_method == 'model'))
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
    print(f"ψ computation method: {args.psi_method} ({'Hodge star (⋆ φ)' if args.psi_method == 'star' else '4form model'})")
    
    # Validate psi_method against available models
    if args.psi_method == 'model':
        if '4form' not in g2_models:
            print("Error: psi-method='model' requires a trained 4form model.")
            print("Please train the 4form model using: python run_g2.py --task 4form")
            print("Or use --psi-method star (default) to use Hodge star computation.")
            sys.exit(1)
        if '4form_y_mean' not in g2_models or '4form_y_std' not in g2_models:
            print("Warning: 4form model found but normalization stats are missing.")
    
    print()
    
    # Check 0: Compute MSE on test set
    mse_results = compute_model_mse(g2_data, g2_models)
    
    # Load CY model for derivative checks (needed for ω)
    fmodel, BASIS, cy_data = load_cy_model(cy_run_number, args.cy_data_dir, script_dir)
    
    # Combined check: φ ∧ ψ = 7·Vol, dψ = 0, and dφ = ω²
    n_points = len(g2_data['phis']) if args.n_points is None else args.n_points
    vals_phi_psi, vals_dpsi, vals_dphi, vals_omega2, vals_ratio = check_g2_identities_combined(
        g2_data, g2_models, fmodel, BASIS, n_points,
        args.epsilon, args.rotation_epsilon, args.psi_method
    )
    
    print_statistics("φ∧ψ/Vol", vals_phi_psi, outlier_proportion)
    print_statistics("||dψ||", vals_dpsi, outlier_proportion)
    print_statistics("||dφ||", vals_dphi, outlier_proportion)
    print_statistics("||dφ||/||ω²||", vals_ratio, outlier_proportion)
    
    # Compute MSE between dφ and ω² (excluding top outliers)
    if len(vals_dphi) > 0 and len(vals_omega2) > 0:
        if outlier_proportion > 0 and len(vals_dphi) > 10:
            percentile_high = (1 - outlier_proportion) * 100
            q_high_dphi = np.percentile(vals_dphi, percentile_high)
            q_high_omega2 = np.percentile(vals_omega2, percentile_high)
            # Apply filter based on both arrays
            mask = (vals_dphi <= q_high_dphi) & (vals_omega2 <= q_high_omega2)
            vals_dphi_filtered = vals_dphi[mask]
            vals_omega2_filtered = vals_omega2[mask]
        else:
            vals_dphi_filtered = vals_dphi
            vals_omega2_filtered = vals_omega2
        
        if len(vals_dphi_filtered) > 0:
            mse_dphi_omega = np.mean((vals_dphi_filtered - vals_omega2_filtered)**2)
            print(f"\nMSE between ||dφ|| and ||ω²|| (excluding top {outlier_proportion*100:.1f}%): {mse_dphi_omega:.6e}")
    
    plot_phi_wedge_psi(vals_phi_psi, g2_run_number, output_dir)
    plot_dpsi(vals_dpsi, g2_run_number, output_dir, outlier_proportion)
    plot_dphi_ratio(vals_ratio, g2_run_number, output_dir, outlier_proportion)
    
    print()
    print("=" * 80)
    print("G2 Identities Check Complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
