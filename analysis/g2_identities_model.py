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
    X = np.concatenate([link_points, etas, drop_maxs, drop_ones], axis=1)
    
    results = {}
    
    # Evaluate 3form model
    Y_true_3form = g2_data['phis']
    Y_pred_3form_raw = g2_models['3form'].predict(X, verbose=0)
    
    # Denormalize predictions if model has normalization metadata
    if '3form_y_mean' in g2_models and '3form_y_std' in g2_models:
        y_mean_phi = g2_models['3form_y_mean']
        y_std_phi = g2_models['3form_y_std']
        Y_pred_3form = Y_pred_3form_raw * y_std_phi + y_mean_phi
        
        # Print normalization diagnostics
        print(f"\n  3form normalization statistics:")
        print(f"    y_mean range: [{np.min(y_mean_phi):.6e}, {np.max(y_mean_phi):.6e}]")
        print(f"    y_std range: [{np.min(y_std_phi):.6e}, {np.max(y_std_phi):.6e}]")
        print(f"    y_std mean: {np.mean(y_std_phi):.6e}")
        print(f"    y_std median: {np.median(y_std_phi):.6e}")
        print(f"    Raw prediction range: [{np.min(Y_pred_3form_raw):.6e}, {np.max(Y_pred_3form_raw):.6e}]")
        print(f"    Denormalized prediction range: [{np.min(Y_pred_3form):.6e}, {np.max(Y_pred_3form):.6e}]")
        print(f"    True data range: [{np.min(Y_true_3form):.6e}, {np.max(Y_true_3form):.6e}]")
    else:
        Y_pred_3form = Y_pred_3form_raw
        print(f"  Warning: 3form model has no normalization metadata!")
    
    mse_3form = np.mean((Y_true_3form - Y_pred_3form)**2)
    mae_3form = np.mean(np.abs(Y_true_3form - Y_pred_3form))
    results['3form'] = mse_3form
    print(f"  3form model MSE: {mse_3form:.6e}")
    print(f"  3form model MAE: {mae_3form:.6e}")
    print(f"  3form relative error (MAE/mean_abs_true): {mae_3form / np.mean(np.abs(Y_true_3form)):.6e}")
    
    # Evaluate metric model
    Y_true_metric = g2_data['g2_metrics']
    Y_pred_metric_raw = g2_models['metric'].predict(X, verbose=0)
    # Denormalize predictions if model has normalization metadata
    if 'metric_y_mean' in g2_models and 'metric_y_std' in g2_models:
        y_mean_metric = g2_models['metric_y_mean']
        y_std_metric = g2_models['metric_y_std']
        Y_pred_metric = Y_pred_metric_raw * y_std_metric + y_mean_metric
        
        # Print normalization diagnostics
        print(f"\n  metric normalization statistics:")
        print(f"    y_mean range: [{np.min(y_mean_metric):.6e}, {np.max(y_mean_metric):.6e}]")
        print(f"    y_std range: [{np.min(y_std_metric):.6e}, {np.max(y_std_metric):.6e}]")
        print(f"    y_std mean: {np.mean(y_std_metric):.6e}")
        print(f"    y_std median: {np.median(y_std_metric):.6e}")
        print(f"    Raw prediction range: [{np.min(Y_pred_metric_raw):.6e}, {np.max(Y_pred_metric_raw):.6e}]")
        print(f"    Denormalized prediction range: [{np.min(Y_pred_metric):.6e}, {np.max(Y_pred_metric):.6e}]")
        print(f"    True data range: [{np.min(Y_true_metric):.6e}, {np.max(Y_true_metric):.6e}]")
    else:
        Y_pred_metric = Y_pred_metric_raw
        print(f"  Warning: metric model has no normalization metadata!")
    
    mse_metric = np.mean((Y_true_metric - Y_pred_metric)**2)
    mae_metric = np.mean(np.abs(Y_true_metric - Y_pred_metric))
    results['metric'] = mse_metric
    print(f"  Metric model MSE: {mse_metric:.6e}")
    print(f"  Metric model MAE: {mae_metric:.6e}")
    print(f"  Metric relative error (MAE/mean_abs_true): {mae_metric / np.mean(np.abs(Y_true_metric)):.6e}")
    
    # Check a single point in detail to understand geometric error propagation
    print(f"\n  Single-point diagnostic (first test sample):")
    idx = 0
    phi_vec_true = Y_true_3form[idx]
    phi_vec_pred = Y_pred_3form[idx]
    metric_vec_true = Y_true_metric[idx]
    metric_vec_pred = Y_pred_metric[idx]
    
    from geometry.compression import vec_to_form, vec_to_metric
    phi_true_tensor = vec_to_form(phi_vec_true, n=7, k=3)
    phi_pred_tensor = vec_to_form(phi_vec_pred, n=7, k=3)
    metric_true_tensor = vec_to_metric(metric_vec_true)
    metric_pred_tensor = vec_to_metric(metric_vec_pred)
    
    print(f"    ||phi_true||: {np.linalg.norm(phi_true_tensor):.6e}")
    print(f"    ||phi_pred||: {np.linalg.norm(phi_pred_tensor):.6e}")
    print(f"    ||phi_error||: {np.linalg.norm(phi_true_tensor - phi_pred_tensor):.6e}")
    print(f"    phi relative error: {np.linalg.norm(phi_true_tensor - phi_pred_tensor) / np.linalg.norm(phi_true_tensor):.6e}")
    
    print(f"    det(metric_true): {np.linalg.det(metric_true_tensor):.6e}")
    print(f"    det(metric_pred): {np.linalg.det(metric_pred_tensor):.6e}")
    print(f"    det relative error: {np.abs(np.linalg.det(metric_true_tensor) - np.linalg.det(metric_pred_tensor)) / np.abs(np.linalg.det(metric_true_tensor)):.6e}")
    
    # Check component-wise vs geometric error
    print(f"    Component-wise phi MAE: {np.mean(np.abs(phi_vec_true - phi_vec_pred)):.6e}")
    print(f"    Geometric phi error: {np.linalg.norm(phi_true_tensor - phi_pred_tensor):.6e}")
    print(f"    Error amplification factor: {np.linalg.norm(phi_true_tensor - phi_pred_tensor) / np.mean(np.abs(phi_vec_true - phi_vec_pred)):.6e}")
    
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
    volumes = []
    prods = []
    skipped = 0
    
    # Diagnostic: Check normalization status and print first point details
    has_phi_norm = '3form_y_mean' in g2_models and '3form_y_std' in g2_models
    has_metric_norm = 'metric_y_mean' in g2_models and 'metric_y_std' in g2_models
    
    if has_phi_norm:
        y_mean_phi_global = g2_models['3form_y_mean']
        y_std_phi_global = g2_models['3form_y_std']
        print(f"\n  Normalization metadata found for 3form model:")
        print(f"    y_mean range: [{np.min(y_mean_phi_global):.6e}, {np.max(y_mean_phi_global):.6e}]")
        print(f"    y_std range: [{np.min(y_std_phi_global):.6e}, {np.max(y_std_phi_global):.6e}]")
    else:
        print(f"\n  WARNING: No normalization metadata for 3form model!")
        y_mean_phi_global = None
        y_std_phi_global = None
    
    if has_metric_norm:
        y_mean_metric_global = g2_models['metric_y_mean']
        y_std_metric_global = g2_models['metric_y_std']
        print(f"  Normalization metadata found for metric model:")
        print(f"    y_mean range: [{np.min(y_mean_metric_global):.6e}, {np.max(y_mean_metric_global):.6e}]")
        print(f"    y_std range: [{np.min(y_std_metric_global):.6e}, {np.max(y_std_metric_global):.6e}]")
    else:
        print(f"  WARNING: No normalization metadata for metric model!")
        y_mean_metric_global = None
        y_std_metric_global = None
    
    # Diagnostic: Check first point in detail
    if len(indices) > 0:
        idx0 = indices[0]
        link_pt0 = link_points[idx0]
        eta0 = etas[idx0]
        drop_max0 = drop_maxs[idx0]
        drop_one0 = drop_ones[idx0]
        X_input0 = np.concatenate([link_pt0, eta0, [drop_max0, drop_one0]])
        X_input0 = np.expand_dims(X_input0, axis=0)
        
        phi_vec_raw0 = g2_models['3form'].predict(X_input0, verbose=0)[0]
        metric_vec_raw0 = g2_models['metric'].predict(X_input0, verbose=0)[0]
        
        phi_vec_true0 = data['phis'][idx0] if 'phis' in data else None
        metric_vec_true0 = data['g2_metrics'][idx0] if 'g2_metrics' in data else None
        
        print(f"\n  First point diagnostic (index {idx0}):")
        print(f"    Raw phi prediction range: [{np.min(phi_vec_raw0):.6e}, {np.max(phi_vec_raw0):.6e}]")
        print(f"    Raw metric prediction range: [{np.min(metric_vec_raw0):.6e}, {np.max(metric_vec_raw0):.6e}]")
        
        if has_phi_norm:
            phi_vec_denorm0 = phi_vec_raw0 * y_std_phi_global + y_mean_phi_global
            print(f"    Denormalized phi range: [{np.min(phi_vec_denorm0):.6e}, {np.max(phi_vec_denorm0):.6e}]")
            if phi_vec_true0 is not None:
                print(f"    True phi range: [{np.min(phi_vec_true0):.6e}, {np.max(phi_vec_true0):.6e}]")
                print(f"    Phi MAE (denorm vs true): {np.mean(np.abs(phi_vec_denorm0 - phi_vec_true0)):.6e}")
        
        if has_metric_norm:
            metric_vec_denorm0 = metric_vec_raw0 * y_std_metric_global + y_mean_metric_global
            print(f"    Denormalized metric range: [{np.min(metric_vec_denorm0):.6e}, {np.max(metric_vec_denorm0):.6e}]")
            if metric_vec_true0 is not None:
                print(f"    True metric range: [{np.min(metric_vec_true0):.6e}, {np.max(metric_vec_true0):.6e}]")
                print(f"    Metric MAE (denorm vs true): {np.mean(np.abs(metric_vec_denorm0 - metric_vec_true0)):.6e}")
    
    for idx in tqdm(indices, desc="φ∧ψ check (model)", file=sys.stdout, dynamic_ncols=True):
        # Prepare model input: link_point (10) + eta (7) + patch indices (2) = 19
        link_pt = link_points[idx]
        eta = etas[idx]
        drop_max = drop_maxs[idx]
        drop_one = drop_ones[idx]
        
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        
        # Predict φ and G2 metric (raw normalized outputs)
        phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
        metric_vec_raw = g2_models['metric'].predict(X_input, verbose=0)[0]

        # Denormalize predictions if models have normalization metadata
        if has_phi_norm:
            phi_vec = phi_vec_raw * y_std_phi_global + y_mean_phi_global
        else:
            phi_vec = phi_vec_raw

        if has_metric_norm:
            metric_vec = metric_vec_raw * y_std_metric_global + y_mean_metric_global
        else:
            metric_vec = metric_vec_raw

        # Convert vector outputs to tensor forms
        phi = vec_to_form(phi_vec, n=7, k=3)
        # If using Cholesky, decode here:
        # from geometry.compression import vec_to_metric_cholesky
        # metric = vec_to_metric_cholesky(metric_vec)
        metric = vec_to_metric(metric_vec)
        # Ensure metric is 2D (squeeze batch dimension if present)
        if metric.ndim == 3:
            metric = metric[0]
        
        # Check for valid metric determinant
        det_metric = np.linalg.det(metric)
        if det_metric <= 0 or not np.isfinite(det_metric):
            skipped += 1
            continue
        
        # Check metric condition number (ill-conditioned metrics cause numerical issues)
        eigvals = np.linalg.eigvalsh(metric)
        min_eigval = np.min(eigvals)
        max_eigval = np.max(eigvals)
        cond_number = max_eigval / min_eigval if min_eigval > 0 else np.inf
        
        # Diagnostic: Check metric quality (only for first few points)
        if idx == indices[0] or (len(vals) == 0 and idx == indices[1]):
            print(f"    Point {idx} metric diagnostics:")
            print(f"      det(metric): {det_metric:.6e}")
            print(f"      min eigval: {min_eigval:.6e}")
            print(f"      max eigval: {max_eigval:.6e}")
            print(f"      condition number: {cond_number:.6e}")
            if 'g2_metrics' in data and idx == indices[0]:
                metric_vec_true_check = data['g2_metrics'][idx]
                metric_true_tensor = vec_to_metric(metric_vec_true_check)
                # Ensure metric is 2D (squeeze batch dimension if present)
                if metric_true_tensor.ndim == 3:
                    metric_true_tensor = metric_true_tensor[0]
                det_metric_true = np.linalg.det(metric_true_tensor)
                eigvals_true = np.linalg.eigvalsh(metric_true_tensor)
                cond_number_true = np.max(eigvals_true) / np.min(eigvals_true) if np.min(eigvals_true) > 0 else np.inf
                print(f"      True det(metric): {det_metric_true:.6e}")
                print(f"      True condition number: {cond_number_true:.6e}")
                print(f"      det ratio (pred/true): {det_metric / det_metric_true:.6e}")
        
        # Skip if metric is too ill-conditioned (could cause numerical issues in Hodge dual)
        # Note: True metrics also have condition numbers ~10^9, so we can't filter too aggressively
        if min_eigval < 1e-15 or cond_number > 1e15:
            skipped += 1
            continue
        
        # Compute ψ = ⋆_{g_G2} φ using the Hodge star operator
        # For ill-conditioned metrics, the standard Hodge_Dual may be unstable
        # We'll try it and catch numerical errors
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            try:
                psi = Hodge_Dual(phi, metric)
                # Check if result is reasonable
                psi_norm = np.linalg.norm(psi)
                if not np.isfinite(psi_norm) or psi_norm > 1e10:
                    # Result is clearly wrong (too large or NaN)
                    skipped += 1
                    continue
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
                # Numerical error in Hodge dual computation
                skipped += 1
                continue
        
        prod = wedge(phi, psi)[0, 1, 2, 3, 4, 5, 6]
        vol = np.sqrt(det_metric)
        # Check for numerical issues
        if not np.isfinite(prod) or not np.isfinite(vol) or abs(vol) < 1e-20:
            skipped += 1
            continue
        val = prod / vol
        # Skip if result is NaN or infinite
        if not np.isfinite(val):
            skipped += 1
            continue
        vals.append(val)
        volumes.append(vol)
        prods.append(prod)
    
    # Print statistics
    total_checked = len(indices)
    valid_count = len(vals)
    skip_proportion = skipped / total_checked * 100 if total_checked > 0 else 0
    print(f"\n  Valid points: {valid_count}/{total_checked} ({100-skip_proportion:.2f}%)")
    print(f"  Skipped (invalid): {skipped}/{total_checked} ({skip_proportion:.2f}%)")
    if valid_count > 0:
        volumes_arr = np.array(volumes)
        prods_arr = np.array(prods)
        print("\n  Volume statistics:")
        print(f"    min: {np.min(volumes_arr):.6e}")
        print(f"    max: {np.max(volumes_arr):.6e}")
        print(f"    mean: {np.mean(volumes_arr):.6e}")
        print(f"    std: {np.std(volumes_arr):.6e}")
        print("\n  Prod statistics:")
        print(f"    min: {np.min(prods_arr):.6e}")
        print(f"    max: {np.max(prods_arr):.6e}")
        print(f"    mean: {np.mean(prods_arr):.6e}")
        print(f"    std: {np.std(prods_arr):.6e}")
    return np.array(vals)


def check_d_psi_and_d_phi(data, g2_models, fmodel, BASIS, n_points=100, 
                          epsilon=1e-5, global_rotation_epsilon=1e-5):
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
    vals_omega2 = []
    vals_ratio = []
    
    def compute_link_features_fixed_patch(p, drop_max, drop_one, rotation=0):
        """Compute link_point and eta from base_point p using FIXED drop_max and drop_one.
        
        p is the base_point (10d), drop_max and drop_one come from the dataset.
        eta is computed from p's coordinates using the fixed patch indices.
        """
        point_cc = p[0:5] + 1.j * p[5:]
        # Apply rotation
        point_cc = np.exp(1.j * rotation) * point_cc
        # Compute eta using FIXED dataset patch indices with current point's coordinates
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
        """Predict φ at a point using trained 3form model.
        
        p: base_point (10d)
        drop_max, drop_one: fixed patch indices from dataset
        rotation: global phase rotation
        """
        link_pt, eta = compute_link_features_fixed_patch(p, drop_max, drop_one, rotation)
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
        # Denormalize if model has normalization metadata
        if '3form_y_mean' in g2_models and '3form_y_std' in g2_models:
            phi_vec = phi_vec_raw * g2_models['3form_y_std'] + g2_models['3form_y_mean']
        else:
            phi_vec = phi_vec_raw
        phi = vec_to_form(phi_vec, n=7, k=3)
        return phi
    
    def predict_psi_at_point(p, drop_max, drop_one, rotation=0):
        """Predict ψ at a point using trained models and Hodge star.
        
        p: base_point (10d)
        drop_max, drop_one: fixed patch indices from dataset
        rotation: global phase rotation
        """
        link_pt, eta = compute_link_features_fixed_patch(p, drop_max, drop_one, rotation)
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
        metric_vec_raw = g2_models['metric'].predict(X_input, verbose=0)[0]
        # Denormalize if models have normalization metadata
        if '3form_y_mean' in g2_models and '3form_y_std' in g2_models:
            phi_vec = phi_vec_raw * g2_models['3form_y_std'] + g2_models['3form_y_mean']
        else:
            phi_vec = phi_vec_raw
        if 'metric_y_mean' in g2_models and 'metric_y_std' in g2_models:
            metric_vec = metric_vec_raw * g2_models['metric_y_std'] + g2_models['metric_y_mean']
        else:
            metric_vec = metric_vec_raw
        phi = vec_to_form(phi_vec, n=7, k=3)
        metric = vec_to_metric(metric_vec)
        # Ensure metric is 2D (squeeze batch dimension if present)
        if metric.ndim == 3:
            metric = metric[0]
        # Check tensor shapes
        if phi.shape != (7, 7, 7):
            raise ValueError(f"phi has wrong shape: {phi.shape}, expected (7,7,7)")
        if metric.shape != (7, 7):
            raise ValueError(f"metric has wrong shape: {metric.shape}, expected (7,7)")
        # Check metric validity before Hodge dual
        det_metric = np.linalg.det(metric)
        if det_metric <= 0 or not np.isfinite(det_metric):
            raise ValueError(f"Invalid metric determinant: {det_metric}")
        psi = Hodge_Dual(phi, metric)
        # Check result validity
        if not np.all(np.isfinite(psi)):
            raise ValueError("Hodge dual produced non-finite values")
        if psi.shape != (7, 7, 7, 7):
            raise ValueError(f"psi has wrong shape: {psi.shape}, expected (7,7,7,7)")
        return psi
    
    find_max_fn = lambda point_cc: find_max_dQ_coords(point_cc, BASIS)
    
    skipped_dpsi = 0
    skipped_dphi = 0
    
    for idx in tqdm(indices, desc="dψ and dφ check (model)", file=sys.stdout, dynamic_ncols=True):
        base_point = base_points[idx]
        rotation = rotations[idx]
        drop_max = int(drop_maxs[idx])
        drop_one = int(drop_ones[idx])
        
        # Check dψ using model predictions in neighborhood
        # drop_max and drop_one are FIXED for all neighborhood points
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                dic_psi = sample_numerical_g2_neighborhood_val(
                    lambda p: predict_psi_at_point(p, drop_max, drop_one, rotation),
                    base_point, epsilon,
                    drop_max=drop_max, drop_one=drop_one
                )
                d_psi = numerical_d_g2(dic_psi, epsilon)
                norm_dpsi = np.linalg.norm(d_psi)
                if np.isfinite(norm_dpsi):
                    vals_dpsi.append(norm_dpsi)
                else:
                    skipped_dpsi += 1
        except Exception as e:
            skipped_dpsi += 1
            # Print first few errors for debugging
            if skipped_dpsi <= 3:
                print(f"\n  Warning: Skipped dψ at idx {idx}: {type(e).__name__}: {str(e)[:100]}")
        
        # Check dφ using model predictions in neighborhood
        # drop_max and drop_one are FIXED for all neighborhood points
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
                if np.isfinite(norm_dphi) and np.isfinite(norm_w2) and norm_w2 > 0:
                    vals_dphi.append(norm_dphi)
                    vals_omega2.append(norm_w2)
                    vals_ratio.append(norm_dphi / norm_w2)
                else:
                    skipped_dphi += 1
        except:
            skipped_dphi += 1
    
    # Print statistics
    total_checked = len(indices)
    valid_dpsi = len(vals_dpsi)
    valid_dphi = len(vals_dphi)
    
    print(f"\n  dψ check - Valid: {valid_dpsi}/{total_checked} ({valid_dpsi/total_checked*100:.2f}%), "
          f"Skipped: {skipped_dpsi} ({skipped_dpsi/total_checked*100:.2f}%)")
    print(f"  dφ check - Valid: {valid_dphi}/{total_checked} ({valid_dphi/total_checked*100:.2f}%), "
          f"Skipped: {skipped_dphi} ({skipped_dphi/total_checked*100:.2f}%)")
    
    return np.array(vals_dpsi), np.array(vals_dphi), np.array(vals_omega2), np.array(vals_ratio)





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
    vals_dpsi, vals_dphi, vals_omega2, vals_ratio = check_d_psi_and_d_phi(
        g2_data, g2_models, fmodel, BASIS, n_points,
        args.epsilon, args.rotation_epsilon
    )
    
    print_statistics("||dψ|| (model predictions)", vals_dpsi)
    print_statistics("||dφ|| (model predictions)", vals_dphi)
    print_statistics("||dφ||/||ω²|| (model predictions)", vals_ratio)
    
    # Compute MSE between dφ and ω²
    if len(vals_dphi) > 0 and len(vals_omega2) > 0:
        mse_dphi_omega = np.mean((vals_dphi - vals_omega2)**2)
        print(f"\nMSE between ||dφ|| and ||ω²||: {mse_dphi_omega:.6e}")
    
    plot_dpsi(vals_dpsi, g2_run_number, output_dir)
    plot_dphi_ratio(vals_ratio, g2_run_number, output_dir)
    
    print()
    print("=" * 80)
    print("G2 Identities Check (MODELS) Complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
