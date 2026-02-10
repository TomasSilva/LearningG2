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

# DEBUG FLAG - Set to True to enable detailed debugging output
DEBUG = False

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import geometry functions
from geometry.geometry import (
    kahler_form_real_matrix, 
    find_max_dQ_coords
)
from geometry.wedge import wedge
from geometry.numerical_exterior_derivative import (
    sample_numerical_g2_neighborhood_val,
    numerical_d_g2
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

def check_g2_identities(data, g2_models, fmodel, BASIS, n_points=100, epsilon=1e-5, global_rotation_epsilon=1e-5, psi_method='model'):
    """
    Consolidated check for G2 identities using trained model predictions:
    1. φ ∧ ψ = 7·Vol(K_f, g_φ)
    2. dψ = 0
    3. dφ = ω²
    Uses the same neighborhood points for all checks and avoids repeated predictions.
    
    Parameters:
    -----------
    psi_method : str, either 'model' or 'star'
        'model': use 4form model to predict psi directly
        'star': compute psi using Hodge star from phi and metric
    """
    # Prepare input features
    link_points = data['link_points']
    etas = data['etas']
    drop_maxs = data['drop_maxs']
    drop_ones = data['drop_ones']
    base_points = data['base_points'] if 'base_points' in data else None
    rotations = data['rotations'] if 'rotations' in data else None
    # Random sampling if n_points < dataset size
    total_points = len(link_points)
    if n_points < total_points:
        indices = np.random.choice(total_points, size=n_points, replace=False)
    else:
        indices = np.arange(total_points)
        n_points = total_points
    # Storage for results
    vals_phi_wedge_psi = []
    vals_dpsi = []
    vals_dphi = []
    vals_omega2 = []
    vals_ratio = []
    skipped_phi_wedge_psi = 0
    skipped_dpsi = 0
    skipped_dphi = 0
    # For statistics
    phi_norms = []
    metric_dets = []
    psi_norms = []
    
    # Track outlier details
    outlier_info = {
        'dpsi': [],  # (idx, dpsi_norm, phi_norm, psi_norm, link_pt, eta, drop_max, drop_one)
        'dphi': []   # (idx, dphi_ratio, phi_norm, psi_norm, link_pt, eta, drop_max, drop_one)
    }
    
    # For each sampled point, predict phi, psi, and metric, check identities
    for idx in tqdm(indices, desc="G2 identity checks", file=sys.stdout, dynamic_ncols=True):
        # Prepare model input
        link_pt = link_points[idx]
        eta = etas[idx]
        drop_max = drop_maxs[idx]
        drop_one = drop_ones[idx]
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)

        if DEBUG and idx == indices[0]:
            print(f"\n[DEBUG] Base point X_input (idx={idx}):")
            print(f"  link_pt from dataset range: [{link_pt.min():.6e}, {link_pt.max():.6e}]")
            print(f"  eta from dataset range: [{eta.min():.6e}, {eta.max():.6e}]")
            print(f"  drop_max: {drop_max}, drop_one: {drop_one}")

        # Predict (raw normalized outputs)
        phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
        metric_vec_raw = g2_models['metric'].predict(X_input, verbose=0)[0]
        
        # Denormalize phi and metric
        if '3form_y_mean' in g2_models and '3form_y_std' in g2_models:
            y_mean_phi = g2_models['3form_y_mean']
            y_std_phi = g2_models['3form_y_std']
            phi_vec = phi_vec_raw * y_std_phi + y_mean_phi
        else:
            phi_vec = phi_vec_raw
        if 'metric_y_mean' in g2_models and 'metric_y_std' in g2_models:
            y_mean_metric = g2_models['metric_y_mean']
            y_std_metric = g2_models['metric_y_std']
            metric_vec = metric_vec_raw * y_std_metric + y_mean_metric
        else:
            metric_vec = metric_vec_raw
        
        # Get psi based on method
        if psi_method == 'model':
            # Use 4form model
            psi_vec_raw = g2_models['4form'].predict(X_input, verbose=0)[0]
            if '4form_y_mean' in g2_models and '4form_y_std' in g2_models:
                y_mean_psi = g2_models['4form_y_mean']
                y_std_psi = g2_models['4form_y_std']
                psi_vec = psi_vec_raw * y_std_psi + y_mean_psi
            else:
                psi_vec = psi_vec_raw
            
            # Flatten to 1D if needed
            if psi_vec.ndim > 1:
                psi_vec = psi_vec.flatten()
            
            # Expand psi_vec from 23 to 35 dimensions if needed
            if psi_vec.shape[0] == 23 and '4form_nonzero_indices' in g2_models:
                psi_vec_23 = psi_vec
                psi_vec = np.zeros(35)
                psi_vec[g2_models['4form_nonzero_indices']] = psi_vec_23
        
        # Convert to tensor forms
        phi = vec_to_form(phi_vec, n=7, k=3)
        metric = vec_to_metric(metric_vec)
        if metric.ndim == 3:
            metric = metric[0]
        
        if psi_method == 'model':
            psi = vec_to_form(psi_vec, n=7, k=4)
        else:
            # Use Hodge star: psi = star(phi, metric)
            try:
                psi = Hodge_Dual(phi, metric)
            except Exception as e:
                # If Hodge dual fails, skip this point
                if DEBUG:
                    print(f"Hodge dual failed at idx {idx}: {e}")
                continue

        if DEBUG and idx == indices[0]:
            print(f"\n[DEBUG] Base point predictions (idx={idx}):")
            print(f"  phi_vec range: [{phi_vec.min():.6e}, {phi_vec.max():.6e}]")
            if psi_method == 'model':
                print(f"  psi_vec range: [{psi_vec.min():.6e}, {psi_vec.max():.6e}]")
            else:
                print(f"  psi (Hodge star) norm: {np.linalg.norm(psi):.6e}")
            print(f"  metric_vec range: [{metric_vec.min():.6e}, {metric_vec.max():.6e}]")
        
        # Compute statistics for phi, metric, psi
        phi_norms.append(np.linalg.norm(phi))
        metric_dets.append(np.linalg.det(metric))
        try:
            psi_norms.append(np.linalg.norm(psi))
        except Exception:
            psi_norms.append(np.nan)

        # 1. φ ∧ ψ = 7·Vol(K_f, g_φ)
        try:
            det_metric = np.linalg.det(metric)
            if det_metric <= 0 or not np.isfinite(det_metric):
                skipped_phi_wedge_psi += 1
            else:              
                prod = wedge(phi, psi)[0, 1, 2, 3, 4, 5, 6]
                vol = np.sqrt(det_metric)
                val = prod / vol
                if np.isfinite(val):
                    vals_phi_wedge_psi.append(val)
                else:
                    skipped_phi_wedge_psi += 1
        except Exception:
            skipped_phi_wedge_psi += 1

        # 2. dψ = 0 and 3. dφ = ω² (neighborhood checks)
        if base_points is not None and rotations is not None:
            base_point = base_points[idx]
            rotation = rotations[idx]
            
            # Use FIXED patch coordinates from dataset (NOT recomputed!)
            base_drop_max = int(drop_max)
            base_drop_one = int(drop_one)
            
            # Helper function to compute link features for a point with FIXED patch coordinates
            def compute_link_features_fixed_patch(p):
                """
                Compute link features for point p using FIXED patch coordinates from base point.
                This matches the logic in sampling.py.
                
                Parameters
                ----------
                p : ndarray, shape (10,)
                    Point in R^10 (already on CY manifold from quintic_solver)
                    
                Returns
                -------
                tuple : (link_pt, eta)
                    link_pt : ndarray, shape (10,) - normalized link coordinates
                    eta : ndarray, shape (7,) - eta coordinates
                """
                point_cc = p[0:5] + 1.j * p[5:]
                
                if DEBUG and not hasattr(compute_link_features_fixed_patch, 'debug_printed'):
                    compute_link_features_fixed_patch.debug_printed = True
                    print(f"  [DEBUG] compute_link_features_fixed_patch:")
                    print(f"    point_cc before rotation: {point_cc}")
                    print(f"    rotation: {rotation:.6f}")
                    print(f"    base_drop_max: {base_drop_max}, base_drop_one: {base_drop_one}")
                
                # Use FIXED patch coordinates from base point (same as in sampling.py)  
                drop_max = base_drop_max
                drop_one = base_drop_one
                
                # DO NOT normalize by drop_one here! The neighborhood points from quintic_solver
                # are already valid CY points. Normalizing would bring them back toward the base point.
                # point_cc = point_cc / point_cc[drop_one]  # REMOVED - defeats neighborhood exploration
                
                # CRITICAL: Apply rotation BEFORE computing eta (same as sampling.py line 235)
                point_cc = np.exp(1.j * rotation) * point_cc
                
                if DEBUG and not hasattr(compute_link_features_fixed_patch, 'debug_printed2'):
                    compute_link_features_fixed_patch.debug_printed2 = True
                    print(f"    point_cc after norm+rotation: {point_cc}")
                    print(f"    ||point_cc||: {np.linalg.norm(point_cc):.6e}")
                
                # Compute u coordinates (excluding drop_max and drop_one)
                u_coords = [i for i in range(5) if i != drop_max and i != drop_one]
                
                # Compute eta coordinates (same as sampling.py line 232)
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
                
                # Compute link_pt (normalized point on S^9)
                link_pt = np.concatenate([
                    (point_cc / np.linalg.norm(point_cc)).real,
                    (point_cc / np.linalg.norm(point_cc)).imag
                ])
                
                if DEBUG and not hasattr(compute_link_features_fixed_patch, 'debug_printed3'):
                    compute_link_features_fixed_patch.debug_printed3 = True
                    print(f"    eta: {eta}")
                    print(f"    eta range: [{eta.min():.6e}, {eta.max():.6e}]")
                    print(f"    link_pt range: [{link_pt.min():.6e}, {link_pt.max():.6e}]")
                
                return link_pt, eta
            
            # Predict φ at neighborhood points
            def predict_phi_at_point(p):
                """
                Predict phi at point p.
                p is a 10D point on CY manifold from sample_numerical_g2_neighborhood_val.
                """
                link_pt, eta = compute_link_features_fixed_patch(p)
                X_input = np.concatenate([link_pt, eta, [base_drop_max, base_drop_one]])
                X_input = np.expand_dims(X_input, axis=0)
                
                if DEBUG and not hasattr(predict_phi_at_point, 'debug_printed'):
                    predict_phi_at_point.debug_printed = True
                    print(f"  [DEBUG] predict_phi_at_point X_input:")
                    print(f"    link_pt range: [{link_pt.min():.6e}, {link_pt.max():.6e}]")
                    print(f"    eta range: [{eta.min():.6e}, {eta.max():.6e}]")
                    print(f"    drop_max: {base_drop_max}, drop_one: {base_drop_one}")
                    print(f"    Full X_input shape: {X_input.shape}")
                
                phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
                
                if DEBUG and not hasattr(predict_phi_at_point, 'debug_printed2'):
                    predict_phi_at_point.debug_printed2 = True
                    print(f"    y_std_phi shape: {y_std_phi.shape}, range: [{y_std_phi.min():.6e}, {y_std_phi.max():.6e}]")
                    print(f"    y_mean_phi shape: {y_mean_phi.shape}, range: [{y_mean_phi.min():.6e}, {y_mean_phi.max():.6e}]")
                
                if '3form_y_mean' in g2_models and '3form_y_std' in g2_models:
                    phi_vec = phi_vec_raw * y_std_phi + y_mean_phi
                else:
                    phi_vec = phi_vec_raw
                phi_form = vec_to_form(phi_vec, n=7, k=3)
                if DEBUG and not hasattr(predict_phi_at_point, 'debug_printed3'):
                    predict_phi_at_point.debug_printed3 = True
                    print(f"  [DEBUG] predict_phi_at_point:")
                    print(f"    phi_vec_raw range: [{phi_vec_raw.min():.6e}, {phi_vec_raw.max():.6e}]")
                    print(f"    phi_vec denorm range: [{phi_vec.min():.6e}, {phi_vec.max():.6e}]")
                    print(f"    phi_form norm: {np.linalg.norm(phi_form):.6e}")
                return phi_form
            
            def predict_psi_at_point(p):
                """
                Predict psi at point p.
                p is a 10D point on CY manifold from sample_numerical_g2_neighborhood_val.
                """
                link_pt, eta = compute_link_features_fixed_patch(p)
                X_input = np.concatenate([link_pt, eta, [base_drop_max, base_drop_one]])
                X_input = np.expand_dims(X_input, axis=0)
                
                if psi_method == 'model':
                    # Use 4form model directly
                    psi_vec_raw = g2_models['4form'].predict(X_input, verbose=0)[0]
                    if '4form_y_mean' in g2_models and '4form_y_std' in g2_models:
                        psi_vec = psi_vec_raw * y_std_psi + y_mean_psi
                    else:
                        psi_vec = psi_vec_raw
                    # Flatten to 1D if needed
                    if psi_vec.ndim > 1:
                        psi_vec = psi_vec.flatten()
                    # Expand from 23 to 35 dimensions if needed
                    if psi_vec.shape[0] == 23 and '4form_nonzero_indices' in g2_models:
                        psi_vec_expanded = np.zeros(35)
                        psi_vec_expanded[g2_models['4form_nonzero_indices']] = psi_vec
                        psi_vec = psi_vec_expanded
                    psi_form = vec_to_form(psi_vec, n=7, k=4)
                    if DEBUG and not hasattr(predict_psi_at_point, 'debug_printed'):
                        predict_psi_at_point.debug_printed = True
                        print(f"  [DEBUG] predict_psi_at_point (model):")
                        print(f"    psi_vec_raw range: [{psi_vec_raw.min():.6e}, {psi_vec_raw.max():.6e}]")
                        print(f"    psi_vec denorm range: [{psi_vec.min():.6e}, {psi_vec.max():.6e}]")
                        print(f"    psi_form norm: {np.linalg.norm(psi_form):.6e}")
                else:
                    # Use Hodge star: compute phi and metric, then psi = star(phi, metric)
                    phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
                    metric_vec_raw = g2_models['metric'].predict(X_input, verbose=0)[0]
                    
                    if '3form_y_mean' in g2_models and '3form_y_std' in g2_models:
                        phi_vec = phi_vec_raw * y_std_phi + y_mean_phi
                    else:
                        phi_vec = phi_vec_raw
                    if 'metric_y_mean' in g2_models and 'metric_y_std' in g2_models:
                        metric_vec = metric_vec_raw * y_std_metric + y_mean_metric
                    else:
                        metric_vec = metric_vec_raw
                    
                    phi_form = vec_to_form(phi_vec, n=7, k=3)
                    metric_tensor = vec_to_metric(metric_vec)
                    if metric_tensor.ndim == 3:
                        metric_tensor = metric_tensor[0]
                    
                    psi_form = Hodge_Dual(phi_form, metric_tensor)
                    
                    if DEBUG and not hasattr(predict_psi_at_point, 'debug_printed'):
                        predict_psi_at_point.debug_printed = True
                        print(f"  [DEBUG] predict_psi_at_point (star):")
                        print(f"    phi_form norm: {np.linalg.norm(phi_form):.6e}")
                        print(f"    det(metric): {np.linalg.det(metric_tensor):.6e}")
                        print(f"    psi_form norm: {np.linalg.norm(psi_form):.6e}")
                
                return psi_form
            try:
                # dφ check - use sample_numerical_g2_neighborhood_val with rotation
                if DEBUG and idx == indices[0]:
                    print(f"\n[DEBUG] Computing dφ check (idx={idx}):")
                    print(f"  epsilon: {epsilon:.6e}")
                    print(f"  base_point: {base_point[:5]}...")
                
                dic_phi = sample_numerical_g2_neighborhood_val(
                    predict_phi_at_point, base_point, epsilon,
                    find_max_dQ_coords_fn=None,
                    global_rotation_epsilon=global_rotation_epsilon,
                    drop_max=base_drop_max,
                    drop_one=base_drop_one
                )
                
                if DEBUG and idx == indices[0]:
                    print(f"  [DEBUG] dic_phi neighborhood samples:")
                    for key in ['0', '1', '2']:
                        if key in dic_phi and len(dic_phi[key]) > 0:
                            phi_sample = dic_phi[key][0]
                            print(f"    key='{key}': norm={np.linalg.norm(phi_sample):.6e}")
                
                d_phi = numerical_d_g2(dic_phi, epsilon)
                norm_dphi = np.linalg.norm(d_phi)
                
                if DEBUG and idx == indices[0]:
                    print(f"  [DEBUG] d_phi computed:")
                    print(f"    d_phi shape: {d_phi.shape}")
                    print(f"    d_phi range: [{d_phi.min():.6e}, {d_phi.max():.6e}]")
                    print(f"    ||dφ||: {norm_dphi:.6e}")
                
                cy_metric = np.array(fmodel(np.expand_dims(base_point, axis=0))[0])
                w = kahler_form_real_matrix(cy_metric)
                w_R7 = np.pad(w, ((0, 1), (0, 1)), mode='constant')
                w2 = wedge(w_R7, w_R7)
                norm_w2 = np.linalg.norm(w2)
                
                if DEBUG and idx == indices[0]:
                    print(f"    ||ω²||: {norm_w2:.6e}")
                    print(f"    ratio ||dφ||/||ω²||: {norm_dphi/norm_w2:.6e}")
                
                if np.isfinite(norm_dphi) and np.isfinite(norm_w2) and norm_w2 > 0:
                    vals_dphi.append(norm_dphi)
                    vals_omega2.append(norm_w2)
                    ratio = norm_dphi / norm_w2
                    vals_ratio.append(ratio)
                    # Track if this is an outlier (>20)
                    if ratio > 20:
                        outlier_info['dphi'].append({
                            'idx': int(idx),
                            'dphi_ratio': float(ratio),
                            'phi_norm': float(np.linalg.norm(phi_vec)),
                            'psi_norm': float(np.linalg.norm(psi_vec)),
                            'link_pt_range': (float(link_pt.min()), float(link_pt.max())),
                            'eta_range': (float(eta.min()), float(eta.max())),
                            'drop_max': int(drop_max),
                            'drop_one': int(drop_one)
                        })
                else:
                    skipped_dphi += 1
            except Exception as e:
                skipped_dphi += 1
                if skipped_dphi == 1:  # Print first error for debugging
                    print(f"\nWarning: dφ check failed with error: {e}")
            try:
                # dψ check - use sample_numerical_g2_neighborhood_val with rotation
                if DEBUG and idx == indices[0]:
                    print(f"\n[DEBUG] Computing dψ check (idx={idx}):")
                    print(f"  epsilon: {epsilon:.6e}")
                
                dic_psi = sample_numerical_g2_neighborhood_val(
                    predict_psi_at_point, base_point, epsilon,
                    find_max_dQ_coords_fn=None,
                    global_rotation_epsilon=global_rotation_epsilon,
                    drop_max=base_drop_max,
                    drop_one=base_drop_one
                )
                
                if DEBUG and idx == indices[0]:
                    print(f"  [DEBUG] dic_psi neighborhood samples:")
                    for key in ['0', '1', '2']:
                        if key in dic_psi and len(dic_psi[key]) > 0:
                            psi_sample = dic_psi[key][0]
                            print(f"    key='{key}': norm={np.linalg.norm(psi_sample):.6e}")
                
                d_psi = numerical_d_g2(dic_psi, epsilon)
                norm_dpsi = np.linalg.norm(d_psi)
                
                if DEBUG and idx == indices[0]:
                    print(f"  [DEBUG] d_psi computed:")
                    print(f"    d_psi shape: {d_psi.shape}")
                    print(f"    d_psi range: [{d_psi.min():.6e}, {d_psi.max():.6e}]")
                    print(f"    ||dψ||: {norm_dpsi:.6e}")
                
                if np.isfinite(norm_dpsi):
                    vals_dpsi.append(norm_dpsi)
                    # Track if this is an outlier (>20)
                    if norm_dpsi > 20:
                        outlier_info['dpsi'].append({
                            'idx': int(idx),
                            'dpsi_norm': float(norm_dpsi),
                            'phi_norm': float(np.linalg.norm(phi_vec)),
                            'psi_norm': float(np.linalg.norm(psi_vec)),
                            'link_pt_range': (float(link_pt.min()), float(link_pt.max())),
                            'eta_range': (float(eta.min()), float(eta.max())),
                            'drop_max': int(drop_max),
                            'drop_one': int(drop_one)
                        })
                else:
                    skipped_dpsi += 1
            except Exception as e:
                skipped_dpsi += 1
                if skipped_dpsi == 1:  # Print first error for debugging
                    print(f"\nWarning: dψ check failed with error: {e}")

    # Print statistics for ||phi||, det(metric), ||psi||
    phi_norms_arr = np.array(phi_norms)
    metric_dets_arr = np.array(metric_dets)
    psi_norms_arr = np.array(psi_norms)
    print("\n  Statistics for predicted ||phi||:")
    print(f"    min: {np.nanmin(phi_norms_arr):.6e}, mean: {np.nanmean(phi_norms_arr):.6e}, max: {np.nanmax(phi_norms_arr):.6e}")
    print("  Statistics for det(metric):")
    print(f"    min: {np.nanmin(metric_dets_arr):.6e}, mean: {np.nanmean(metric_dets_arr):.6e}, max: {np.nanmax(metric_dets_arr):.6e}")
    print("  Statistics for predicted ||psi||:")
    print(f"    min: {np.nanmin(psi_norms_arr):.6e}, mean: {np.nanmean(psi_norms_arr):.6e}, max: {np.nanmax(psi_norms_arr):.6e}")
      
    # Print torsion statistics
    print(f"\n  φ∧ψ check - Valid: {len(vals_phi_wedge_psi)}/{n_points}, Skipped: {skipped_phi_wedge_psi}")
    print(f"  dψ check - Valid: {len(vals_dpsi)}/{n_points}, Skipped: {skipped_dpsi}")
    print(f"  dφ check - Valid: {len(vals_dphi)}/{n_points}, Skipped: {skipped_dphi}")
    return np.array(vals_phi_wedge_psi), np.array(vals_dpsi), np.array(vals_dphi), np.array(vals_omega2), np.array(vals_ratio), outlier_info

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
    parser.add_argument('--psi-method', type=str, default='model', choices=['model', 'star'],
                       help='Method to compute psi: "model" (use 4form model) or "star" (use Hodge star from phi and metric)')
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
    n_points = len(g2_data['phis']) if args.n_points is None else args.n_points
    
    # Load G2 models
    g2_models = load_g2_models(g2_run_number, script_dir)
    if g2_models is None:
        print("Error: Could not load G2 models")
        sys.exit(1)
    
    # If using star method, remove 4form model requirement
    if args.psi_method == 'star':
        if '4form' in g2_models:
            del g2_models['4form']
        print(f"Using psi_method='star': computing psi via Hodge star from phi and metric")
    else:
        print(f"Using psi_method='model': predicting psi directly from 4form model")
    
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
    
    # Load CY model for derivative checks (needed for ω)
    fmodel, BASIS, cy_data = load_cy_model(cy_run_number, args.cy_data_dir, script_dir)
    
    # Run checks and print statistics
    vals_phi_psi, vals_dpsi, vals_dphi, vals_omega2, vals_ratio, outlier_info = check_g2_identities(
        g2_data, g2_models, fmodel, BASIS, n_points,
        args.epsilon, args.rotation_epsilon, args.psi_method
    )
    
    print_statistics("φ∧ψ/Vol (model predictions)", vals_phi_psi)
    print_statistics("||dψ|| (model predictions)", vals_dpsi)
    print_statistics("||dφ|| (model predictions)", vals_dphi)
    print_statistics("||dφ||/||ω²|| (model predictions)", vals_ratio)
    
    # Compute MSE between dφ and ω²
    if len(vals_dphi) > 0 and len(vals_omega2) > 0:
        mse_dphi_omega = np.mean((vals_dphi - vals_omega2)**2)
        print(f"\nMSE between ||dφ|| and ||ω²||: {mse_dphi_omega:.6e}")
    
    # Print outlier analysis
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS (values > 20)")
    print("="*80)
    
    print(f"\n||dψ|| > 20 outliers: {len(outlier_info['dpsi'])}")
    if len(outlier_info['dpsi']) > 0:
        print("\nDetails of ||dψ|| outliers:")
        for i, info in enumerate(outlier_info['dpsi'][:5]):  # Show first 5
            print(f"\n  Outlier {i+1} (idx={info['idx']}):")
            print(f"    ||dψ||: {info['dpsi_norm']:.2f}")
            print(f"    ||φ||: {info['phi_norm']:.6f}")
            print(f"    ||ψ||: {info['psi_norm']:.6f}")
            print(f"    link_pt range: [{info['link_pt_range'][0]:.6f}, {info['link_pt_range'][1]:.6f}]")
            print(f"    eta range: [{info['eta_range'][0]:.6f}, {info['eta_range'][1]:.6f}]")
            print(f"    drop_max: {info['drop_max']}, drop_one: {info['drop_one']}")
        if len(outlier_info['dpsi']) > 5:
            print(f"  ... and {len(outlier_info['dpsi'])-5} more")
    
    print(f"\n||dφ||/||ω²|| > 20 outliers: {len(outlier_info['dphi'])}")
    if len(outlier_info['dphi']) > 0:
        print("\nDetails of ||dφ||/||ω²|| outliers:")
        for i, info in enumerate(outlier_info['dphi'][:5]):  # Show first 5
            print(f"\n  Outlier {i+1} (idx={info['idx']}):")
            print(f"    ||dφ||/||ω²||: {info['dphi_ratio']:.2f}")
            print(f"    ||φ||: {info['phi_norm']:.6f}")
            print(f"    ||ψ||: {info['psi_norm']:.6f}")
            print(f"    link_pt range: [{info['link_pt_range'][0]:.6f}, {info['link_pt_range'][1]:.6f}]")
            print(f"    eta range: [{info['eta_range'][0]:.6f}, {info['eta_range'][1]:.6f}]")
            print(f"    drop_max: {info['drop_max']}, drop_one: {info['drop_one']}")
        if len(outlier_info['dphi']) > 5:
            print(f"  ... and {len(outlier_info['dphi'])-5} more")
    
    # Check overlap
    dpsi_indices = set(info['idx'] for info in outlier_info['dpsi'])
    dphi_indices = set(info['idx'] for info in outlier_info['dphi'])
    overlap = dpsi_indices & dphi_indices
    print(f"\n  Points with BOTH outliers: {len(overlap)}")
    if len(overlap) > 0:
        print(f"    Indices: {sorted(list(overlap))[:10]}" + (" ..." if len(overlap) > 10 else ""))
    
    plot_phi_wedge_psi(vals_phi_psi, g2_run_number, output_dir)
    plot_dphi_ratio(vals_ratio, g2_run_number, output_dir)
    plot_dpsi(vals_dpsi, g2_run_number, output_dir)

    print()
    print("=" * 80)
    print("G2 Identities Check (MODELS) Complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
