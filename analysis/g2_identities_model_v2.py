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

def check_g2_identities(data, g2_models, fmodel, BASIS, n_points=100, epsilon=1e-5, global_rotation_epsilon=1e-5):
    """
    Consolidated check for G2 identities using trained model predictions:
    1. φ ∧ ψ = 7·Vol(K_f, g_φ)
    2. dψ = 0
    3. dφ = ω²
    Uses the same neighborhood points for all checks and avoids repeated predictions.
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
    vals_ratio = []
    skipped_phi_wedge_psi = 0
    skipped_dpsi = 0
    skipped_dphi = 0
    # For statistics
    phi_norms = []
    metric_dets = []
    psi_norms = []
    # For each sampled point, predict φ and metric, check identities
    for idx in tqdm(indices, desc="G2 identity checks", file=sys.stdout, dynamic_ncols=True):
        # Prepare model input
        link_pt = link_points[idx]
        eta = etas[idx]
        drop_max = drop_maxs[idx]
        drop_one = drop_ones[idx]
        X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
        X_input = np.expand_dims(X_input, axis=0)
        # Predict φ and metric (raw normalized outputs)
        phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
        metric_vec_raw = g2_models['metric'].predict(X_input, verbose=0)[0]
        # Denormalize
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
        # Convert to tensor forms
        phi = vec_to_form(phi_vec, n=7, k=3)
        metric = vec_to_metric(metric_vec)
        # Compute statistics for phi, metric, psi
        phi_norms.append(np.linalg.norm(phi))
        metric_dets.append(np.linalg.det(metric))
        
        # Predict psi directly from 4form model
        psi_vec_raw = g2_models['4form'].predict(X_input, verbose=0)[0]
        if '4form_y_mean' in g2_models and '4form_y_std' in g2_models:
            y_mean_psi = g2_models['4form_y_mean']
            y_std_psi = g2_models['4form_y_std']
            psi_vec = psi_vec_raw * y_std_psi + y_mean_psi
        else:
            psi_vec = psi_vec_raw
        
        # Expand from 23 to 35 dimensions if needed
        if psi_vec.shape[0] == 23 and '4form_nonzero_indices' in g2_models:
            psi_vec_expanded = np.zeros(35)
            psi_vec_expanded[g2_models['4form_nonzero_indices']] = psi_vec
            psi_vec = psi_vec_expanded
        
        try:
            psi = vec_to_form(psi_vec, n=7, k=4)
            psi_norms.append(np.linalg.norm(psi))
        except Exception:
            psi_norms.append(np.nan)

        # 1. φ ∧ ψ = 7·Vol(K_f, g_φ)
        try:
            det_metric = np.linalg.det(metric)
            if det_metric <= 0 or not np.isfinite(det_metric):
                skipped_phi_wedge_psi += 1
            else:
                # Predict psi from 4form model
                psi_vec_raw = g2_models['4form'].predict(X_input, verbose=0)[0]
                if '4form_y_mean' in g2_models and '4form_y_std' in g2_models:
                    y_mean_psi = g2_models['4form_y_mean']
                    y_std_psi = g2_models['4form_y_std']
                    psi_vec = psi_vec_raw * y_std_psi + y_mean_psi
                else:
                    psi_vec = psi_vec_raw
                # Expand from 23 to 35 dimensions if needed
                if psi_vec.shape[0] == 23 and '4form_nonzero_indices' in g2_models:
                    psi_vec_expanded = np.zeros(35)
                    psi_vec_expanded[g2_models['4form_nonzero_indices']] = psi_vec
                    psi_vec = psi_vec_expanded
                psi = vec_to_form(psi_vec, n=7, k=4)
                
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
            # Compute link features for neighborhood
            def compute_link_features(p, rotation=0):
                point_cc = p[0:5] + 1.j * p[5:]
                drop_max = int(np.argmax(np.abs(point_cc)))
                drop_one = int(np.argmin(np.abs(point_cc - 1)))
                point_cc = np.exp(1.j * rotation) * point_cc
                u_coords = [i for i in range(5) if i != drop_max and i != drop_one]
                eta = np.zeros(7)
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
                return link_pt, eta, drop_max, drop_one
            # Predict φ and ψ at neighborhood points
            def predict_phi_at_point(p, rotation=0):
                link_pt, eta, drop_max, drop_one = compute_link_features(p, rotation)
                X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
                X_input = np.expand_dims(X_input, axis=0)
                phi_vec_raw = g2_models['3form'].predict(X_input, verbose=0)[0]
                if '3form_y_mean' in g2_models and '3form_y_std' in g2_models:
                    phi_vec = phi_vec_raw * y_std_phi + y_mean_phi
                else:
                    phi_vec = phi_vec_raw
                return vec_to_form(phi_vec, n=7, k=3)
            def predict_psi_at_point(p, rotation=0):
                link_pt, eta, drop_max, drop_one = compute_link_features(p, rotation)
                X_input = np.concatenate([link_pt, eta, [drop_max, drop_one]])
                X_input = np.expand_dims(X_input, axis=0)
                # Use 4form model directly
                psi_vec_raw = g2_models['4form'].predict(X_input, verbose=0)[0]
                if '4form_y_mean' in g2_models and '4form_y_std' in g2_models:
                    psi_vec = psi_vec_raw * y_std_psi + y_mean_psi
                else:
                    psi_vec = psi_vec_raw
                # Expand from 23 to 35 dimensions if needed
                if psi_vec.shape[0] == 23 and '4form_nonzero_indices' in g2_models:
                    psi_vec_expanded = np.zeros(35)
                    psi_vec_expanded[g2_models['4form_nonzero_indices']] = psi_vec
                    psi_vec = psi_vec_expanded
                return vec_to_form(psi_vec, n=7, k=4)
            try:
                # dφ check
                dic_phi = sample_numerical_g2_neighborhood_val(
                    lambda p: predict_phi_at_point(p, rotation), base_point, epsilon,
                    find_max_dQ_coords_fn=None,
                    global_rotation_epsilon=global_rotation_epsilon
                )
                d_phi = numerical_d_g2(dic_phi, epsilon)
                norm_dphi = np.linalg.norm(d_phi)
                cy_metric = np.array(fmodel(np.expand_dims(base_point, axis=0))[0])
                w = kahler_form_real_matrix(cy_metric)
                w_R7 = np.pad(w, ((0, 1), (0, 1)), mode='constant')
                w2 = wedge(w_R7, w_R7)
                norm_w2 = np.linalg.norm(w2)
                if np.isfinite(norm_dphi) and np.isfinite(norm_w2) and norm_w2 > 0:
                    vals_dphi.append(norm_dphi)
                    vals_ratio.append(norm_dphi / norm_w2)
                else:
                    skipped_dphi += 1
            except Exception:
                skipped_dphi += 1
            try:
                # dψ check
                dic_psi = sample_numerical_g2_neighborhood_val(
                    lambda p: predict_psi_at_point(p, rotation), base_point, epsilon,
                    find_max_dQ_coords_fn=None,
                    global_rotation_epsilon=global_rotation_epsilon
                )
                d_psi = numerical_d_g2(dic_psi, epsilon)
                norm_dpsi = np.linalg.norm(d_psi)
                if np.isfinite(norm_dpsi):
                    vals_dpsi.append(norm_dpsi)
                else:
                    skipped_dpsi += 1
            except Exception:
                skipped_dpsi += 1

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
    return np.array(vals_phi_wedge_psi), np.array(vals_dpsi), np.array(vals_dphi), np.array(vals_ratio)

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
    n_points = len(g2_data['phis']) if args.n_points is None else args.n_points
    
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
    
    # Load CY model for derivative checks (needed for ω)
    fmodel, BASIS, cy_data = load_cy_model(cy_run_number, args.cy_data_dir, script_dir)
    
    # Run checks and print statistics
    vals_phi_psi, vals_dpsi, vals_dphi, vals_ratio = check_g2_identities(
        g2_data, g2_models, fmodel, BASIS, n_points,
        args.epsilon, args.rotation_epsilon
    )
    
    print_statistics("φ∧ψ/Vol (model predictions)", vals_phi_psi)
    print_statistics("||dψ|| (model predictions)", vals_dpsi)
    print_statistics("||dφ|| (model predictions)", vals_dphi)
    print_statistics("||dφ||/||ω²|| (model predictions)", vals_ratio)
    
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
