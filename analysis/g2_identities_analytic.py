#!/usr/bin/env python3
"""
Check G2 identities on learned G2 structures.

This script verifies the following G2 identities (see https://arxiv.org/pdf/math/0702077):
1. φ ∧ ψ = 7·Vol(K_f, g_φ)
2. dψ = 0
3. dφ = ω²

where φ is the G2 3-form, ψ = ⋆_φ φ is the dual 4-form, and ω is the Kähler form
on the base CY manifold lifted to the link K_f (S¹ bundle over the CY).
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

# Import analysis utilities
from analysis.utils import (

    get_most_recent_cy_run_number,
    load_cy_model,
    load_g2_data,
    print_statistics
)


def sampler_g2_package(p, fmodel, BASIS):
    """Sample G2 structure (φ and ψ) at a point."""
    point_cc = p[0:5] + 1.j*p[5:]
    drop_max = int(find_max_dQ_coords(point_cc, BASIS))
    drop_one = int(np.argmin(np.abs(point_cc - 1)))
    
    # Get Kähler form from CY metric
    w = kahler_form_real_matrix(np.array(fmodel(np.expand_dims(p, axis=0))[0]))
    w_R7 = np.pad(w, ((0, 1), (0, 1)), mode='constant')
    
    # Holomorphic volume form
    holomorphic_volume_form = 1 / (5 * point_cc[drop_max]**4)
    ReOmega, ImOmega = holomorphic_volume_real_imag(holomorphic_volume_form)
    ReOmega_R7 = np.pad(ReOmega, ((0, 1), (0, 1), (0, 1)), mode='constant')
    ImOmega_R7 = np.pad(ImOmega, ((0, 1), (0, 1), (0, 1)), mode='constant')
    
    # Construct eta (Reeb vector component)
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
    
    # Construct G2 forms
    g2 = wedge(w_R7, eta) + ReOmega_R7  # φ
    star_g2 = (1/2) * wedge(w_R7, w_R7) + wedge(eta, ImOmega_R7)  # ψ
    
    return g2, star_g2


def check_phi_wedge_psi(data, n_points=100):
    """Check φ ∧ ψ = 7·Vol(K_f, g_φ)."""
    phis = data["phis"]
    psis = data["psis"]
    gG2s = data["g2_metrics"]
    
    # Random sampling if n_points < dataset size
    total_points = len(phis)
    if n_points < total_points:
        indices = np.random.choice(total_points, size=n_points, replace=False)
        print(f"\nChecking φ ∧ ψ = 7·Vol(g_φ) on {n_points} randomly sampled points...")
    else:
        indices = np.arange(total_points)
        n_points = total_points
        print(f"\nChecking φ ∧ ψ = 7·Vol(g_φ) on all {n_points} points...")
    
    vals = []
    for idx in tqdm(indices, desc="φ∧ψ check"):
        phi = vec_to_form(phis[idx], n=7, k=3)
        psi = vec_to_form(psis[idx], n=7, k=4)
        gG2 = vec_to_metric(gG2s[idx])
        
        prod = wedge(phi, psi)[0, 1, 2, 3, 4, 5, 6]
        vol = np.sqrt(np.linalg.det(gG2))
        
        vals.append(prod / vol)
    
    return np.array(vals)


def check_d_psi_and_d_phi(data, fmodel, BASIS, n_points=100, epsilon=1e-5, 
                          global_rotation_epsilon=1e-5):
    """Check dψ = 0 and dφ = ω²."""
    base_points = data['base_points']
    
    # Random sampling if n_points < dataset size
    total_points = len(base_points)
    if n_points < total_points:
        indices = np.random.choice(total_points, size=n_points, replace=False)
        print(f"\nChecking dψ = 0 and dφ = ω² on {n_points} randomly sampled points...")
    else:
        indices = np.arange(total_points)
        n_points = total_points
        print(f"\nChecking dψ = 0 and dφ = ω² on all {n_points} points...")
    
    vals_dpsi = []
    vals_dphi = []
    vals_ratio = []
    
    def sampler_phi(p):
        phi, _ = sampler_g2_package(p, fmodel, BASIS)
        return phi
    
    def sampler_psi(p):
        _, psi = sampler_g2_package(p, fmodel, BASIS)
        return psi
    
    # Use find_max_dQ_coords with BASIS
    find_max_fn = lambda point_cc: find_max_dQ_coords(point_cc, BASIS)
    
    for idx in tqdm(indices, desc="dψ and dφ check"):
        point = base_points[idx]
        
        # Check dψ
        dic_psi = sample_numerical_g2_neighborhood_val(
            sampler_psi, point, epsilon, 
            find_max_dQ_coords_fn=find_max_fn,
            global_rotation_epsilon=global_rotation_epsilon
        )
        d_psi = numerical_d_g2(dic_psi, epsilon)
        vals_dpsi.append(np.linalg.norm(d_psi))
        
        # Check dφ
        dic_phi = sample_numerical_g2_neighborhood_val(
            sampler_phi, point, epsilon,
            find_max_dQ_coords_fn=find_max_fn,
            global_rotation_epsilon=global_rotation_epsilon
        )
        d_phi = numerical_d_g2(dic_phi, epsilon)
        vals_dphi.append(np.linalg.norm(d_phi))
        
        # Compute ω²
        w = kahler_form_real_matrix(np.array(fmodel(np.expand_dims(point, axis=0))[0]))
        w_R7 = np.pad(w, ((0, 1), (0, 1)), mode='constant')
        w2 = wedge(w_R7, w_R7)
        
        # Ratio ||dφ|| / ||ω²||
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
    # plt.ylim(6.5, 7.5)  # Temporarily commented out for debugging
    plt.legend()
    plt.tight_layout()
    
    output_path = output_dir / f"g2_phi_wedge_psi_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_dpsi(vals_dpsi, run_number, output_dir):
    """Plot ||dψ|| distribution."""
    # Filter outliers (1st to 99th percentile)
    q_low, q_high = np.percentile(vals_dpsi, [1, 99])
    vals_filtered = vals_dpsi[(vals_dpsi >= q_low) & (vals_dpsi <= q_high)]
    
    # Scatter plot
    plt.figure(figsize=(7, 5))
    plt.plot(vals_dpsi, marker='.', linestyle='None', alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\|\mathrm{d}\psi\|$")
    plt.tight_layout()
    output_path = output_dir / f"g2_dpsi_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    
    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel(r"$\|\mathrm{d}\psi\|$")
    plt.ylabel("Count")
    plt.tight_layout()
    output_path = output_dir / f"g2_dpsi_run{run_number}_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_dphi(vals_dphi, run_number, output_dir):
    """Plot ||dφ|| distribution."""
    # Filter outliers
    q_low, q_high = np.percentile(vals_dphi, [1, 99])
    vals_filtered = vals_dphi[(vals_dphi >= q_low) & (vals_dphi <= q_high)]
    
    # Scatter plot
    plt.figure(figsize=(7, 5))
    plt.plot(vals_dphi, marker='.', linestyle='None', alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\|\mathrm{d}\varphi\|$")
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    
    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel(r"$\|\mathrm{d}\varphi\|$")
    plt.ylabel("Count")
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_run{run_number}_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_ratio(vals_ratio, run_number, output_dir):
    """Plot ||dφ|| / ||ω²|| ratio distribution."""
    # Filter outliers
    q_low, q_high = np.percentile(vals_ratio, [1, 99])
    vals_filtered = vals_ratio[(vals_ratio >= q_low) & (vals_ratio <= q_high)]
    
    # Scatter plot
    plt.figure(figsize=(7, 5))
    plt.plot(vals_ratio, marker='.', linestyle='None', alpha=0.6)
    plt.axhline(y=1.0, linestyle='--', color='red', alpha=0.7, label='Ideal ratio = 1')
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\|\mathrm{d}\varphi\| / \|\omega^2\|$")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_omega_ratio_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    
    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=1.0, linestyle='--', color='red', alpha=0.7, label='Ideal ratio = 1')
    plt.xlabel(r"$\|\mathrm{d}\varphi\| / \|\omega^2\|$")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_omega_ratio_run{run_number}_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Check G2 identities on learned G2 structures'
    )
    parser.add_argument('--cy-run-number', type=int, default=None,
                       help='CY model run number to use (default: auto-detect from dataset)')
    parser.add_argument('--g2-data', type=str, default='./samples/link_data/g2_test.npz',
                       help='Path to G2 dataset')
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
    print("G2 Identities Check (Analytical Construction)")
    print("=" * 80)
    
    # Load G2 data
    g2_data, dataset_cy_run = load_g2_data(g2_data_path)
    
    # Determine CY run number
    if args.cy_run_number is None:
        if dataset_cy_run is not None:
            cy_run_number = dataset_cy_run
            print(f"Using CY run {cy_run_number} from dataset metadata")
        else:
            cy_run_number = get_most_recent_cy_run_number(script_dir / 'models' / 'cy_models')
            if cy_run_number is None:
                print("Error: No CY models found and no cy_run_number in dataset")
                sys.exit(1)
            print(f"Auto-detected most recent CY model: run {cy_run_number}")
    else:
        cy_run_number = args.cy_run_number
        print(f"Using specified CY model run: {cy_run_number}")
        
        # Warn if mismatch
        if dataset_cy_run is not None and cy_run_number != dataset_cy_run:
            print("=" * 80)
            print("WARNING: CY run number mismatch!")
            print(f"  Dataset was generated with CY run {dataset_cy_run}")
            print(f"  But you specified CY run {cy_run_number}")
            print("=" * 80)
    
    print()
    
    # Check 1: φ ∧ ψ = 7·Vol
    n_points = len(g2_data['phis']) if args.n_points is None else args.n_points
    vals_phi_psi = check_phi_wedge_psi(g2_data, n_points)
    print_statistics("φ∧ψ/Vol", vals_phi_psi)
    plot_phi_wedge_psi(vals_phi_psi, "analytic", output_dir)
    
    # Check 2 & 3: dψ = 0 and dφ = ω²
    fmodel, BASIS, cy_data = load_cy_model(cy_run_number, args.cy_data_dir, script_dir)
    
    vals_dpsi, vals_dphi, vals_ratio = check_d_psi_and_d_phi(
        g2_data, fmodel, BASIS, n_points, 
        args.epsilon, args.rotation_epsilon
    )
    
    print_statistics("||dψ||", vals_dpsi)
    print_statistics("||dφ||", vals_dphi)
    print_statistics("||dφ||/||ω²||", vals_ratio)
    
    plot_dpsi(vals_dpsi, "analytic", output_dir)
    plot_dphi(vals_dphi, "analytic", output_dir)
    plot_ratio(vals_ratio, "analytic", output_dir)
    
    print()
    print("=" * 80)
    print("G2 Identities Check Complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
