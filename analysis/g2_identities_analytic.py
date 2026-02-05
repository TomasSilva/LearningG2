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
    get_most_recent_g2_run_number,
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


def check_phi_wedge_psi(data, n_points=100, start_idx=0):
    """Check φ ∧ ψ = 7·Vol(K_f, g_φ)."""
    print(f"\nChecking φ ∧ ψ = 7·Vol(g_φ) on {n_points} points...")
    
    phis = data["phis"]
    psis = data["psis"]
    gG2s = data["g2_metrics"]
    
    vals = []
    for i in tqdm(range(start_idx, start_idx + n_points), desc="φ∧ψ check"):
        phi = vec_to_form(phis[i], n=7, k=3)
        psi = vec_to_form(psis[i], n=7, k=4)
        gG2 = vec_to_metric(gG2s[i])
        
        prod = wedge(phi, psi)[0, 1, 2, 3, 4, 5, 6]
        vol = np.sqrt(np.linalg.det(gG2))
        
        vals.append(prod / vol)
    
    return np.array(vals)


def check_d_psi_and_d_phi(data, fmodel, BASIS, n_points=100, epsilon=1e-12, 
                          global_rotation_epsilon=1e-12):
    """Check dψ = 0 and dφ = ω²."""
    print(f"\nChecking dψ = 0 and dφ = ω² on {n_points} points...")
    
    base_points = data['base_points']
    
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
    
    for i in tqdm(range(n_points), desc="dψ and dφ check"):
        point = base_points[i]
        
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
    plt.ylim(6.5, 7.5)
    plt.title(f"G2 Identity Check: φ∧ψ = 7·Vol (Run {run_number})")
    plt.legend()
    plt.grid(True, alpha=0.3)
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1.plot(vals_dpsi, marker='.', linestyle='None', alpha=0.6)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel(r"$\|\mathrm{d}\psi\|$")
    ax1.set_title(f"||dψ|| per Sample (Run {run_number})")
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel(r"$\|\mathrm{d}\psi\|$")
    ax2.set_ylabel("Count")
    ax2.set_title(f"||dψ|| Distribution (1-99 percentile, Run {run_number})")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"g2_dpsi_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_dphi(vals_dphi, run_number, output_dir):
    """Plot ||dφ|| distribution."""
    # Filter outliers
    q_low, q_high = np.percentile(vals_dphi, [1, 99])
    vals_filtered = vals_dphi[(vals_dphi >= q_low) & (vals_dphi <= q_high)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1.plot(vals_dphi, marker='.', linestyle='None', alpha=0.6)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel(r"$\|\mathrm{d}\varphi\|$")
    ax1.set_title(f"||dφ|| per Sample (Run {run_number})")
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel(r"$\|\mathrm{d}\varphi\|$")
    ax2.set_ylabel("Count")
    ax2.set_title(f"||dφ|| Distribution (1-99 percentile, Run {run_number})")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_ratio(vals_ratio, run_number, output_dir):
    """Plot ||dφ|| / ||ω²|| ratio distribution."""
    # Filter outliers
    q_low, q_high = np.percentile(vals_ratio, [1, 99])
    vals_filtered = vals_ratio[(vals_ratio >= q_low) & (vals_ratio <= q_high)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1.plot(vals_ratio, marker='.', linestyle='None', alpha=0.6)
    ax1.axhline(y=1.0, linestyle='--', color='red', alpha=0.7, label='Ideal ratio = 1')
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel(r"$\|\mathrm{d}\varphi\| / \|\omega^2\|$")
    ax1.set_title(f"dφ = ω² Check: Ratio (Run {run_number})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    ax2.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(x=1.0, linestyle='--', color='red', alpha=0.7, label='Ideal ratio = 1')
    ax2.set_xlabel(r"$\|\mathrm{d}\varphi\| / \|\omega^2\|$")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Ratio Distribution (1-99 percentile, Run {run_number})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_omega_ratio_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Check G2 identities on learned G2 structures'
    )
    parser.add_argument('--g2-run-number', type=int, default=None,
                       help='G2 model run number to use (default: most recent)')
    parser.add_argument('--cy-run-number', type=int, default=None,
                       help='CY model run number to use (default: auto-detect from dataset)')
    parser.add_argument('--g2-data', type=str, default='./samples/link_data/g2_test.npz',
                       help='Path to G2 dataset')
    parser.add_argument('--cy-data-dir', type=str, default='./samples/cy_data',
                       help='Directory containing CY data')
    parser.add_argument('--n-phi-psi', type=int, default=None,
                       help='Number of points for φ∧ψ check (default: all)')
    parser.add_argument('--n-derivatives', type=int, default=None,
                       help='Number of points for dψ and dφ checks (default: all)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Starting index in dataset')
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
    print("G2 Identities Check")
    print("=" * 80)
    
    # Determine G2 run number
    if args.g2_run_number is None:
        g2_run_number = get_most_recent_g2_run_number(script_dir / 'models' / 'link_models')
        if g2_run_number is None:
            print("Warning: No G2 models found. Will check dataset only.")
            g2_run_number = 0  # Use 0 for dataset-only checks
        else:
            print(f"Auto-detected most recent G2 model: run {g2_run_number}")
    else:
        g2_run_number = args.g2_run_number
        print(f"Using specified G2 model run: {g2_run_number}")
    
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
    n_phi_psi = len(g2_data['phis']) - args.start_idx if args.n_phi_psi is None else args.n_phi_psi
    vals_phi_psi = check_phi_wedge_psi(g2_data, n_phi_psi, args.start_idx)
    print_statistics("φ∧ψ/Vol", vals_phi_psi)
    plot_phi_wedge_psi(vals_phi_psi, g2_run_number, output_dir)
    
    # Check 2 & 3: dψ = 0 and dφ = ω²
    fmodel, BASIS, cy_data = load_cy_model(cy_run_number, args.cy_data_dir, script_dir)
    
    n_derivatives = len(g2_data['base_points']) if args.n_derivatives is None else args.n_derivatives
    vals_dpsi, vals_dphi, vals_ratio = check_d_psi_and_d_phi(
        g2_data, fmodel, BASIS, n_derivatives, 
        args.epsilon, args.rotation_epsilon
    )
    
    print_statistics("||dψ||", vals_dpsi)
    print_statistics("||dφ||", vals_dphi)
    print_statistics("||dφ||/||ω²||", vals_ratio)
    
    plot_dpsi(vals_dpsi, g2_run_number, output_dir)
    plot_dphi(vals_dphi, g2_run_number, output_dir)
    plot_ratio(vals_ratio, g2_run_number, output_dir)
    
    print()
    print("=" * 80)
    print("G2 Identities Check Complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
