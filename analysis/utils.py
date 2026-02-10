"""
Common utility functions for analysis scripts.

This module provides shared functionality for model loading, run number management,
and statistics reporting across different analysis scripts.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml
import glob
import re
import matplotlib.pyplot as plt

# Setup paths for project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def get_most_recent_g2_run_number(model_dir='./models/link_models'):
    """
    Find the most recent run number for G2 models.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing G2 model files
        
    Returns
    -------
    int or None
        Most recent run number, or None if no runs exist
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return None
    
    pattern = str(model_dir / "*_run*.keras")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return None
    
    run_numbers = []
    for filepath in existing_files:
        filename = Path(filepath).stem
        match = re.search(r'_run(\d+)', filename)
        if match:
            run_numbers.append(int(match.group(1)))
    
    return max(run_numbers) if run_numbers else None


def get_most_recent_cy_run_number(model_dir='./models/cy_models'):
    """
    Find the most recent run number for CY models.
    
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
    
    pattern = str(model_dir / "cy_metric_model_run*.keras")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return None
    
    run_numbers = []
    for filepath in existing_files:
        filename = Path(filepath).stem
        match = re.search(r'_run(\d+)', filename)
        if match:
            run_numbers.append(int(match.group(1)))
    
    return max(run_numbers) if run_numbers else None


def load_g2_models(g2_run_number, script_dir, load_4form=False):
    """
    Load G2 models (3form, metric, and optionally 4form).
    
    Parameters
    ----------
    g2_run_number : int
        Run number of the G2 models to load
    script_dir : Path
        Root directory of the project
    load_4form : bool, optional
        Whether to load the 4form model (default: False)
        
    Returns
    -------
    dict or None
        Dictionary with '3form', 'metric', and optionally '4form' keys containing loaded models,
        or None if required models not found
    """
    models_dir = script_dir / 'models' / 'link_models'

    form_model_path = models_dir / f"3form_run{g2_run_number}.keras"
    metric_model_path = models_dir / f"metric_run{g2_run_number}.keras"
    fourform_model_path = models_dir / f"4form_run{g2_run_number}.keras"

    models = {}

    if not form_model_path.exists():
        print(f"Error: 3form model not found: {form_model_path}")
        return None

    if not metric_model_path.exists():
        print(f"Error: Metric model not found: {metric_model_path}")
        return None
    
    models['3form'] = tf.keras.models.load_model(str(form_model_path))
    models['metric'] = tf.keras.models.load_model(str(metric_model_path))
    
    return_dict = {
        '3form': models['3form'],
        'metric': models['metric']
    }
    
    # Load 4form model only if requested
    if load_4form and fourform_model_path.exists():
        models['4form'] = tf.keras.models.load_model(str(fourform_model_path))
        return_dict['4form'] = models['4form']

    # Load normalization statistics from .npz files
    form_norm_path = models_dir / f"3form_run{g2_run_number}_norm_stats.npz"
    metric_norm_path = models_dir / f"metric_run{g2_run_number}_norm_stats.npz"
    fourform_norm_path = models_dir / f"4form_run{g2_run_number}_norm_stats.npz"

    if form_norm_path.exists():
        form_norm_data = np.load(form_norm_path)
        y_mean_form = form_norm_data['y_mean']
        y_std_form = np.sqrt(form_norm_data['y_variance'])
        # Ensure 1D arrays for consistent broadcasting
        if y_mean_form.ndim > 1:
            y_mean_form = y_mean_form.flatten()
        if y_std_form.ndim > 1:
            y_std_form = y_std_form.flatten()
        return_dict['3form_y_mean'] = y_mean_form
        return_dict['3form_y_std'] = y_std_form
    else:
        print(f"Warning: 3form normalization stats not found: {form_norm_path}")
    
    if metric_norm_path.exists():
        metric_norm_data = np.load(metric_norm_path)
        y_mean_metric = metric_norm_data['y_mean']
        y_std_metric = np.sqrt(metric_norm_data['y_variance'])
        # Ensure 1D arrays for consistent broadcasting
        if y_mean_metric.ndim > 1:
            y_mean_metric = y_mean_metric.flatten()
        if y_std_metric.ndim > 1:
            y_std_metric = y_std_metric.flatten()
        return_dict['metric_y_mean'] = y_mean_metric
        return_dict['metric_y_std'] = y_std_metric
    else:
        print(f"Warning: Metric normalization stats not found: {metric_norm_path}")
    
    print(f"Loaded 3form model from run {g2_run_number}")
    print(f"Loaded metric model from run {g2_run_number}")
    
    # Load 4form normalization stats if 4form model was loaded
    if load_4form and '4form' in return_dict:
        if fourform_norm_path.exists():
            fourform_norm_data = np.load(fourform_norm_path)
            y_mean_fourform = fourform_norm_data['y_mean']
            y_std_fourform = np.sqrt(fourform_norm_data['y_variance'])
            # Ensure 1D arrays for consistent broadcasting
            if y_mean_fourform.ndim > 1:
                y_mean_fourform = y_mean_fourform.flatten()
            if y_std_fourform.ndim > 1:
                y_std_fourform = y_std_fourform.flatten()
            return_dict['4form_y_mean'] = y_mean_fourform
            return_dict['4form_y_std'] = y_std_fourform
        else:
            print(f"Warning: 4form normalization stats not found: {fourform_norm_path}")
        
        # Load index mapping for 4form (to expand from 23 to 35 dims)
        index_map_path = models_dir / f"4form_run{g2_run_number}_index_map.npz"
        if index_map_path.exists():
            index_map_data = np.load(index_map_path)
            return_dict['4form_zero_indices'] = index_map_data['psi_zero_indices']
            return_dict['4form_nonzero_indices'] = index_map_data['psi_nonzero_indices']
        else:
            # Use default indices
            return_dict['4form_zero_indices'] = np.array([6, 8, 12, 15, 17, 18, 24, 25, 27, 29, 32, 33])
            return_dict['4form_nonzero_indices'] = np.array([i for i in range(35) if i not in return_dict['4form_zero_indices']])
        
        print(f"Loaded 4form model from run {g2_run_number}")
    
    return return_dict

def load_cy_model(cy_run_number, data_dir, script_dir):
    """
    Load CY metric model and data.
    
    Parameters
    ----------
    cy_run_number : int
        Run number of the CY model to load
    data_dir : str or Path
        Directory containing CY dataset and basis
    script_dir : Path
        Root directory of the project
        
    Returns
    -------
    tuple
        (fmodel, BASIS, cy_data) - CY model wrapper, basis, and dataset
    """
    if not CYMETRIC_AVAILABLE:
        raise ImportError("cymetric package is required but not available")
    
    model_path = script_dir / f'models/cy_models/cy_metric_model_run{cy_run_number}.keras'
    config_path = script_dir / f'models/cy_models/cy_metric_model_run{cy_run_number}_config.yaml'
    data_dir = script_dir / data_dir
    
    cy_data = np.load(data_dir / 'dataset.npz')
    BASIS = np.load(data_dir / 'basis.pickle', allow_pickle=True)
    BASIS = prepare_basis(BASIS)
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    loaded_nn = tf.keras.models.load_model(str(model_path))
    fmodel = MultFSModel(loaded_nn, BASIS, alpha=config['alpha'])
    
    print(f"Loaded CY model from run {cy_run_number}")
    
    return fmodel, BASIS, cy_data


def load_g2_data(data_path):
    """
    Load G2 dataset.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the G2 dataset .npz file
        
    Returns
    -------
    tuple
        (data, cy_run_number) - Dataset dict and CY run number if available
    """
    data = np.load(data_path)
    print(f"Loaded G2 data from {data_path}")
    print(f"Dataset contains {len(data['phis'])} samples")
    
    if 'cy_run_number' in data.files:
        cy_run = int(data['cy_run_number'][0])
        print(f"Dataset was generated using CY model run {cy_run}")
        return data, cy_run
    else:
        print("Warning: Dataset does not contain cy_run_number metadata")
        return data, None


def print_statistics(name, vals, outlier_proportion=0.0):
    """
    Print statistics for a set of values.
    
    Parameters
    ----------
    name : str
        Name/description of the values being analyzed
    vals : array_like
        Array of numerical values
    outlier_proportion : float, optional
        Proportion of top outliers to remove (default: 0.0 = no filtering)
    """
    vals = np.asarray(vals, dtype=float)
    
    # Filter outliers if requested
    if outlier_proportion > 0 and len(vals) > 10:
        percentile_high = (1 - outlier_proportion) * 100
        q_high = np.percentile(vals, percentile_high)
        vals_filtered = vals[vals <= q_high]
        print(f"\n{name} Statistics (excluding top {outlier_proportion*100:.1f}%):")
    else:
        vals_filtered = vals
        print(f"\n{name} Statistics:")
    
    if len(vals_filtered) == 0:
        print(f"  No valid values to analyze")
        return
    print(f"  Mean:   {np.mean(vals_filtered):.6e}")
    print(f"  Median: {np.median(vals_filtered):.6e}")
    print(f"  Std:    {np.std(vals_filtered):.6e}")
    print(f"  Min:    {np.min(vals_filtered):.6e}")
    print(f"  Max:    {np.max(vals_filtered):.6e}")

def plot_phi_wedge_psi(vals, run_number, output_dir):
    """Plot φ∧ψ/Vol check results."""
    if len(vals) == 0:
        print(f"Skipping φ∧ψ plot: no valid data")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(vals)), vals, marker='.', alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\frac{\varphi\wedge\psi}{\sqrt{\det(g_{\varphi})}}$")
    plt.axhline(y=7, linestyle=':', linewidth=2, color='red',
                label=r"$\frac{\varphi\wedge\psi}{\sqrt{\det(g_{\varphi})}}=7$")
    # plt.ylim(6.5, 7.5)  # Temporarily commented out for debugging
    plt.legend()
    plt.tight_layout()
    
    output_path = output_dir / f"g2_phi_wedge_psi_model_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def plot_dphi_ratio(vals_ratio, run_number, output_dir, outlier_proportion=0.05):
    """Plot ||dφ||/||ω²|| distribution.
    
    Parameters
    ----------
    vals_ratio : array
        Ratio values
    run_number : int
        Model run number
    output_dir : Path
        Directory to save plots
    outlier_proportion : float, optional
        Proportion of top outliers to remove (default: 0.05 = 5%)
    """
    if len(vals_ratio) == 0:
        return
    
    # Filter top outliers for plotting
    if outlier_proportion > 0 and len(vals_ratio) > 10:
        percentile_high = (1 - outlier_proportion) * 100
        q_high = np.percentile(vals_ratio, percentile_high)
        vals_filtered = vals_ratio[vals_ratio <= q_high]
    else:
        vals_filtered = vals_ratio
    
    # Scatter plot (using filtered values)
    plt.figure(figsize=(7, 5))
    plt.plot(vals_filtered, marker='.', linestyle='None', alpha=0.6)
    plt.axhline(y=1.0, linestyle='--', color='red', alpha=0.7, label='Ideal ratio = 1')
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\|\mathrm{d}\varphi\| / \|\omega^2\|$")
    plt.ylim(bottom=0, top=10)
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_omega_ratio_model_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    
    '''
    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=1.0, linestyle='--', color='red', alpha=0.7, label='Ideal ratio = 1')
    plt.xlabel(r"$\|\mathrm{d}\varphi\| / \|\omega^2\|$")
    plt.ylabel("Count")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / f"g2_dphi_omega_ratio_model_run{run_number}_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    '''

def plot_dpsi(vals_dpsi, run_number, output_dir, outlier_proportion=0.05):
    """Plot ||dψ|| distribution.
    
    Parameters
    ----------
    vals_dpsi : array
        dψ norm values
    run_number : int
        Model run number
    output_dir : Path
        Directory to save plots
    outlier_proportion : float, optional
        Proportion of top outliers to remove (default: 0.05 = 5%)
    """
    if len(vals_dpsi) == 0:
        print(f"Skipping dψ plots: no valid data")
        return
    
    # Filter top outliers for plotting
    if outlier_proportion > 0 and len(vals_dpsi) > 10:
        percentile_high = (1 - outlier_proportion) * 100
        q_high = np.percentile(vals_dpsi, percentile_high)
        vals_filtered = vals_dpsi[vals_dpsi <= q_high]
    else:
        vals_filtered = vals_dpsi
    
    # Scatter plot (using filtered values)
    plt.figure(figsize=(7, 5))
    plt.plot(vals_filtered, marker='.', linestyle='None', alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel(r"$\|\mathrm{d}\psi\|$")
    plt.ylim(bottom=0, top=1)
    plt.tight_layout()
    output_path = output_dir / f"g2_dpsi_model_run{run_number}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()
    
    '''
    # Histogram
    if len(vals_filtered) > 0:
        plt.figure(figsize=(7, 5))
        plt.hist(vals_filtered, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel(r"$\|\mathrm{d}\psi\|$")
        plt.ylabel("Count")
        plt.ylim(bottom=0)
        plt.tight_layout()
        output_path = output_dir / f"g2_dpsi_model_run{run_number}_histogram.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
        plt.close()
    '''