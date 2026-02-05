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
    print("Warning: cymetric package not available. Some functionality may be limited.")


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


def load_g2_models(g2_run_number, script_dir):
    """
    Load trained 3form and metric G2 models.
    
    Parameters
    ----------
    g2_run_number : int
        Run number of the G2 models to load
    script_dir : Path
        Root directory of the project
        
    Returns
    -------
    dict or None
        Dictionary with '3form' and 'metric' keys containing loaded models,
        or None if models not found
    """
    models_dir = script_dir / 'models' / 'link_models'
    
    form_model_path = models_dir / f"3form_run{g2_run_number}.keras"
    metric_model_path = models_dir / f"metric_run{g2_run_number}.keras"
    
    models = {}
    
    if not form_model_path.exists():
        print(f"Warning: 3form model not found: {form_model_path}")
        return None
    
    if not metric_model_path.exists():
        print(f"Warning: Metric model not found: {metric_model_path}")
        return None
    
    models['3form'] = tf.keras.models.load_model(str(form_model_path))
    models['metric'] = tf.keras.models.load_model(str(metric_model_path))
    
    print(f"Loaded 3form model from run {g2_run_number}")
    print(f"Loaded metric model from run {g2_run_number}")
    
    return models


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
        config = yaml.safe_load(f)
    
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


def print_statistics(name, vals):
    """
    Print statistics for a set of values.
    
    Parameters
    ----------
    name : str
        Name/description of the values being analyzed
    vals : array_like
        Array of numerical values
    """
    vals = np.asarray(vals, dtype=float)
    print(f"\n{name} Statistics:")
    print(f"  Mean:   {np.mean(vals):.6e}")
    print(f"  Median: {np.median(vals):.6e}")
    print(f"  Std:    {np.std(vals):.6e}")
    print(f"  Min:    {np.min(vals):.6e}")
    print(f"  Max:    {np.max(vals):.6e}")
