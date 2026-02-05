#!/usr/bin/env python3
"""
Train the Calabi-Yau metric using cymetric package.

This script generates training data, builds a neural network model,
and trains it to learn the CY metric.
"""

import numpy as np
import os
import sys
import yaml
import argparse
import tensorflow as tf
from pathlib import Path
import glob
import re

# ============================================================================
# HYPERPARAMETERS - Edit these for easy customization
# ============================================================================

# Geometry parameters --> Defining the quintic CY 3-fold
MONOMIALS = 5 * np.eye(5, dtype=np.int64)   # Monomial matrix
COEFFICIENTS = np.ones(5)                   # Coefficients for defining equation
KMODULI = np.ones(1)                        # Kähler moduli
AMBIENT = np.array([4])                     # Ambient space dimension

# Data generation
N_POINTS = 200000                           # Number of training points to generate

# Model architecture
N_LAYERS = 3                                # Number of hidden layers
N_HIDDEN = 64                               # Hidden units per layer
ACTIVATION = 'gelu'                         # Activation function
N_FOLD = 3                                  # Number of folds (dimension of CY)

# Training parameters
N_EPOCHS = 500                              # Number of training epochs
BATCH_SIZES = [64, 50000]                   # Batch sizes for training phases
ALPHA = [1., 1., 1., 1., 1.]                # Loss weights [ricci, sigma, kaehler, transition, volk]

# Paths
DATA_DIR = './samples/cy_data'              # Directory for training data
SAVE_DIR = './models/cy_models'             # Directory to save trained model
MODEL_NAME = 'cy_metric_model'              # Name for saved model

# ============================================================================

# Setup path for cymetric package
try:
    # When running as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # When running in a Jupyter notebook
    SCRIPT_DIR = Path().resolve()

# Add parent directory and cymetric to path
_parent_dir = SCRIPT_DIR.parent
_cymetric_dir = _parent_dir / "cymetric"

if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_cymetric_dir) not in sys.path:
    sys.path.insert(0, str(_cymetric_dir))

# Import cymetric functions
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.callbacks import (
    RicciCallback, 
    SigmaCallback, 
    VolkCallback, 
    KaehlerCallback, 
    TransitionCallback
)
from cymetric.models.models import MultFSModel
from cymetric.models.helper import prepare_basis, train_model

# Import utilities
from analysis.utils import get_most_recent_cy_run_number


def get_next_cy_run_number(save_dir='./models/cy_models'):
    """Find the next available run number for CY models.
    
    Parameters
    ----------
    save_dir : str or Path
        Directory containing CY model files
        
    Returns
    -------
    int
        Next available run number
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return 1
    
    # Find all existing model files
    pattern = str(save_dir / "cy_metric_model_run*.keras")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 1
    
    # Extract run numbers from filenames
    run_numbers = []
    for filepath in existing_files:
        filename = Path(filepath).stem  # Gets filename without extension
        match = re.search(r"cy_metric_model_run(\d+)", filename)
        if match:
            run_numbers.append(int(match.group(1)))
    
    # Return next number
    return max(run_numbers) + 1 if run_numbers else 1


def generate_training_data(n_points=N_POINTS, output_dir=DATA_DIR, 
                          monomials=MONOMIALS, coefficients=COEFFICIENTS,
                          kmoduli=KMODULI, ambient=AMBIENT):
    """
    Generate training data for CY metric learning.
    
    Parameters
    ----------
    n_points : int
        Number of training points to generate
    output_dir : str or Path
        Directory to save the training data
    monomials : ndarray
        Monomial matrix defining the CY
    coefficients : ndarray
        Coefficients for defining equation
    kmoduli : ndarray
        Kähler moduli
    ambient : ndarray
        Ambient space dimension
        
    Returns
    -------
    data : dict
        Loaded training data
    BASIS : dict
        Prepared basis functions
    """
    print(f"Generating {n_points} training points...")
    
    # Create point generator
    pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
    
    # Generate and save dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    kappa = pg.prepare_dataset(n_points, str(output_dir))
    pg.prepare_basis(str(output_dir), kappa=kappa)
    
    # Load the generated data
    data = np.load(output_dir / 'dataset.npz')
    BASIS = np.load(output_dir / 'basis.pickle', allow_pickle=True)
    BASIS = prepare_basis(BASIS)
    
    print(f"Training data saved to {output_dir}")
    print(f"Data keys: {list(data.keys())}")
    
    # Store geometry config for later use
    geometry_config = {
        'monomials': monomials.tolist(),
        'coefficients': coefficients.tolist(),
        'kmoduli': kmoduli.tolist(),
        'ambient': ambient.tolist()
    }
    
    return data, BASIS, geometry_config


def build_model(n_in=10, n_out=9, n_layers=3, n_hidden=64, activation='gelu', 
                BASIS=None, alpha=None):
    """
    Build the neural network model for CY metric learning.
    
    Parameters
    ----------
    n_in : int
        Input dimension (2 * n_fold)
    n_out : int
        Output dimension (n_fold^2)
    n_layers : int
        Number of hidden layers
    n_hidden : int
        Number of units per hidden layer
    activation : str
        Activation function
    BASIS : dict
        Basis functions from prepare_basis
    alpha : list
        Loss weights [alpha_ricci, alpha_sigma, alpha_kaehler, alpha_transition, alpha_volk]
        
    Returns
    -------
    fmodel : MultFSModel
        The constructed model
    """
    if alpha is None:
        alpha = [1., 1., 1., 1., 1.]
    
    print(f"Building model: {n_layers} layers x {n_hidden} units, activation={activation}")
    
    # Build neural network
    nn = tf.keras.Sequential()
    nn.add(tf.keras.Input(shape=(n_in,)))
    for i in range(n_layers):
        nn.add(tf.keras.layers.Dense(n_hidden, activation=activation))
    nn.add(tf.keras.layers.Dense(n_out, use_bias=False))
    
    # Wrap in MultFSModel
    fmodel = MultFSModel(nn, BASIS, alpha=alpha)
    
    return fmodel


def setup_callbacks(data):
    """
    Setup training callbacks.
    
    Parameters
    ----------
    data : dict
        Training data containing validation sets
        
    Returns
    -------
    list
        List of callback objects
    """
    rcb = RicciCallback((data['X_val'], data['y_val']), data['val_pullbacks'])
    scb = SigmaCallback((data['X_val'], data['y_val']))
    volkcb = VolkCallback((data['X_val'], data['y_val']))
    kcb = KaehlerCallback((data['X_val'], data['y_val']))
    tcb = TransitionCallback((data['X_val'], data['y_val']))
    
    return [rcb, scb, kcb, tcb, volkcb]


def train_cy_metric(data, BASIS, geometry_config, 
                    n_layers=3, n_hidden=64, activation='gelu',
                    n_epochs=500, batch_sizes=None,
                    alpha=None, n_fold=3,
                    save_dir='./models/cy_models',
                    run_number=None):
    """
    Train the CY metric model.
    
    Parameters
    ----------
    data : dict
        Training data
    BASIS : dict
        Basis functions
    geometry_config : dict
        Geometry configuration
    n_layers : int
        Number of hidden layers
    n_hidden : int
        Hidden units per layer
    activation : str
        Activation function
    n_epochs : int
        Number of training epochs
    batch_sizes : list
        Batch sizes for different training phases
    alpha : list
        Loss weights
    n_fold : int
        Number of folds
    save_dir : str
        Directory to save trained model
    run_number : int or None
        Run number for model naming. If None, automatically determine next number.
        
    Returns
    -------
    fmodel : MultFSModel
        Trained model
    history : dict
        Training history
    run_number : int
        Run number used for saving
    """
    if batch_sizes is None:
        batch_sizes = [64, 50000]
    if alpha is None:
        alpha = [1., 1., 1., 1., 1.]
    
    # Determine run number
    if run_number is None:
        run_number = get_next_cy_run_number(save_dir)
    
    model_name = f'cy_metric_model_run{run_number}'
    
    print(f"Training CY metric model - Run {run_number}")
    
    # Model dimensions
    # n_in is 2 * (ambient + 1) for complex coordinates in projective space
    # For quintic: ambient=4 means P^4 with 5 homogeneous coords -> 10 real coords
    n_ambient = geometry_config['ambient'][0]
    n_in = 2 * (n_ambient + 1)
    n_out = n_fold ** 2
    
    # Build model
    fmodel = build_model(
        n_in=n_in, 
        n_out=n_out, 
        n_layers=n_layers,
        n_hidden=n_hidden, 
        activation=activation,
        BASIS=BASIS, 
        alpha=alpha
    )
    
    # Setup callbacks
    cb_list = setup_callbacks(data)
    
    # Setup optimizer
    opt = tf.keras.optimizers.Adam()
    
    # Train model
    print(f"\nStarting training for {n_epochs} epochs...")
    fmodel, training_history = train_model(
        fmodel, 
        data, 
        optimizer=opt, 
        epochs=n_epochs, 
        batch_sizes=batch_sizes,
        verbose=1, 
        callbacks=cb_list
    )
    
    # Save model
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f'{model_name}.keras'
    fmodel.save(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    # Save configuration
    config = {
        'n_in': n_in,
        'n_out': n_out,
        'n_layers': n_layers,
        'n_hidden': n_hidden,
        'activation': activation,
        'n_epochs': n_epochs,
        'batch_sizes': batch_sizes,
        'alpha': alpha,
        'n_fold': n_fold,
        **geometry_config
    }
    
    config_path = save_dir / f'{model_name}_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Configuration saved to {config_path}")
    
    return fmodel, training_history, run_number


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train CY metric model')
    parser.add_argument('--n-points', type=int, default=N_POINTS,
                       help=f'Number of training points to generate (default: {N_POINTS})')
    parser.add_argument('--n-layers', type=int, default=N_LAYERS,
                       help=f'Number of hidden layers (default: {N_LAYERS})')
    parser.add_argument('--n-hidden', type=int, default=N_HIDDEN,
                       help=f'Hidden units per layer (default: {N_HIDDEN})')
    parser.add_argument('--activation', type=str, default=ACTIVATION,
                       help=f'Activation function (default: {ACTIVATION})')
    parser.add_argument('--n-epochs', type=int, default=N_EPOCHS,
                       help=f'Number of training epochs (default: {N_EPOCHS})')
    parser.add_argument('--n-fold', type=int, default=N_FOLD,
                       help=f'Number of folds (default: {N_FOLD})')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help=f'Directory for training data (default: {DATA_DIR})')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR,
                       help=f'Directory to save trained model (default: {SAVE_DIR})')
    parser.add_argument('--run-number', type=int, default=None,
                       help='Run number for model naming (default: auto-increment)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CY Metric Training")
    print("=" * 80)
    
    # Generate or load training data
    data, BASIS, geometry_config = generate_training_data(
        n_points=args.n_points,
        output_dir=args.data_dir
    )
    
    # Train model
    fmodel, history, run_number = train_cy_metric(
        data=data,
        BASIS=BASIS,
        geometry_config=geometry_config,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        activation=args.activation,
        batch_sizes=BATCH_SIZES,
        alpha=ALPHA,
        n_epochs=args.n_epochs,
        n_fold=args.n_fold,
        save_dir=args.save_dir,
        run_number=args.run_number
    )
    
    print("\n" + "=" * 80)
    print(f"Training complete! CY model saved as run {run_number}")
    print("=" * 80)


if __name__ == '__main__':
    main()
