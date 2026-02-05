#!/usr/bin/env python3
"""
Training script for G2 structure learning.
Reads hyperparameters from hyperparameters/hps.yaml and trains either
the 3-form or metric regressor.
"""

import sys
import yaml
import argparse
from pathlib import Path
import numpy as np
import glob
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import (
    prepare_data,
    train_regressor,
    evaluate,
    plot_history,
    plot_true_vs_pred
)


def load_hyperparameters(hps_path):
    """Load hyperparameters from YAML file."""
    with open(hps_path, 'r') as f:
        hps = yaml.safe_load(f)
    return hps


def get_next_run_number(output_dir, task):
    """Find the next available run number for the given task.
    
    Parameters
    ----------
    output_dir : Path
        Directory containing model files
    task : str
        Task name ('3form' or 'metric')
        
    Returns
    -------
    int
        Next available run number
    """
    # Find all existing model files for this task
    pattern = str(output_dir / f"{task}_run*.keras")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 1
    
    # Extract run numbers from filenames
    run_numbers = []
    for filepath in existing_files:
        filename = Path(filepath).stem  # Gets filename without extension
        match = re.search(rf"{task}_run(\d+)", filename)
        if match:
            run_numbers.append(int(match.group(1)))
    
    # Return next number
    return max(run_numbers) + 1 if run_numbers else 1


def main():
    parser = argparse.ArgumentParser(description='Train G2 structure regressor')
    parser.add_argument('--hps', type=str, 
                       default='./hyperparameters/hps.yaml',
                       help='Path to hyperparameters YAML file')
    parser.add_argument('--train-data', type=str,
                       default='./samples/link_data/g2_train.npz',
                       help='Path to training data NPZ file')
    parser.add_argument('--val-data', type=str,
                       default='./samples/link_data/g2_val.npz',
                       help='Path to validation data NPZ file')
    parser.add_argument('--output-dir', type=str,
                       default='./models/link_models',
                       help='Directory to save trained models')
    parser.add_argument('--plots-dir', type=str,
                       default='./plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    hps_path = (script_dir / args.hps).resolve()
    train_data_path = (script_dir / args.train_data).resolve()
    val_data_path = (script_dir / args.val_data).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    plots_dir = (script_dir / args.plots_dir).resolve()
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("G2 Structure Learning - Training Script")
    print("=" * 80)
    print(f"Hyperparameters file: {hps_path}")
    print(f"Training data file: {train_data_path}")
    print(f"Validation data file: {val_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Plots directory: {plots_dir}")
    print()
    
    # Load hyperparameters
    print("Loading hyperparameters...")
    hps = load_hyperparameters(hps_path)
    
    # Determine task
    task = 'metric' if hps.get('metric', False) else '3form'
    print(f"Task: Learning G2 {task}")
    print()
    
    # Load and subsample training data
    print(f"Loading training data from {train_data_path}...")
    train_data = np.load(train_data_path)
    
    # Extract cy_run_number from training data if available
    dataset_cy_run = None
    if 'cy_run_number' in train_data.files:
        dataset_cy_run = int(train_data['cy_run_number'][0])
        print(f"Training data was generated using CY model run {dataset_cy_run}")
    
    # Check cymetric_run_number from hps.yaml
    hps_cy_run = hps.get('cymetric_run_number', None)
    if hps_cy_run is not None:
        print(f"Hyperparameters specify CY model run {hps_cy_run}")
        
        # Warn if mismatch
        if dataset_cy_run is not None and hps_cy_run != dataset_cy_run:
            print("=" * 80)
            print("WARNING: CY model run number mismatch!")
            print(f"  Dataset was generated with CY run {dataset_cy_run}")
            print(f"  But hps.yaml specifies CY run {hps_cy_run}")
            print("  This may lead to inconsistent results.")
            print("=" * 80)
    
    num_samples = hps.get('num_samples', None)
    
    if num_samples is not None and num_samples < len(train_data['phis']):
        print(f"Subsampling {num_samples} training samples (total available: {len(train_data['phis'])})...")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(train_data['phis']), size=num_samples, replace=False)
        # Memory efficient: only load selected indices
        train_data_subset = {key: train_data[key][indices] for key in train_data.files}
    else:
        train_data_subset = {key: train_data[key][:] for key in train_data.files}
        num_samples = len(train_data_subset['phis'])
    
    print(f"Using {num_samples} training samples")
    
    # Prepare training X and Y
    X_train, Y_train = prepare_data(train_data_subset, task=task)
    print(f"Training input shape: {X_train.shape}")
    print(f"Training output shape: {Y_train.shape}")
    
    # Load and subsample validation data if validation is enabled
    validate = hps.get('validate', True)
    X_val, Y_val = None, None
    
    if validate:
        print(f"Loading validation data from {val_data_path}...")
        val_data = np.load(val_data_path)
        num_val_samples = hps.get('num_val_samples', None)
        
        if num_val_samples is not None and num_val_samples < len(val_data['phis']):
            print(f"Subsampling {num_val_samples} validation samples (total available: {len(val_data['phis'])})...")
            rng = np.random.default_rng(43)  # Different seed for validation
            val_indices = rng.choice(len(val_data['phis']), size=num_val_samples, replace=False)
            val_data_subset = {key: val_data[key][val_indices] for key in val_data.files}
        else:
            val_data_subset = {key: val_data[key][:] for key in val_data.files}
            num_val_samples = len(val_data_subset['phis'])
        
        print(f"Using {num_val_samples} validation samples")
        
        # Prepare validation X and Y
        X_val, Y_val = prepare_data(val_data_subset, task=task)
        print(f"Validation input shape: {X_val.shape}")
        print(f"Validation output shape: {Y_val.shape}")
    else:
        print("Validation disabled")
    
    print()
    
    # Extract training hyperparameters
    seed = 42  # Fixed seed for reproducibility
    batch_size = hps.get('batch_size', 2048)
    val_batch_size = hps.get('val_batch_size', 500)
    epochs = hps.get('epochs', 200)
    lr = hps.get('init_learning_rate', 1e-3)
    dropout = hps.get('dropout_rate', 0.0)
    activation = hps.get('activations', 'gelu')
    use_bias = hps.get('use_bias', True)
    init_scale = hps.get('parameter_initialisation_scale', 1.0)
    l2_reg = hps.get('l2_regularization', 0.0)
    huber_delta = hps.get('huber_delta', None)
    
    # Learning rate scheduler settings
    lr_reduce_factor = hps.get('lr_reduce_factor', 0.5)
    lr_reduce_patience = hps.get('lr_reduce_patience', 8)
    min_lr = hps.get('min_learning_rate', 1e-6)
    early_stop_patience = hps.get('early_stop_patience', 20)
    
    # Verbosity
    verbosity = hps.get('verbosity', 1)
    
    # Normalization settings
    normalize_inputs = hps.get('normalize_inputs', True)
    normalize_outputs = hps.get('normalize_outputs', True)
    
    # Architecture hyperparameters
    n_hidden = hps.get('n_hidden', 512)
    n_layers = hps.get('n_layers', 4)
    hidden = tuple([n_hidden] * n_layers)
    
    print("Training Configuration:")
    print(f"  Architecture: {hidden}")
    print(f"  Activation: {activation}")
    print(f"  Use bias: {use_bias}")
    print(f"  Init scale: {init_scale}")
    print(f"  Dropout: {dropout}")
    print(f"  L2 regularization: {l2_reg}")
    print(f"  Normalize inputs: {normalize_inputs}")
    print(f"  Normalize outputs: {normalize_outputs}")
    print(f"  Huber delta: {huber_delta}")
    print(f"  Batch size: {batch_size}")
    print(f"  Val batch size: {val_batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  LR reduce factor: {lr_reduce_factor}")
    print(f"  LR reduce patience: {lr_reduce_patience}")
    print(f"  Min learning rate: {min_lr}")
    print(f"  Early stop patience: {early_stop_patience}")
    print(f"  Validate: {validate}")
    print(f"  Verbosity: {verbosity}")
    print(f"  Seed: {seed}")
    print()
    
    # Check for saved model to load
    saved_model_path = hps.get('saved_model_path', None)
    if saved_model_path is not None:
        saved_model_path = (script_dir / saved_model_path).resolve()
        if saved_model_path.exists():
            print(f"Loading saved model from: {saved_model_path}")
            print("Note: Model will be fine-tuned with current hyperparameters")
            print()
        else:
            print(f"Warning: saved_model_path specified but not found: {saved_model_path}")
            print("Training new model from scratch")
            saved_model_path = None
    
    # Get next run number for this task
    run_number = get_next_run_number(output_dir, task)
    run_name = f"{task}_run{run_number}"
    
    # Set up paths
    model_path = output_dir / f"{run_name}.keras"
    
    print(f"Starting training for {task}...")
    print(f"Model will be saved to: {model_path}")
    print("=" * 80)
    print()
    
    # Load saved model if specified
    pretrained_model = None
    if saved_model_path is not None:
        import tensorflow as tf
        print("Loading existing model...")
        pretrained_model = tf.keras.models.load_model(saved_model_path)
        print(f"Model loaded: {pretrained_model.name}")
        print("Will continue training with new hyperparameters.")
        print()
    
    # Train the model
    model, hist, (X_test, Y_test, Y_pred), norms, test_metrics = train_regressor(
        X_train, Y_train,
        X_val=X_val, 
        Y_val=Y_val,
        pretrained_model=pretrained_model,
        task=task,
        seed=seed,
        batch=batch_size,
        val_batch=val_batch_size,
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        dropout=dropout,
        activation=activation,
        use_bias=use_bias,
        init_scale=init_scale,
        l2_reg=l2_reg,
        huber_delta=huber_delta,
        normalize_inputs=normalize_inputs,
        normalize_outputs=normalize_outputs,
        lr_reduce_factor=lr_reduce_factor,
        lr_reduce_patience=lr_reduce_patience,
        min_lr=min_lr,
        early_stop_patience=early_stop_patience,
        validate=validate,
        verbosity=verbosity,
        checkpoint_path=model_path,
    )
    
    print()
    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print()
    
    # Print test metrics
    print("Test Metrics (normalized space):")
    for key, val in test_metrics.items():
        print(f"  {key}: {val:.6f}")
    print()
    
    # Evaluate on test set (original scale)
    print("Test Metrics (original scale):")
    evaluate(Y_test, Y_pred)
    print()
    
    # Generate and save plots
    print("Generating plots...")
    
    # Training history plot
    history_plot_path = plots_dir / f"{run_name}_history.png"
    plot_history(hist, save_path=history_plot_path)
    
    # Prediction plot
    pred_plot_path = plots_dir / f"{run_name}_predictions.png"
    plot_true_vs_pred(Y_test, Y_pred, save_path=pred_plot_path)
    
    print()
    print("=" * 80)
    print("All done!")
    print(f"Model saved to: {model_path}")
    print(f"Plots saved to: {plots_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
