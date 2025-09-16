#!/usr/bin/env python3
"""
G2 Structure Learning - Massive Scale Training (100K+ samples)
HPC-ready standalone script for large-scale training experiment

FEATURES:
- GPU/CPU auto-detection and optimization
- No command line arguments required (uses sensible defaults)
- Memory-efficient data handling
- Progress monitoring with flush=True for HPC environments

Memory Requirements Estimation:
- Training data: 100K samples × 7 features × 8 bytes = ~5.6 MB
- Target data: 100K samples × 17 features × 8 bytes = ~13.6 MB  
- Normalized copies: ~19.2 MB (duplicated)
- Test data: 10K samples = ~1.9 MB
- Model parameters: 9,617 × 4 bytes = ~38 KB
- TensorFlow overhead: ~2-4 GB (CPU) or ~3-5 GB (GPU)
- Training history: ~50 MB
- Intermediate calculations: ~500 MB
TOTAL ESTIMATED: 6-8 GB RAM

Recommended HPC Resources:
- RAM: 16 GB (safe margin)
- CPU: 4-8 cores (if no GPU)
- GPU: Any modern GPU with 4GB+ VRAM (recommended)
- Time: 1-4 hours (GPU: 1-2h, CPU: 2-4h)

Usage:
- Simple: python3 train_massive_g2.py
- Custom: python3 train_massive_g2.py --train-samples 200000 --test-samples 20000
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import G2 project modules
from sampling.sampling import LinkSample
from geometry.compression import form_to_vec

# Configure TensorFlow before any operations
def configure_tensorflow():
    """Configure TensorFlow settings before initialization"""
    try:
        # Configure CPU threading before any TF operations
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except RuntimeError:
        # Already configured, ignore
        pass

# Configure TensorFlow immediately after import
configure_tensorflow()

def detect_and_setup_device():
    """Detect and configure GPU/CPU device for training"""
    # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        physical_devices = tf.config.list_physical_devices('GPU')
        print(f"🚀 GPU detected: {len(physical_devices)} device(s)", flush=True)
        
        # Enable memory growth to avoid allocating all GPU memory at once
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        
        device_name = tf.test.gpu_device_name()
        if device_name:
            print(f"� Using GPU: {device_name}", flush=True)
        else:
            print("📱 Using GPU: Default GPU device", flush=True)
        
        return "GPU"
    else:
        print("💻 No GPU detected, using CPU", flush=True)
        return "CPU"

def train_massive_g2_model(n_train=100000, n_test=10000, output_dir='massive_results'):
    """
    Train G2 structure learning model with massive dataset
    
    Args:
        n_train (int): Number of training samples (default: 100,000)
        n_test (int): Number of test samples (default: 10,000) 
        output_dir (str): Directory to save results
    """
    
    print("🚀" + "="*80, flush=True)
    print("🚀 G2 STRUCTURE LEARNING - MASSIVE SCALE TRAINING", flush=True)
    print("🚀" + "="*80, flush=True)
    print(f"Training samples: {n_train:,}", flush=True)
    print(f"Test samples: {n_test:,}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    
    # Detect and setup GPU/CPU
    device_type = detect_and_setup_device()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define zero and non-zero indices (from weighted loss function)
    zero_indices = [1, 2, 4, 5, 7, 8, 9, 10, 16, 17, 18, 19, 22, 26, 28, 32, 33, 34]
    all_indices = list(range(35))  # Total G2 form components
    non_zero_indices = [i for i in all_indices if i not in zero_indices]
    
    print(f"\n📊 G2 Structure Analysis:", flush=True)
    print(f"  Total G2 components: {len(all_indices)}", flush=True)
    print(f"  Zero components: {len(zero_indices)}", flush=True)
    print(f"  Non-zero components: {len(non_zero_indices)}", flush=True)
    
    # ===============================
    # GENERATE MASSIVE DATASET
    # ===============================
    
    print(f"\n=== Generating Massive Dataset ===", flush=True)
    print("⚠️  This may take 10-30 minutes depending on hardware...", flush=True)
    
    start_time = time.time()
    
    # Training data
    print(f"\n🔄 Generating {n_train:,} training samples...", flush=True)
    train_dataset_massive = LinkSample(n_pts=n_train)
    train_coords_massive = tf.convert_to_tensor(train_dataset_massive.link_points())
    train_g2_forms_massive = train_dataset_massive.g2_form
    train_output_vecs_massive = form_to_vec(tf.convert_to_tensor(train_g2_forms_massive))
    
    # Extract only non-zero components
    train_output_nonzero_massive = tf.gather(train_output_vecs_massive, non_zero_indices, axis=1)
    
    generation_time = time.time() - start_time
    print(f"✅ Training data generated in {generation_time:.1f} seconds", flush=True)
    
    # Test data
    print(f"\n🔄 Generating {n_test:,} test samples...", flush=True)
    test_dataset_massive = LinkSample(n_pts=n_test)
    test_coords_massive = tf.convert_to_tensor(test_dataset_massive.link_points())
    test_g2_forms_massive = test_dataset_massive.g2_form
    test_output_vecs_massive = form_to_vec(tf.convert_to_tensor(test_g2_forms_massive))
    test_output_nonzero_massive = tf.gather(test_output_vecs_massive, non_zero_indices, axis=1)
    
    test_generation_time = time.time() - start_time - generation_time
    print(f"✅ Test data generated in {test_generation_time:.1f} seconds", flush=True)
    
    print(f"\n📊 Dataset shapes:", flush=True)
    print(f"  Training: {train_coords_massive.shape} → {train_output_nonzero_massive.shape}", flush=True)
    print(f"  Test: {test_coords_massive.shape} → {test_output_nonzero_massive.shape}", flush=True)
    
    # ===============================
    # NORMALIZE DATA
    # ===============================
    
    print(f"\n🔄 Normalizing massive dataset...", flush=True)
    input_scaler_massive = StandardScaler()
    output_scaler_massive = StandardScaler()
    
    train_coords_massive_norm = input_scaler_massive.fit_transform(train_coords_massive.numpy())
    train_output_massive_norm = output_scaler_massive.fit_transform(train_output_nonzero_massive.numpy())
    test_coords_massive_norm = input_scaler_massive.transform(test_coords_massive.numpy())
    test_output_massive_norm = output_scaler_massive.transform(test_output_nonzero_massive.numpy())
    
    print("✅ Normalization complete", flush=True)
    
    # ===============================
    # BUILD MODEL
    # ===============================
    
    print(f"\n=== Building Optimal Model Architecture ===", flush=True)
    
    model_massive = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,)),
        
        # Optimal architecture from previous experiments
        tf.keras.layers.Dense(96, activation='gelu'),
        tf.keras.layers.Dropout(0.15),
        
        tf.keras.layers.Dense(64, activation='gelu'),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(32, activation='gelu'),
        
        tf.keras.layers.Dense(len(non_zero_indices), activation=None)
    ])
    
    # Compile model
    model_massive.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),
        loss='mse',
        metrics=['mae']
    )
    
    print("Model architecture:", flush=True)
    model_massive.summary()
    
    # Calculate data adequacy
    model_params = model_massive.count_params()
    samples_per_param = n_train / model_params
    
    print(f"\n📈 Data Adequacy Analysis:", flush=True)
    print(f"  Model parameters: {model_params:,}", flush=True)
    print(f"  Training samples: {n_train:,}", flush=True)
    print(f"  Samples per parameter: {samples_per_param:.1f}", flush=True)
    
    if samples_per_param >= 10:
        print(f"  ✅ {samples_per_param:.1f} ≥ 10: Should have sufficient data!", flush=True)
    else:
        print(f"  ⚠️  {samples_per_param:.1f} < 10: May still be insufficient", flush=True)
    
    # ===============================
    # SETUP TRAINING CALLBACKS
    # ===============================
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=15,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv')
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # ===============================
    # TRAIN MODEL
    # ===============================
    
    print(f"\n🚀 === Training Massive Model ===", flush=True)
    print(f"Expected improvements with {n_train:,} samples:", flush=True)
    print("✅ Better generalization (more samples per parameter)", flush=True)
    print("✅ Reduced overfitting tendency", flush=True) 
    print("✅ More stable learning curves", flush=True)
    
    training_start_time = time.time()
    
    # Train the model
    history_massive = model_massive.fit(
        train_coords_massive_norm,
        train_output_massive_norm,
        epochs=200,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - training_start_time
    print(f"\n✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)", flush=True)
    
    # ===============================
    # EVALUATE MODEL
    # ===============================
    
    print(f"\n=== Evaluating Model ===", flush=True)
    
    # Predictions on test set
    predictions_massive_norm = model_massive.predict(test_coords_massive_norm)
    predictions_massive = output_scaler_massive.inverse_transform(predictions_massive_norm)
    
    # Calculate metrics
    test_loss_massive = tf.keras.losses.MeanSquaredError()(test_output_nonzero_massive, predictions_massive)
    test_mae_massive = tf.keras.losses.MeanAbsoluteError()(test_output_nonzero_massive, predictions_massive)
    
    # Calculate MAPE
    threshold = 1e-6
    mask_massive = tf.abs(test_output_nonzero_massive) > threshold
    valid_true_massive = tf.boolean_mask(test_output_nonzero_massive, mask_massive)
    valid_pred_massive = tf.boolean_mask(predictions_massive, mask_massive)
    
    valid_true_massive = tf.cast(valid_true_massive, tf.float32)
    valid_pred_massive = tf.cast(valid_pred_massive, tf.float32)
    
    if tf.size(valid_true_massive) > 0:
        mape_massive = tf.reduce_mean(tf.abs((valid_true_massive - valid_pred_massive) / valid_true_massive)) * 100
    else:
        mape_massive = float('inf')
    
    # Learning analysis
    learning_massive = (history_massive.history['loss'][0] - history_massive.history['loss'][-1]) / history_massive.history['loss'][0] * 100
    overfit_massive = history_massive.history['val_loss'][-1] / history_massive.history['loss'][-1]
    
    # ===============================
    # SAVE RESULTS
    # ===============================
    
    print(f"\n=== Saving Results ===", flush=True)
    
    # Save metrics
    results = {
        'n_train': n_train,
        'n_test': n_test,
        'model_params': model_params,
        'samples_per_param': samples_per_param,
        'test_loss': float(test_loss_massive),
        'test_mae': float(test_mae_massive),
        'test_mape': float(mape_massive),
        'learning_improvement_pct': learning_massive,
        'overfitting_ratio': overfit_massive,
        'training_time_seconds': training_time,
        'data_generation_time_seconds': generation_time + test_generation_time
    }
    
    # Save results as text file
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write("G2 STRUCTURE LEARNING - MASSIVE SCALE RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    # Save training history plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history_massive.history['loss'], label='Training Loss', color='blue')
    plt.plot(history_massive.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history_massive.history['mae'], label='Training MAE', color='blue')
    plt.plot(history_massive.history['val_mae'], label='Validation MAE', color='orange')
    plt.title('MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    epochs = range(len(history_massive.history['loss']))
    learning_progress = [(history_massive.history['loss'][0] - history_massive.history['loss'][i]) / history_massive.history['loss'][0] * 100 for i in epochs]
    plt.plot(epochs, learning_progress, label=f'{n_train:,} samples', color='green', linewidth=2)
    plt.title('Learning Progress (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Improvement (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Plots saved to {output_dir}/training_curves.png", flush=True)
    
    # Save model
    model_massive.save(os.path.join(output_dir, 'final_model.keras'))
    print(f"✅ Model saved to {output_dir}/final_model.keras", flush=True)
    
    # ===============================
    # FINAL REPORT
    # ===============================
    
    total_time = time.time() - start_time
    
    print(f"\n🎯 === FINAL RESULTS ===", flush=True)
    print(f"Training samples: {n_train:,}", flush=True)
    print(f"Test MSE: {test_loss_massive:.6f}", flush=True)
    print(f"Test MAE: {test_mae_massive:.6f}", flush=True)
    print(f"Test MAPE: {mape_massive:.1f}%", flush=True)
    print(f"Learning improvement: {learning_massive:.1f}%", flush=True)
    print(f"Overfitting ratio: {overfit_massive:.2f}x", flush=True)
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)", flush=True)
    
    print(f"\n🎯 === VERDICT ===", flush=True)
    if learning_massive > 35 and overfit_massive < 1.3:
        print("🎉 SUCCESS: Excellent learning with good generalization!", flush=True)
        verdict = "SUCCESS"
    elif learning_massive > 25:
        print("✅ GOOD: Strong learning achieved!", flush=True)
        verdict = "GOOD"
    elif learning_massive > 15:
        print("⚠️  MODERATE: Some learning but room for improvement", flush=True)
        verdict = "MODERATE"  
    else:
        print("❌ POOR: Insufficient learning - architectural changes needed", flush=True)
        verdict = "POOR"
    
    results['verdict'] = verdict
    
    # Save final results
    np.save(os.path.join(output_dir, 'results.npy'), results)
    
    print(f"\n✅ All results saved to: {output_dir}/", flush=True)
    print("🚀" + "="*80, flush=True)
    
    return results, model_massive, history_massive

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='G2 Structure Learning - Massive Scale Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--train-samples', type=int, default=100000,
                      help='Number of training samples')
    parser.add_argument('--test-samples', type=int, default=10000,
                      help='Number of test samples')
    parser.add_argument('--output-dir', type=str, default='massive_results',
                      help='Output directory')
    
    # Parse arguments (will use defaults if no args provided)
    args = parser.parse_args()
    
    print("🚀 Starting G2 massive scale training...", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"TensorFlow version: {tf.__version__}", flush=True)
    print(f"Numpy version: {np.__version__}", flush=True)
    
    try:
        results, model, history = train_massive_g2_model(
            n_train=args.train_samples,
            n_test=args.test_samples, 
            output_dir=args.output_dir
        )
        print("🎉 Training completed successfully!", flush=True)
        return 0
    except Exception as e:
        print(f"❌ Training failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
