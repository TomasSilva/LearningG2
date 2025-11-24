#!/usr/bin/env python3

####NOTE the 'constant componets are actually non-constant, so need to add these back in to be learnt (indices {11,23,30}), but these do have much lower values (1e-9 vs 1e-1)!!
"""
G2 Structure Learning - Massive Scale Training (500K+ samples)
HPC-ready standalone script for large-scale training experiment

FEATURES:
- GPU/CPU auto-detection with performance optimization
- Comprehensive data generation with LinkSample
- Advanced neural architecture with regularization
- Robust loss functions and training callbacks
- Extensive logging and result visualization
- Memory-efficient batch processing
- Component filtering to remove constant values
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

# Add parent directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import custom modules
try:
    from sampling.sampling import LinkSample
    from geometry.compression import form_to_vec
except ImportError as e:
    print(f"❌ Import error: {e}", flush=True)
    print("Make sure the parent directory is accessible", flush=True)
    sys.exit(1)

# TensorFlow configuration
print("🔧 Configuring TensorFlow...", flush=True)

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
    
    # Analyze data quality and scales
    print(f"\n=== Data Quality Analysis ===", flush=True)
    print(f"Input coordinate range: [{tf.reduce_min(train_coords_massive):.6f}, {tf.reduce_max(train_coords_massive):.6f}]", flush=True)
    print(f"Output G2 range: [{tf.reduce_min(train_output_nonzero_massive):.6f}, {tf.reduce_max(train_output_nonzero_massive):.6f}]", flush=True)
    print(f"Output mean: {tf.reduce_mean(train_output_nonzero_massive):.6f}, std: {tf.math.reduce_std(train_output_nonzero_massive):.6f}", flush=True)
    
    # Check for problematic values
    near_zero = tf.reduce_sum(tf.cast(tf.abs(train_output_nonzero_massive) < 1e-6, tf.int32))
    small_values = tf.reduce_sum(tf.cast(tf.abs(train_output_nonzero_massive) < 1e-4, tf.int32))
    total_values = tf.size(train_output_nonzero_massive)
    print(f"Near-zero values (<1e-6): {near_zero.numpy()}/{total_values.numpy()} ({near_zero.numpy()/total_values.numpy()*100:.1f}%)", flush=True)
    print(f"Small values (<1e-4): {small_values.numpy()}/{total_values.numpy()} ({small_values.numpy()/total_values.numpy()*100:.1f}%)", flush=True)
    
    # ===============================
    # HARDCODE FIX: EXCLUDE KNOWN CONSTANT COMPONENTS
    # ===============================
    
    print(f"\n🔧 === APPLYING HARDCODED COMPONENT FILTERING ===", flush=True)
    
    # Based on your output, components 11, 23, 30 are always constant
    # Let's explicitly exclude them from the non_zero_indices
    original_non_zero_indices = [0, 3, 6, 11, 12, 13, 14, 15, 20, 21, 23, 24, 25, 27, 29, 30, 31]
    constant_component_indices = [11, 23, 30]  # These always have std=0.0000
    
    # Filter out the constant components
    filtered_non_zero_indices = [idx for idx in original_non_zero_indices if idx not in constant_component_indices]
    
    print(f"🗑️  Removing known constant components: {constant_component_indices}")
    print(f"✅ Active components: {filtered_non_zero_indices}")
    print(f"📊 Reduction: {len(original_non_zero_indices)} → {len(filtered_non_zero_indices)} components")
    
    # Extract only the truly variable components
    train_output_filtered = tf.gather(train_output_vecs_massive, filtered_non_zero_indices, axis=1)
    test_output_filtered = tf.gather(test_output_vecs_massive, filtered_non_zero_indices, axis=1)
    
    # Verify no constant components remain
    filtered_stds = tf.math.reduce_std(train_output_filtered, axis=0)
    min_std = tf.reduce_min(filtered_stds)
    
    print(f"🔍 Verification:")
    print(f"  Filtered data shape: {train_output_filtered.shape}")
    print(f"  Minimum std in filtered data: {min_std.numpy():.2e}")
    print(f"  All components variable: {min_std.numpy() > 1e-10}")
    
    # Update variables for the rest of the pipeline
    train_output_nonzero_massive = train_output_filtered
    test_output_nonzero_massive = test_output_filtered
    non_zero_indices = filtered_non_zero_indices
    n_filtered_components = len(filtered_non_zero_indices)

    # ===============================
    # NORMALIZE DATA
    # ===============================
    
    print(f"\n🔄 Normalizing filtered dataset ({n_filtered_components} components)...", flush=True)
    input_scaler_massive = StandardScaler()
    output_scaler_massive = StandardScaler()
    
    # Normalize data
    train_coords_massive_norm = input_scaler_massive.fit_transform(train_coords_massive.numpy())
    train_output_massive_norm = output_scaler_massive.fit_transform(train_output_nonzero_massive.numpy())
    test_coords_massive_norm = input_scaler_massive.transform(test_coords_massive.numpy())
    test_output_massive_norm = output_scaler_massive.transform(test_output_nonzero_massive.numpy())
    
    # CRITICAL: Verify normalization worked
    print(f"\n🔍 === NORMALIZATION VERIFICATION ===", flush=True)
    print(f"Input normalization:", flush=True)
    print(f"  Mean: {np.mean(train_coords_massive_norm, axis=0)[:3]} (should be ~0)", flush=True)
    print(f"  Std:  {np.std(train_coords_massive_norm, axis=0)[:3]} (should be ~1)", flush=True)
    
    print(f"Output normalization:", flush=True)
    output_means = np.mean(train_output_massive_norm, axis=0)
    output_stds = np.std(train_output_massive_norm, axis=0)
    print(f"  Mean range: [{np.min(output_means):.2e}, {np.max(output_means):.2e}] (should be ~0)", flush=True)
    print(f"  Std range:  [{np.min(output_stds):.2e}, {np.max(output_stds):.2e}] (should be ~1)", flush=True)
    print(f"  All means < 0.1: {np.all(np.abs(output_means) < 0.1)}", flush=True)
    print(f"  All stds > 0.9: {np.all(output_stds > 0.9)}", flush=True)
    
    if not (np.all(np.abs(output_means) < 0.1) and np.all(output_stds > 0.9)):
        print(f"⚠️  WARNING: Normalization may have failed!", flush=True)
    else:
        print(f"✅ Normalization successful!", flush=True)
    
    print("✅ Normalization complete", flush=True)
    
    # Analyze normalized data scales  
    print(f"\n=== Normalized Data Analysis ===", flush=True)
    print(f"Normalized input mean: {np.mean(train_coords_massive_norm, axis=0)[:3]} (should be ~0)", flush=True)
    print(f"Normalized input std:  {np.std(train_coords_massive_norm, axis=0)[:3]} (should be ~1)", flush=True)
    print(f"Normalized output mean: {np.mean(train_output_massive_norm, axis=0)[:3]} (should be ~0)", flush=True)
    print(f"Normalized output std:  {np.std(train_output_massive_norm, axis=0)[:3]} (should be ~1)", flush=True)
    
    # ===============================
    # BUILD MODEL
    # ===============================
    
    print(f"\n=== Building Advanced Model Architecture ===", flush=True)
    
    # Build advanced model with residual connections and attention-like mechanism
    inputs = tf.keras.layers.Input(shape=(7,))
    
    # Initial feature extraction
    x = tf.keras.layers.Dense(256, activation='swish')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # Residual blocks for better gradient flow
    def residual_block(x, units, dropout=0.1):
        residual = tf.keras.layers.Dense(units, activation=None)(x)
        x = tf.keras.layers.Dense(units, activation='swish')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(units, activation=None)(x)
        x = tf.keras.layers.Add()([x, residual])  # Residual connection
        x = tf.keras.layers.Activation('swish')(x)
        return x
    
    # Multiple residual blocks
    x = residual_block(x, 256, 0.15)
    x = tf.keras.layers.Dense(192, activation='swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = residual_block(x, 192, 0.1)
    x = tf.keras.layers.Dense(128, activation='swish')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = residual_block(x, 128, 0.05)
    
    # Output head with separate branches for different scales
    # Major components branch (high variance)
    major_features = tf.keras.layers.Dense(64, activation='swish', name='major_features')(x)
    major_features = tf.keras.layers.Dropout(0.05)(major_features)
    
    # Minor components branch (low variance)  
    minor_features = tf.keras.layers.Dense(32, activation='swish', name='minor_features')(x)
    minor_features = tf.keras.layers.Dropout(0.05)(minor_features)
    
    # Combine features
    combined = tf.keras.layers.Concatenate()([major_features, minor_features])
    combined = tf.keras.layers.Dense(64, activation='swish')(combined)
    
    # Final output
    outputs = tf.keras.layers.Dense(n_filtered_components, activation=None, name='predictions')(combined)
    
    model_massive = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use more sophisticated loss with adaptive weighting
    class AdaptiveLoss(tf.keras.losses.Loss):
        def __init__(self, alpha=0.5, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            
        def call(self, y_true, y_pred):
            # Standard MSE
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # MAE component for robustness
            mae = tf.reduce_mean(tf.abs(y_true - y_pred))
            
            # Combine losses
            return self.alpha * mse + (1 - self.alpha) * mae
    
    # Use more conservative but effective learning
    initial_lr = 0.015  # Slightly higher
    model_massive.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=initial_lr, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-7,
            clipnorm=1.0  # Gradient clipping for stability
        ),
        loss=AdaptiveLoss(alpha=0.7),  # MSE + MAE hybrid
        metrics=['mae', 'mse']
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
    # SETUP ADVANCED TRAINING STRATEGY
    # ===============================
    
    # Cosine annealing with warm restarts
    def cosine_annealing_with_restarts(epoch, lr):
        """Cosine annealing with warm restarts for better exploration"""
        restart_period = 50
        t_cur = epoch % restart_period
        t_i = restart_period
        lr_min = 1e-6
        lr_max = initial_lr
        
        if t_cur == 0:
            return lr_max
        else:
            return lr_min + (lr_max - lr_min) * (1 + np.cos(np.pi * t_cur / t_i)) / 2
    
    # Custom callback for monitoring training quality
    class TrainingQualityCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.best_loss_improvement = 0
            
        def on_epoch_end(self, epoch, logs=None):
            if epoch > 10:
                current_improvement = (self.model.history.history['loss'][0] - logs['loss']) / self.model.history.history['loss'][0]
                if current_improvement > self.best_loss_improvement:
                    self.best_loss_improvement = current_improvement
                    if epoch % 20 == 0:
                        print(f"Best learning improvement so far: {current_improvement*100:.1f}%")
    
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_restarts, verbose=0),
        
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=60,  # More patience for cosine restarts
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-6
        ),
        
        # Plateau reduction as backup
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,
            patience=25,
            min_lr=1e-7,
            verbose=1,
            cooldown=10
        ),
        
        TrainingQualityCallback(),
        
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'training_log.csv')
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # ===============================
    # TRAIN MODEL
    # ===============================
    
    print(f"\n🚀 === Training Advanced Model ===", flush=True)
    print(f"Advanced improvements:", flush=True)
    print("✅ Residual connections for better gradient flow", flush=True)
    print("✅ Separate feature branches for different scales", flush=True) 
    print("✅ Adaptive MSE+MAE loss function", flush=True)
    print("✅ Cosine annealing with warm restarts", flush=True)
    print("✅ Gradient clipping for stability", flush=True)
    print("✅ Extended training with quality monitoring", flush=True)
    
    # FINAL VERIFICATION: Check training data normalization
    print(f"\n🔍 Pre-training data check:", flush=True)
    print(f"  Training input shape: {train_coords_massive_norm.shape}", flush=True)
    print(f"  Training output shape: {train_output_massive_norm.shape}", flush=True)
    print(f"  Output means: {np.mean(train_output_massive_norm, axis=0)[:5]}", flush=True)
    print(f"  Output stds: {np.std(train_output_massive_norm, axis=0)[:5]}", flush=True)
    
    training_start_time = time.time()
    
    # Train with advanced settings
    history_massive = model_massive.fit(
        train_coords_massive_norm,
        train_output_massive_norm,
        epochs=300,  # More epochs for cosine annealing
        batch_size=512,  # Larger batch for better gradient estimates
        validation_split=0.15,  # Less validation data for more training
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    training_time = time.time() - training_start_time
    print(f"\n✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)", flush=True)
    
    # ===============================
    # EVALUATE MODEL
    # ===============================
    
    print(f"\n=== Evaluating Model ===", flush=True)
    
    # Predictions on test set (normalized)
    predictions_massive_norm = model_massive.predict(test_coords_massive_norm)
    
    # Transform back to original space for evaluation
    predictions_massive = output_scaler_massive.inverse_transform(predictions_massive_norm)
    test_output_original = test_output_nonzero_massive.numpy()  # Original space targets
    
    # VERIFICATION: Check that we're using the right data shapes
    print(f"🔍 Data shape verification:", flush=True)
    print(f"  Predictions shape: {predictions_massive.shape}", flush=True)
    print(f"  Targets shape: {test_output_original.shape}", flush=True)
    print(f"  Shapes match: {predictions_massive.shape == test_output_original.shape}", flush=True)
    
    # Scale comparison analysis - use filtered indices correctly
    print(f"\n=== Scale Comparison Analysis ===", flush=True)
    print(f"Target scales (mean ± std):")
    for i, idx in enumerate(filtered_non_zero_indices):
        target_mean = np.mean(test_output_original[:, i])
        target_std = np.std(test_output_original[:, i])
        pred_mean = np.mean(predictions_massive[:, i])
        pred_std = np.std(predictions_massive[:, i])
        print(f"  Component {idx:2d}: Target={target_mean:8.4f}±{target_std:6.4f}, Pred={pred_mean:8.4f}±{pred_std:6.4f}")
    
    # Calculate metrics in original space
    test_loss_massive = tf.keras.losses.MeanSquaredError()(test_output_original, predictions_massive)
    test_mae_massive = tf.keras.losses.MeanAbsoluteError()(test_output_original, predictions_massive)
    
    # Calculate robust MAPE avoiding epsilon artifacts
    abs_error = np.abs(predictions_massive - test_output_original)
    abs_target = np.abs(test_output_original)
    
    # Use dynamic threshold based on data (5th percentile)
    robust_threshold = np.percentile(abs_target, 5)
    mask_massive = abs_target > robust_threshold
    
    if np.sum(mask_massive) > 0:
        mape_massive = 100 * np.mean(abs_error[mask_massive] / abs_target[mask_massive])
        valid_components = np.sum(mask_massive)
    else:
        mape_massive = float('inf')
        valid_components = 0
    
    print(f"Robust MAPE (>{robust_threshold:.2e}): {mape_massive:.2f}% on {valid_components} values")
    
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
