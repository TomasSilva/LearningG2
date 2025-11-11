'''Run file for the G2-structure learning'''
# Import libraries
import os
import sys
import yaml
import numpy as np
import tensorflow as tf

# Import functions
from models.model import (
    GlobalModel, TrainingModel, NormalisationLayer, 
    DenormalisationLayer, NormalisedModel, ScaledGlorotUniform
)
from sampling.sampling import LinkSample
from geometry.compression import form_to_vec, metric_to_vec
# from geometry.patches import patch_indices_to_scalar  # DEPRECATED: now using separate embeddings
from geometry.normalisation import Normaliser

def weighted_mse_loss(zero_weight=0.1):
    """MSE loss with reduced weight on structurally zero entries"""
    # Indices where outputs are identically zero
    zero_indices = tf.constant([1, 2, 4, 5, 7, 8, 9, 10, 16, 17, 18, 19, 22, 26, 28, 32, 33, 34], dtype=tf.int32)
    
    def loss_fn(y_true, y_pred):
        # Create weight tensor: 1.0 for all indices initially
        weights = tf.ones_like(y_true)
        
        # Create mask for zero indices
        # For each position in the output vector, check if it's in zero_indices
        output_dim = tf.shape(y_true)[-1]
        indices_range = tf.range(output_dim)
        
        # Create boolean mask: True where index is in zero_indices
        zero_mask = tf.reduce_any(
            tf.equal(indices_range[:, None], zero_indices[None, :]), axis=1
        )
        
        # Apply reduced weight to zero indices
        weights = tf.where(zero_mask, zero_weight, 1.0)
        
        # Compute weighted MSE loss
        squared_diff = tf.square(y_true - y_pred)
        weighted_loss = squared_diff * weights
        return tf.reduce_mean(weighted_loss)
    
    return loss_fn

# Main body function for performing the metric training
def main(hyperparameters_file):
    ###########################################################################
    ### Import run hyperparameters ###
    # Load the YAML file
    with open(hyperparameters_file, "r") as file:
        hp = yaml.safe_load(file)
    
    # Get number of data resamples (default to 1 for backward compatibility)
    n_resamples = hp.get("n_data_resamples", 1)
    
    # Get target patch filter (default to None for all patches)
    target_patch = hp.get("target_patch", None)
    if target_patch is not None:
        target_patch = tuple(target_patch)  # Convert list to tuple
        print(f"▸ Target patch: [{target_patch[0]}, {target_patch[1]}]")
    
    ###########################################################################
    ### Set up optimiser (before loop) ###
    if hp["init_learning_rate"] == hp["min_learning_rate"]:
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=hp["init_learning_rate"],
            clipnorm=1.0  # Gradient clipping for training stability
        )
    else:
        # Cosine annealing with warm restarts
        # Scale first_decay_steps proportionally to total epochs
        base_cycle_length = max(50, hp["epochs"] // 20)  # At least 50 epochs per cycle
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=hp["init_learning_rate"],
            first_decay_steps=base_cycle_length,  # Scales with total epochs
            t_mul=1.5,  # Increase cycle length by 1.5x each restart
            m_mul=0.8,  # Reduce max LR by 20% each restart
            alpha=hp["min_learning_rate"] / hp["init_learning_rate"]
        )
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0  # Gradient clipping for training stability
        )
        
        # Calculate restart epochs for display
        n_restarts = hp['epochs'] // base_cycle_length
        restart_epochs = [int(base_cycle_length * (1.5**i - 1) / 0.5) for i in range(1, n_restarts + 1)]
        restart_str = ", ".join([str(e) for e in restart_epochs[:5]])  # Show first 5
        if n_restarts > 5:
            restart_str += f", ... ({n_restarts} total)"
        
        print(f"▸ LR schedule: {base_cycle_length}ep cycles, restarts at epochs: {restart_str}")
    
    # Create the global model (handles original scale I/O)
    if hp["saved_model"]:
        # Load the saved global model (includes normalisation layers)
        model_path = os.path.join(os.path.dirname(__file__), 'runs', hp["saved_model"])
        print(f"▸ Loading: {model_path}")
        
        # Custom objects for loading
        custom_objects = {
            'GlobalModel': GlobalModel,
            'NormalisationLayer': NormalisationLayer,
            'DenormalisationLayer': DenormalisationLayer,
            'NormalisedModel': NormalisedModel,
            'ScaledGlorotUniform': ScaledGlorotUniform
        }
        
        try:
            global_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("  ✓ Model loaded (normalisation layers preserved)")
            normalisers_already_fitted = True
        except Exception as e:
            print(f"  ✗ Load failed: {e}")
            print("  Creating new model...")
            global_model = GlobalModel(hp)
            normalisers_already_fitted = False
    else:
        global_model = GlobalModel(hp)
        normalisers_already_fitted = False
    
    # Create training model that operates at normalised scale
    training_model = TrainingModel(global_model)
    training_model.compile(optimizer=optimiser, loss=weighted_mse_loss(zero_weight=0.))

    ###########################################################################
    ### Generate fixed validation set ###
    if hp["validate"]:
        print("  Generating fixed validation set...")
        val_dataset = LinkSample(n_pts=hp["num_val_samples"], target_patch=target_patch, dataset_type='val')
        val_sample = tf.convert_to_tensor(val_dataset.link_points())
        val_patch_indices = tf.stack([
            tf.convert_to_tensor(val_dataset.one_idxs, dtype=tf.int32),
            tf.convert_to_tensor(val_dataset.dropped_idxs, dtype=tf.int32)
        ], axis=1)
        if not hp["metric"]:
            val_output = val_dataset.g2_form
            val_output_vecs = form_to_vec(tf.convert_to_tensor(val_output))
        else:
            val_output = val_dataset.g2_metric
            val_output_vecs = metric_to_vec(tf.convert_to_tensor(val_output))
    
    ###########################################################################
    ### Data resampling loop ###
    if n_resamples > 1:
        print(f"\n{'='*80}")
        print(f"▸ Training: {n_resamples} resamples × {hp['epochs']} epochs = {n_resamples * hp['epochs']} effective epochs")
        print(f"{'='*80}\n")
    
    for resample_idx in range(n_resamples):
        if n_resamples > 1:
            print(f"\n{'─'*80}")
            print(f"▸▸ RESAMPLE {resample_idx + 1}/{n_resamples}")
            print(f"{'─'*80}")
        
        # Generate fresh training data
        train_dataset = LinkSample(n_pts=hp["num_samples"], target_patch=target_patch, dataset_type='train')
        train_sample = tf.convert_to_tensor(train_dataset.link_points())
        # Stack patch indices into 2D vector [one_idx, dropped_idx]
        train_patch_indices = tf.stack([
            tf.convert_to_tensor(train_dataset.one_idxs, dtype=tf.int32),
            tf.convert_to_tensor(train_dataset.dropped_idxs, dtype=tf.int32)
        ], axis=1)
        if not hp["metric"]:
            train_output = train_dataset.g2_form
            train_output_vecs = form_to_vec(tf.convert_to_tensor(train_output))
        else:
            train_output = train_dataset.g2_metric
            train_output_vecs = metric_to_vec(tf.convert_to_tensor(train_output))

        # Fit normalisers on first iteration only (keep original statistics)
        if resample_idx == 0 and not normalisers_already_fitted:
            print("  Fitting normalisation layers...")
            global_model.fit_normalisers(train_sample, train_output_vecs)
            
            # Normalize validation data once after normalisers are fitted
            if hp["validate"]:
                val_output_vecs_normalised = global_model.normalise_targets(val_output_vecs)
                val_data_normalised = ([val_sample, val_patch_indices], val_output_vecs_normalised)
        
        # Set validation data (None if not validating)
        if not hp["validate"]:
            val_data_normalised = None

        # Train (continues from previous weights if resample_idx > 0)
        if n_resamples > 1:
            print(f"  Training epochs {resample_idx * hp['epochs'] + 1}-{(resample_idx + 1) * hp['epochs']}...")
        
        loss_hist = training_model.fit_with_normalised_targets(
            [train_sample, train_patch_indices],
            train_output_vecs,  
            batch_size=hp["batch_size"],
            epochs=hp["epochs"],
            verbose=hp["verbosity"],
            validation_data=val_data_normalised,
            shuffle=True,
        )
        
        if n_resamples > 1:
            final_train_loss = loss_hist.history['loss'][-1]
            if hp["validate"]:
                final_val_loss = loss_hist.history['val_loss'][-1]
                print(f"  ✓ Resample {resample_idx + 1}/{n_resamples}: train={final_train_loss:.6f}, val={final_val_loss:.6f}")
            else:
                print(f"  ✓ Resample {resample_idx + 1}/{n_resamples}: train={final_train_loss:.6f}")
    
    # Print final summary
    final_train_loss = loss_hist.history['loss'][-1]
    if n_resamples == 1:
        if hp["validate"]:
            final_val_loss = loss_hist.history['val_loss'][-1]
            print(f"\nFinal: train={final_train_loss:.6f}, val={final_val_loss:.6f}")
        else:
            print(f"\nFinal: train={final_train_loss:.6f}")

    return (
        global_model,  
        loss_hist,
        train_sample,
        train_output_vecs,
        val_data_normalised,
    )


###############################################################################
if __name__ == "__main__":
    # Supervised run hyperparameters
    save = True   #...whether to save the trained supervised model
    save_flag = 'test' #...the name of the trained supervised model
    if len(sys.argv) > 1:
        save_flag = sys.argv[1]

    # Define and train the model
    hyperparams_filepath = os.path.dirname(__file__)+'/hyperparameters/hps.yaml'
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")
    
    global_model, lh, train_coords, train_metrics, val_data = main(hyperparams_filepath)
    
    print(f"\n{'='*80}")
    print("✓ Training complete!")
    print(f"{'='*80}\n")
    
    # Save the model
    if save == True:
        # If the runs folder for saving models doesn't exist, create it
        logging_path = os.path.dirname(__file__)+'/runs/'
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
            
        # Build the model explicitly to avoid saving warnings
        # Use the same input shapes as training (2D patch indices vector)
        dummy_coords = tf.zeros((1, 7), dtype=train_coords.dtype)
        dummy_patch_indices = tf.zeros((1, 2), dtype=tf.int32)
        global_model([dummy_coords, dummy_patch_indices])  # This builds the model
            
        # Save the full external model (includes normalisation layers)
        save_path = logging_path+f'global_model_{save_flag}.keras'
        global_model.save(save_path)
        print(f'▸ Model saved: {save_path}')
    
