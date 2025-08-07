'''Run file for the G2-structure learning'''
# Import libraries
import os
import sys
import yaml
import numpy as np
import tensorflow as tf

# Import functions
from models.normalised_model import GlobalNormalisedModel, NormalisedTrainingModel
from sampling.sampling import LinkSample
from geometry.compression import form_to_vec, metric_to_vec
from geometry.patches import patch_indices_to_scalar
from geometry.normalisation import Normaliser

# Main body function for performing the metric training
def main(hyperparameters_file):
    ###########################################################################
    ### Import run hyperparameters ###
    # Load the YAML file
    with open(hyperparameters_file, "r") as file:
        hp = yaml.safe_load(file)
    
    ###########################################################################
    ### Data set-up ###
    # Create training sample
    train_dataset = LinkSample(n_pts=hp["num_samples"])
    train_sample = tf.convert_to_tensor(train_dataset.link_points())
    train_patch_idxs = patch_indices_to_scalar(train_dataset.one_idxs, train_dataset.dropped_idxs)
    if not hp["metric"]:
        train_output = train_dataset.g2_form
        train_output_vecs = form_to_vec(tf.convert_to_tensor(train_output))
    else:
        train_output = train_dataset.g2_metric
        train_output_vecs = metric_to_vec(tf.convert_to_tensor(train_output))

    ###########################################################################
    ### Run ML ###
    # Set up optimiser
    if hp["init_learning_rate"] == hp["min_learning_rate"]:
        optimiser = tf.keras.optimizers.Adam(learning_rate=hp["init_learning_rate"])
    else:
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=hp["init_learning_rate"],
            decay_steps=1000,
            end_learning_rate=hp["min_learning_rate"],
            power=1.0
            )
        optimiser = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Create the global model (handles original scale I/O)
    if hp["saved_model"]:
        # Loading saved models would need special handling - you could either:
        # 1. Keep the old GlobalModel for loading, then transfer weights
        # 2. Implement custom loading logic for the new architecture
        raise NotImplementedError("Loading saved normalised models not yet implemented")
    else:
        global_model = GlobalNormalisedModel(hp)

    # Fit normalisers on training data (raw, unnormalised)
    global_model.fit_normalisers(train_sample, train_output_vecs)
    
    # Create training model that operates at normalised scale
    training_model = NormalisedTrainingModel(global_model)
    training_model.compile(optimizer=optimiser, loss="MSE")

    # Create validation sample
    if hp["validate"]:
        val_dataset = LinkSample(n_pts=hp["num_val_samples"])
        val_sample = tf.convert_to_tensor(val_dataset.link_points())
        val_patch_idxs = patch_indices_to_scalar(val_dataset.one_idxs, val_dataset.dropped_idxs)
        if not hp["metric"]:
            val_output = val_dataset.g2_form
            val_output_vecs = form_to_vec(tf.convert_to_tensor(val_output))
        else:
            val_output = val_dataset.g2_metric
            val_output_vecs = metric_to_vec(tf.convert_to_tensor(val_output))
        # Normalise validation targets for training
        val_output_vecs_normalised = global_model.normalise_targets(val_output_vecs)
        val_data_normalised = ([val_sample, val_patch_idxs], val_output_vecs_normalised)
    else:
        val_data_normalised = None

    # Train at normalised scale!
    loss_hist = training_model.fit_with_normalised_targets(
        [train_sample, train_patch_idxs],
        train_output_vecs,  
        batch_size=hp["batch_size"],
        epochs=hp["epochs"],
        verbose=hp["verbosity"],
        validation_data=val_data_normalised,
        shuffle=True,
    )

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
    network, lh, train_coords, train_metrics, val_data = main(hyperparams_filepath)
    print('trained.....')
    
    # Save the model
    if save == True:
        # If the runs folder for saving models doesn't exist, create it
        logging_path = os.path.dirname(__file__)+'/runs/'
        if not os.path.exists(logging_path):
            os.makedirs(logging_path)
            
        # Build the model explicitly to avoid saving warnings
        # Use the same input shapes as training
        dummy_coords = tf.zeros((1, 7), dtype=train_coords.dtype)
        dummy_patch_idx = tf.zeros((1,), dtype=tf.int32)
        network([dummy_coords, dummy_patch_idx])  # This builds the model
            
        # Save the model
        network.save(logging_path+f'link_model_{save_flag}.keras')
    
