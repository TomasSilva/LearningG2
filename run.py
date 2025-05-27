'''Run file for the G2-structure learning'''
# Import libraries
import os
import sys
import yaml
import numpy as np
import tensorflow as tf

# Import functions
from models.model import GlobalModel
from sampling.sampling import LinkSample
from geometry.compression import form_to_vec, metric_to_vec
from geometry.patches import PatchChange_Coords, PatchChange_G2form, PatchChange_G2metric

# Main body function for performing the metric training
def main(hyperparameters_file):
    ###########################################################################
    ### Import run hyperparameters ###
    # Load the YAML file
    with open(hyperparameters_file, "r") as file:
        hp = yaml.safe_load(file)
    
    ###########################################################################
    ### Data set-up ###
    # Create training and validation samples
    train_sample, train_output = LinkSample(
        hp["num_samples"], hp
    )
    train_sample_tf = tf.convert_to_tensor(train_sample)
    train_output_tf = tf.convert_to_tensor(train_output)
    
    if hp["validate"]:
        val_sample, val_output = LinkSample(
            hp["num_val_samples"], hp
        )
        val_sample_tf = tf.convert_to_tensor(val_sample)
        val_output_tf = tf.convert_to_tensor(val_output)
    
    train_outputs = [train_output_tf]
    if hp["n_patches"] > 1:
        if not hp["metric"]:
            train_outputs += [PatchChange_G2form(
                train_sample_tf, 
                train_output_tf, 
                output_patch=o_patch,
                ) for o_patch in range(1,5)
                ]
        else:
            train_outputs += [PatchChange_G2metric(
                train_sample_tf, 
                train_output_tf, 
                output_patch=o_patch,
                ) for o_patch in range(1,5)
                ]

    # Generate validation data if required
    if hp["validate"]:
        val_outputs = [val_output_tf]
        if hp["n_patches"] > 1:
            if not hp["metric"]:
                val_outputs += [PatchChange_G2form(
                    val_sample_tf, 
                    val_output_tf, 
                    output_patch=o_patch,
                    ) for o_patch in range(1,5)
                    ]
            else:
                val_outputs += [PatchChange_G2metric(
                    val_sample_tf, 
                    val_output_tf, 
                    output_patch=o_patch,
                    ) for o_patch in range(1,5)
                    ]
            
        
    # Convert to dof vectors (vielbeins)
    if not hp["metric"]:
        train_outputs_vecs = [form_to_vec(tsm) for tsm in train_outputs]
        train_outputs_vecs_tf = tf.convert_to_tensor(tf.concat(train_outputs_vecs, axis=1))
    else:
        train_outputs_vecs = [metric_to_vec(tsm) for tsm in train_outputs]
        train_outputs_vecs_tf = tf.convert_to_tensor(tf.concat(train_outputs_vecs, axis=1))

    if hp["validate"]:
        if not hp["metric"]:
            val_outputs_vecs = [form_to_vec(vsm) for vsm in val_outputs]
            val_outputs_vecs_tf = tf.convert_to_tensor(tf.concat(val_outputs_vecs, axis=1))
        else:
            val_outputs_vecs = [metric_to_vec(vsm) for vsm in val_outputs]
            val_outputs_vecs_tf = tf.convert_to_tensor(tf.concat(val_outputs_vecs, axis=1))
        val_data = (val_sample_tf, val_outputs_vecs_tf)
    else:
        val_sample_tf = None
        val_outputs_vecs_tf = None
        val_data = None
        
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
    
    # Import the model
    if hp["saved_model"]:
        model = tf.keras.models.load_model(hp["saved_model_path"])
        model.compile(optimizer=optimiser, loss="MSE")
        # Update imported model implicit hps
        hp["dim"]         = model.hp["dim"]
        hp["n_patches"]   = model.hp["n_patches"]
        hp["n_hidden"]    = model.hp["n_hidden"] 
        hp["n_layers"]    = model.hp["n_layers"]
        hp["activations"] = model.hp["activations"]
        hp["use_bias"]    = model.hp["use_bias"] #...these are overwritten by the import
        model.hp = hp  
        model.set_serializable_hp()        
    # Build the model
    else:
        model = GlobalModel(hp)
        model.compile(optimizer=optimiser, loss="MSE")
    
    # Train!
    loss_hist = model.fit(
        train_sample_tf,
        train_outputs_vecs_tf,
        batch_size=hp["batch_size"],
        epochs=hp["epochs"],
        verbose=hp["verbosity"],
        validation_data=val_data,
        shuffle=True,
    )

    return (
        model,
        loss_hist,
        train_sample_tf,
        train_outputs_vecs_tf,
        val_data,
    )


###############################################################################
if __name__ == "__main__":
    # Supervised run hyperparameters
    save = True   #...whether to save the trained supervised model
    save_flag = 'test_metric' #...the name of the trained supervised model
    if len(sys.argv) > 1:
        save_flag = sys.argv[1]

    # Define and train the model
    hyperparams_filepath = os.path.dirname(__file__)+'/hyperparameters/hps.yaml'
    network, lh, train_coords, train_metrics, val_data = main(hyperparams_filepath)
    print('trained.....')
    
    # Save the model
    if save == True:
        network.save(os.path.dirname(__file__)+f'/runs/link_model_{save_flag}.keras')
    
