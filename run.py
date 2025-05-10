'''Run file for the G2-structure learning'''
# Import libraries
import os
import yaml
import numpy as np
import tensorflow as tf

# Import functions
from models.model import GlobalModel
from sampling.sampling import LinkSample
from geometry.geometry import PatchChange_Coords, PatchChange_G2form, form_to_vec

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
    train_sample, train_3form = LinkSample(
        hp["num_samples"],
    )
    train_sample_tf = tf.convert_to_tensor(train_sample)
    train_3form_tf = tf.convert_to_tensor(train_3form)
    
    if hp["validate"]:
        val_sample, val_3form = LinkSample(
            hp.num_val_samples,
        )
        val_sample_tf = tf.convert_to_tensor(val_sample)
        val_3form_tf = tf.convert_to_tensor(val_3form)

    train_sample_inputs = [train_sample_tf] 
    ### Only need below if train_sample_inputs[i] used in PatchChange_3form, otherwise delete...
    if hp["n_patches"] > 1:
        train_sample_inputs += [PatchChange_Coords(
            train_sample_tf,
            input_patch=0,
            output_patch=i,
            ) for i in range(1,5)
            ]
    
    train_3forms = [train_3form_tf]
    if hp["n_patches"] > 1:
        train_3forms += [PatchChange_G2form(
            train_sample_tf, 
            train_3form_tf, 
            output_patch=o_patch,
            ) for o_patch in range(1,5)
            ]

    # Generate validation data if required ###
    if hp["validate"]:
        #val_sample_inputs = ...
        val_3forms = [val_3form_tf]
        if hp["n_patches"] > 1:
            val_3forms += [PatchChange_G2form(
                val_sample_tf, 
                val_3form_tf, 
                output_patch=o_patch,
                ) for o_patch in range(1,5)
                ]
        raise NotImplementedError("Validation hasn't been implemented yet!")
        
    # Convert to dof vectors (vielbeins)
    train_3forms_vecs = [form_to_vec(tsm) for tsm in train_3forms]
    train_3forms_vecs_tf = tf.convert_to_tensor(tf.concat(train_3forms_vecs,axis=1))
    if hp["validate"]:
        val_3forms_vecs = [form_to_vec(vsm) for vsm in val_3forms]
        val_3forms_vecs_tf = tf.convert_to_tensor(tf.concat(val_3forms_vecs,axis=1))
        val_data = (val_sample_tf, val_3forms_vecs_tf)
    else:
        val_sample_tf = None
        val_3forms_vecs_tf = None
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
        train_3forms_vecs_tf,
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
        train_3forms_vecs_tf,
        val_data,
    )


###############################################################################
if __name__ == "__main__":
    # Supervised run hyperparameters
    save = True   #...whether to save the trained supervised model
    save_flag = 'test' #...some int to append to file names to differentiate supervised models

    # Define and train the model
    hyperparams_filepath = os.path.dirname(__file__)+'/hyperparameters/hps.yaml'
    network, lh, train_coords, train_metrics, val_data = main(hyperparams_filepath)
    print('trained.....')
    
    # Save the model
    if save == True:
        network.save(os.path.dirname(__file__)+f'/runs/supervised_model_{save_flag}.keras')
    
