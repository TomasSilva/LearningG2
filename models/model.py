'''ML architecture for the G2-structure learning'''
# Import libraries
import tensorflow as tf
from math import comb

# Import functions
from geometry.geometry import PatchChange_Coords, PatchChange_G2form

class PatchSubModel(tf.keras.Model):
    def __init__(self, hp, n_out, **kwargs):
        super(PatchSubModel, self).__init__(**kwargs)
        # Define hyperparameters
        self.hp = hp
        self.serializable_hp = None
        self.set_serializable_hp()
        
        # Define subnetwork architecture
        inputs = tf.keras.layers.Input(shape=(7,))
        x = tf.keras.layers.Dense(
            self.hp["n_hidden"], activation=self.hp["activations"], use_bias=self.hp["use_bias"]
        )(inputs)
        for _ in range(self.hp["n_layers"] - 2):
            x = tf.keras.layers.Dense(
                self.hp["n_hidden"],
                activation=self.hp["activations"],
                use_bias=self.hp["use_bias"]
            )(x)
        outputs = tf.keras.layers.Dense(n_out, activation=None, use_bias=False)(x)

        self.submodel = tf.keras.Model(inputs=inputs, outputs=outputs)

    def call(self, inputs):
        return self.submodel(inputs)

    def _is_serializable(self, value):
        try:
            tf.keras.utils.serialize_keras_object(value)
            return True
        except (TypeError, ValueError):
            return False

    def set_serializable_hp(self):
        self.serializable_hp = {
            key: value for key, value in self.hp.items() if self._is_serializable(value)
        }

    def get_config(self):
        # Return the configuration necessary to recreate this model
        config = super(GlobalModel, self).get_config()
        config.update({"hp": self.serializable_hp, "n_out": self.hp["n_out"]})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GlobalModel(tf.keras.Model):
    def __init__(self, hp, **kwargs):
        super(GlobalModel, self).__init__(**kwargs)
        # Define hyperparameters
        self.hp = hp
        self.serializable_hp = None
        self.set_serializable_hp()
        self.n_patches = self.hp["n_patches"]
        self.metric = self.hp["metric"]

        # Compute the number of independent metric entries, this is the number 
        # of vielbein entries used as the model outputs for each patch
        if self.metric:
            n_out = 28 #...upper triangle of symmetric matrix has: 7 * (7 + 1) / 2 entries
        else:
            n_out = comb(7, 3) #...rank hardcoded as 3 here

        # Define submodels for each patch
        self.patch_submodels = [PatchSubModel(self.hp, n_out) for _ in range(self.n_patches)]
        if self.n_patches > 1:
            self.patch_transform_layers = [
                tf.keras.layers.Lambda(
                    lambda *args: PatchChange_Coords(
                        *args, output_patch=patch_idx,
                    ),
                ) for patch_idx in range(1, self.n_patches)]

    def call(self, inputs):
        # Transform input data to all patches
        patch_inputs = [inputs]
        if self.n_patches > 1:
            patch_inputs += [self.patch_transform_layers[i](inputs) for i in range(self.n_patches)]

        # Compute the outputs for all patches
        concatenated_output = tf.keras.layers.Concatenate()([
            self.patch_submodels[i](patch_inputs[i]) for i in range(self.n_patches)
            ])

        return concatenated_output

    def _is_serializable(self, value):
        try:
            tf.keras.utils.serialize_keras_object(value)
            return True
        except (TypeError, ValueError):
            return False

    def set_serializable_hp(self):
        self.serializable_hp = {
            key: value for key, value in self.hp.items() if self._is_serializable(value)
        }

    def get_config(self):
        # Return the configuration necessary to recreate this model
        config = super(GlobalModel, self).get_config()
        config.update({"hp": self.serializable_hp})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


