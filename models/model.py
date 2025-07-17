'''ML architecture for the G2-structure learning'''
# Import libraries
import tensorflow as tf
from math import comb
from geometry.normalisation import Normaliser

class ScaledGlorotUniform(tf.keras.initializers.GlorotUniform):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def __call__(self, shape, dtype=None):
        return self.scale * super().__call__(shape, dtype=dtype)

class GlobalModel(tf.keras.Model):
    def __init__(self, hp, **kwargs):
        super(GlobalModel, self).__init__(**kwargs)
        # Define hyperparameters
        self.hp = hp
        self.serializable_hp = None
        self.set_serializable_hp()
        self.metric = self.hp["metric"]

        # Normalisation statistics (set via fit_normalisers)
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

        # Compute the number of independent metric entries, this is the number 
        # of vielbein entries used as the model outputs for each patch
        if self.metric:
            self.n_out = 28 #...upper triangle of symmetric matrix has: 7 * (7 + 1) / 2 entries
        else:
            self.n_out = comb(7, 3) #...rank hardcoded as 3 here
            
        # Define embedding parameters (the 20 is 5*(5-1) which is number of patches)
        self.embedding_dim = self.hp["embedding_dim"]
        self.patch_embedding = tf.keras.layers.Embedding(
            input_dim=20, output_dim=self.embedding_dim, name="patch_embedding"
        )

        # Define architecture
        coord_input = tf.keras.layers.Input(shape=(7,), name="coord_input")
        patch_input = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="patch_input")

        # Embed patch ID and concatenate with coordinates
        patch_embed = self.patch_embedding(patch_input)
        combined_input = tf.keras.layers.Concatenate()([coord_input, patch_embed])

        # Feedforward layers
        initializer = ScaledGlorotUniform(scale=self.hp["parameter_initialisation_scale"])
        x = tf.keras.layers.Dense(
            self.hp["n_hidden"], activation=self.hp["activations"], use_bias=self.hp["use_bias"], kernel_initializer=initializer
        )(combined_input)
        for _ in range(self.hp["n_layers"] - 2):
            x = tf.keras.layers.Dense(
                self.hp["n_hidden"],
                activation=self.hp["activations"],
                use_bias=self.hp["use_bias"],
                kernel_initializer=initializer
            )(x)
        outputs = tf.keras.layers.Dense(self.n_out, activation=None, use_bias=False, kernel_initializer=initializer)(x)
           
        self.model = tf.keras.Model(inputs=[coord_input, patch_input], outputs=outputs)

    def fit_normalisers(self, x, y):
        """
        Fit normalisation statistics for inputs and outputs.
        Args:
            x: tf.Tensor, shape (N, 7) input coordinates
            y: tf.Tensor, shape (N, n_out) output vielbein/metric vectors
        """
        dtype = x.dtype
        self.input_mean = tf.cast(tf.reduce_mean(x, axis=0), dtype)
        self.input_std = tf.cast(tf.math.reduce_std(x, axis=0) + 1e-8, dtype)
        self.output_mean = tf.cast(tf.reduce_mean(y, axis=0), dtype)
        self.output_std = tf.cast(tf.math.reduce_std(y, axis=0) + 1e-8, dtype)

    def call(self, inputs):
        # Unpack inputs
        coords, patch_idxs = inputs
        # Normalise coordinates
        if (self.input_mean is not None) and (self.input_std is not None):
            coords = (coords - self.input_mean) / self.input_std
        # Forward pass
        y_pred_norm = self.model([coords, patch_idxs])
        # Denormalise outputs
        if (self.output_mean is not None) and (self.output_std is not None):
            y_pred = y_pred_norm * self.output_std + self.output_mean
        else:
            y_pred = y_pred_norm
        return y_pred

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
        # Save normalisation statistics as lists for serialization
        config["input_mean"] = self.input_mean.numpy().tolist() if self.input_mean is not None else None
        config["input_std"] = self.input_std.numpy().tolist() if self.input_std is not None else None
        config["output_mean"] = self.output_mean.numpy().tolist() if self.output_mean is not None else None
        config["output_std"] = self.output_std.numpy().tolist() if self.output_std is not None else None
        return config

    @classmethod
    def from_config(cls, config):
        # Extract normalisation statistics
        input_mean = config.pop("input_mean", None)
        input_std = config.pop("input_std", None)
        output_mean = config.pop("output_mean", None)
        output_std = config.pop("output_std", None)
        model = cls(**config)
        # Restore normalisation statistics as tf.Tensor
        if input_mean is not None:
            model.input_mean = tf.convert_to_tensor(input_mean, dtype=tf.float32)
        if input_std is not None:
            model.input_std = tf.convert_to_tensor(input_std, dtype=tf.float32)
        if output_mean is not None:
            model.output_mean = tf.convert_to_tensor(output_mean, dtype=tf.float32)
        if output_std is not None:
            model.output_std = tf.convert_to_tensor(output_std, dtype=tf.float32)
        return model


