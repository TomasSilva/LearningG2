'''Restructured ML architecture with explicit normalisation layers for G2-structure learning'''
# Import libraries
import tensorflow as tf
from math import comb


class ScaledGlorotUniform(tf.keras.initializers.GlorotUniform):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def __call__(self, shape, dtype=None):
        return self.scale * super().__call__(shape, dtype=dtype)


class NormalisationLayer(tf.keras.layers.Layer):
    """Layer that applies z-score normalisation using fitted statistics"""
    
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean = None
        self.std = None
        
    def fit_statistics(self, data):
        """Fit normalisation statistics on data"""
        self.mean = tf.Variable(
            tf.reduce_mean(data, axis=0), 
            trainable=False, 
            name=f"{self.name}_mean"
        )
        self.std = tf.Variable(
            tf.math.reduce_std(data, axis=0) + 1e-8, 
            trainable=False, 
            name=f"{self.name}_std"
        )
        
    def call(self, inputs):
        if self.mean is None or self.std is None:
            raise ValueError(f"Normalisation layer {self.name} not fitted. Call fit_statistics() first.")
        return (inputs - self.mean) / self.std
        
    def get_config(self):
        config = super().get_config()
        if self.mean is not None and self.std is not None:
            config.update({
                "mean": self.mean.numpy().tolist(),
                "std": self.std.numpy().tolist()
            })
        return config
        
    @classmethod
    def from_config(cls, config):
        mean = config.pop("mean", None)
        std = config.pop("std", None)
        layer = cls(**config)
        if mean is not None and std is not None:
            layer.mean = tf.Variable(mean, trainable=False)
            layer.std = tf.Variable(std, trainable=False)
        return layer


class DenormalisationLayer(tf.keras.layers.Layer):
    """Layer that reverses z-score normalisation using fitted statistics"""
    
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean = None
        self.std = None
        
    def fit_statistics(self, data):
        """Fit denormalisation statistics on data"""
        self.mean = tf.Variable(
            tf.reduce_mean(data, axis=0), 
            trainable=False, 
            name=f"{self.name}_mean"
        )
        self.std = tf.Variable(
            tf.math.reduce_std(data, axis=0) + 1e-8, 
            trainable=False, 
            name=f"{self.name}_std"
        )
        
    def call(self, inputs):
        if self.mean is None or self.std is None:
            raise ValueError(f"Denormalisation layer {self.name} not fitted. Call fit_statistics() first.")
        return inputs * self.std + self.mean
        
    def get_config(self):
        config = super().get_config()
        if self.mean is not None and self.std is not None:
            config.update({
                "mean": self.mean.numpy().tolist(),
                "std": self.std.numpy().tolist()
            })
        return config
        
    @classmethod
    def from_config(cls, config):
        mean = config.pop("mean", None)
        std = config.pop("std", None)
        layer = cls(**config)
        if mean is not None and std is not None:
            layer.mean = tf.Variable(mean, trainable=False)
            layer.std = tf.Variable(std, trainable=False)
        return layer


class CoreNormalisedModel(tf.keras.Model):
    """Core model that operates on normalised inputs and outputs"""
    
    def __init__(self, hp, **kwargs):
        super().__init__(**kwargs)
        self.hp = hp
        self.metric = self.hp["metric"]

        # Compute the number of independent metric entries
        if self.metric:
            self.n_out = 28  # upper triangle of symmetric matrix has: 7 * (7 + 1) / 2 entries
        else:
            self.n_out = comb(7, 3)  # rank hardcoded as 3 here
            
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
            self.hp["n_hidden"], 
            activation=self.hp["activations"], 
            use_bias=self.hp["use_bias"], 
            kernel_initializer=initializer
        )(combined_input)
        
        for _ in range(self.hp["n_layers"] - 2):
            x = tf.keras.layers.Dense(
                self.hp["n_hidden"],
                activation=self.hp["activations"],
                use_bias=self.hp["use_bias"],
                kernel_initializer=initializer
            )(x)
            
        outputs = tf.keras.layers.Dense(
            self.n_out, 
            activation=None, 
            use_bias=False, 
            kernel_initializer=initializer
        )(x)
           
        self.model = tf.keras.Model(inputs=[coord_input, patch_input], outputs=outputs)

    def call(self, inputs):
        return self.model(inputs)


class GlobalNormalisedModel(tf.keras.Model):
    """Outer model that handles normalisation and contains the core normalised model"""
    
    def __init__(self, hp, **kwargs):
        super().__init__(**kwargs)
        self.hp = hp
        self.serializable_hp = None
        self.set_serializable_hp()
        
        # Create normalisation layers
        self.input_normaliser = NormalisationLayer(name="input_normaliser")
        self.output_denormaliser = DenormalisationLayer(name="output_denormaliser")
        
        # Create the core model that operates on normalised data
        self.core_model = CoreNormalisedModel(hp)
        
        # Flag to track if normalisation has been fitted
        self._normalisation_fitted = False

    def fit_normalisers(self, x, y):
        """
        Fit normalisation statistics for inputs and outputs.
        Args:
            x: tf.Tensor, shape (N, 7) input coordinates
            y: tf.Tensor, shape (N, n_out) output vielbein/metric vectors
        """
        self.input_normaliser.fit_statistics(x)
        self.output_denormaliser.fit_statistics(y)
        self._normalisation_fitted = True

    def call(self, inputs):
        """Forward pass through the full model (original scale -> normalised -> original scale)"""
        if not self._normalisation_fitted:
            raise ValueError("Normalisation not fitted. Call fit_normalisers() first.")
            
        coords, patch_idxs = inputs
        
        # Normalise inputs
        normalised_coords = self.input_normaliser(coords)
        
        # Forward pass through core model (operates on normalised data)
        normalised_outputs = self.core_model([normalised_coords, patch_idxs])
        
        # Denormalise outputs
        outputs = self.output_denormaliser(normalised_outputs)
        
        return outputs

    def call_normalised(self, inputs):
        """Forward pass that returns normalised outputs (for training at normalised scale)"""
        if not self._normalisation_fitted:
            raise ValueError("Normalisation not fitted. Call fit_normalisers() first.")
            
        coords, patch_idxs = inputs
        
        # Normalise inputs
        normalised_coords = self.input_normaliser(coords)
        
        # Forward pass through core model (operates on normalised data)
        normalised_outputs = self.core_model([normalised_coords, patch_idxs])
        
        return normalised_outputs

    def normalise_targets(self, targets):
        """Normalise target outputs for training at normalised scale"""
        if not self._normalisation_fitted:
            raise ValueError("Normalisation not fitted. Call fit_normalisers() first.")
        return (targets - self.output_denormaliser.mean) / self.output_denormaliser.std

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
        config = super().get_config()
        config.update({"hp": self.serializable_hp})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NormalisedTrainingModel(tf.keras.Model):
    """Training wrapper that operates at normalised scale"""
    
    def __init__(self, global_model, **kwargs):
        super().__init__(**kwargs)
        self.global_model = global_model
        
    def call(self, inputs):
        return self.global_model.call_normalised(inputs)
    
    def fit_with_normalised_targets(self, x, y, **fit_kwargs):
        """Fit the model using normalised targets"""
        # Normalise targets
        y_normalised = self.global_model.normalise_targets(y)
        
        # Train at normalised scale
        return super().fit(x, y_normalised, **fit_kwargs)
