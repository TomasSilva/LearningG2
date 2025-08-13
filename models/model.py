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
        # Validation
        if data is None or tf.size(data) == 0:
            raise ValueError("Cannot fit statistics on empty data")
        if tf.reduce_any(tf.math.is_nan(data)):
            raise ValueError("Data contains NaN values")
            
        # Use the same dtype as the input data
        dtype = data.dtype
        
        # Ensure we have a valid name for variables
        base_name = self.name if self.name else "normalisation"
        
        mean_value = tf.cast(tf.reduce_mean(data, axis=0), dtype)
        std_value = tf.cast(tf.math.reduce_std(data, axis=0) + 1e-8, dtype)
        
        self.mean = tf.Variable(
            mean_value, 
            trainable=False, 
            name=f"{base_name}_mean"
        )
        self.std = tf.Variable(
            std_value, 
            trainable=False, 
            name=f"{base_name}_std"
        )
        
    def call(self, inputs):
        if self.mean is None or self.std is None:
            raise ValueError(f"Normalisation layer {self.name} not fitted. Call fit_statistics() first.")
        # Cast mean and std to match input dtype
        mean = tf.cast(self.mean, inputs.dtype)
        std = tf.cast(self.std, inputs.dtype)
        return (inputs - mean) / std
        
    def get_config(self):
        config = super().get_config()
        if self.mean is not None and self.std is not None:
            config.update({
                "mean": self.mean.numpy().tolist(),
                "std": self.std.numpy().tolist(),
                "dtype": str(self.mean.dtype.name)  # Preserve original dtype
            })
        return config
        
    @classmethod
    def from_config(cls, config):
        mean = config.pop("mean", None)
        std = config.pop("std", None)
        dtype_name = config.pop("dtype", "float32")  # Default to float32 if not specified
        
        layer = cls(**config)
        if mean is not None and std is not None:
            # Use original dtype or specified dtype
            try:
                dtype = getattr(tf, dtype_name)
            except AttributeError:
                dtype = tf.float32  # Fallback
                
            base_name = layer.name if layer.name else "normalisation"
            layer.mean = tf.Variable(
                tf.cast(mean, dtype), 
                trainable=False,
                name=f"{base_name}_mean"
            )
            layer.std = tf.Variable(
                tf.cast(std, dtype), 
                trainable=False,
                name=f"{base_name}_std"
            )
        return layer


class DenormalisationLayer(tf.keras.layers.Layer):
    """Layer that reverses z-score normalisation using fitted statistics"""
    
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mean = None
        self.std = None
        
    def fit_statistics(self, data):
        """Fit denormalisation statistics on data"""
        # Validation
        if data is None or tf.size(data) == 0:
            raise ValueError("Cannot fit statistics on empty data")
        if tf.reduce_any(tf.math.is_nan(data)):
            raise ValueError("Data contains NaN values")
            
        # Use the same dtype as the input data
        dtype = data.dtype
        
        # Ensure we have a valid name for variables
        base_name = self.name if self.name else "denormalisation"
        
        mean_value = tf.cast(tf.reduce_mean(data, axis=0), dtype)
        std_value = tf.cast(tf.math.reduce_std(data, axis=0) + 1e-8, dtype)
        
        self.mean = tf.Variable(
            mean_value, 
            trainable=False, 
            name=f"{base_name}_mean"
        )
        self.std = tf.Variable(
            std_value, 
            trainable=False, 
            name=f"{base_name}_std"
        )
        
    def call(self, inputs):
        if self.mean is None or self.std is None:
            raise ValueError(f"Denormalisation layer {self.name} not fitted. Call fit_statistics() first.")
        # Cast mean and std to match input dtype
        mean = tf.cast(self.mean, inputs.dtype)
        std = tf.cast(self.std, inputs.dtype)
        return inputs * std + mean
        
    def get_config(self):
        config = super().get_config()
        if self.mean is not None and self.std is not None:
            config.update({
                "mean": self.mean.numpy().tolist(),
                "std": self.std.numpy().tolist(),
                "dtype": str(self.mean.dtype.name)  # Preserve original dtype
            })
        return config
        
    @classmethod
    def from_config(cls, config):
        mean = config.pop("mean", None)
        std = config.pop("std", None)
        dtype_name = config.pop("dtype", "float32")  # Default to float32 if not specified
        
        layer = cls(**config)
        if mean is not None and std is not None:
            # Use original dtype or specified dtype
            try:
                dtype = getattr(tf, dtype_name)
            except AttributeError:
                dtype = tf.float32  # Fallback
                
            base_name = layer.name if layer.name else "denormalisation"
            layer.mean = tf.Variable(
                tf.cast(mean, dtype), 
                trainable=False,
                name=f"{base_name}_mean"
            )
            layer.std = tf.Variable(
                tf.cast(std, dtype), 
                trainable=False,
                name=f"{base_name}_std"
            )
        return layer


class NormalisedModel(tf.keras.Model):
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
    
    def get_config(self):
        """Get configuration for saving"""
        config = super().get_config()
        # Create serializable hp dict
        serializable_hp = {}
        for key, value in self.hp.items():
            try:
                tf.keras.utils.serialize_keras_object(value)
                serializable_hp[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable values
                pass
        
        config.update({
            'hp': serializable_hp,
            'metric': self.metric,
            'n_out': self.n_out,
            'embedding_dim': self.embedding_dim
        })
        return config
    
    @classmethod 
    def from_config(cls, config, custom_objects=None):
        """Create model from configuration"""
        hp_data = config.pop('hp')
        
        # Create mock hp object with required methods
        class MockHP:
            def __init__(self, data):
                self._data = dict(data)
                for k, v in self._data.items():
                    setattr(self, k, v)
            
            def items(self):
                return self._data.items()
            
            def __getitem__(self, key):
                return self._data[key]
        
        hp = MockHP(hp_data)
        return cls(hp, **config)


class GlobalModel(tf.keras.Model):
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
        self.normalised_model = NormalisedModel(hp)
        
        # Flag to track if normalisation has been fitted
        self._normalisation_fitted = False

    def fit_normalisers(self, x, y):
        """
        Fit normalisation statistics for inputs and outputs.
        Args:
            x: tf.Tensor, shape (N, 7) input coordinates
            y: tf.Tensor, shape (N, n_out) output vielbein/metric vectors
        """
        # Validation
        if x is None or y is None:
            raise ValueError("Input data (x, y) cannot be None")
        if len(x.shape) != 2 or x.shape[1] != 7:
            raise ValueError(f"Expected x shape (N, 7), got {x.shape}")
        if len(y.shape) != 2:
            raise ValueError(f"Expected y shape (N, n_out), got {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch size mismatch: x={x.shape[0]}, y={y.shape[0]}")
        if x.shape[0] == 0:
            raise ValueError("Cannot fit normalizers on empty data")
            
        self.input_normaliser.fit_statistics(x)
        self.output_denormaliser.fit_statistics(y)
        self._normalisation_fitted = True
        
        # Build the model now that normalisation is fitted
        self.build(input_shape=[(None, 7), (None,)])

    @property
    def is_normalisation_fitted(self):
        """Check if normalization layers are properly fitted"""
        return (self._normalisation_fitted and 
                self.input_normaliser.mean is not None and 
                self.input_normaliser.std is not None and
                self.output_denormaliser.mean is not None and 
                self.output_denormaliser.std is not None)

    def call(self, inputs):
        """Forward pass through the full model (original scale -> normalised -> original scale)"""
        if not self.is_normalisation_fitted:
            raise ValueError("Normalisation not fitted. Call fit_normalisers() first.")
            
        coords, patch_idxs = inputs
        
        # Normalise inputs
        normalised_coords = self.input_normaliser(coords)
        
        # Forward pass through core model (operates on normalised data)
        normalised_outputs = self.normalised_model([normalised_coords, patch_idxs])
        
        # Denormalise outputs
        outputs = self.output_denormaliser(normalised_outputs)
        
        return outputs

    def call_normalised(self, inputs):
        """Forward pass that returns normalised outputs (for training at normalised scale)"""
        if not self.is_normalisation_fitted:
            raise ValueError("Normalisation not fitted. Call fit_normalisers() first.")
            
        coords, patch_idxs = inputs
        
        # Normalise inputs
        normalised_coords = self.input_normaliser(coords)
        
        # Forward pass through core model (operates on normalised data)
        normalised_outputs = self.normalised_model([normalised_coords, patch_idxs])
        
        return normalised_outputs

    def normalise_targets(self, targets):
        """Normalise target outputs for training at normalised scale"""
        if not self.is_normalisation_fitted:
            raise ValueError("Normalisation not fitted. Call fit_normalisers() first.")
        # Cast mean and std to match targets dtype
        mean = tf.cast(self.output_denormaliser.mean, targets.dtype)
        std = tf.cast(self.output_denormaliser.std, targets.dtype)
        return (targets - mean) / std

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
        """Get configuration for saving"""
        config = super().get_config()
        config.update({
            'serializable_hp': self.serializable_hp,
            'normalisation_fitted': self._normalisation_fitted
        })
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create model from configuration"""
        try:
            # Extract the hyperparameters from config
            serializable_hp = config.pop('serializable_hp')
            normalisation_fitted = config.pop('normalisation_fitted')
        except KeyError as e:
            raise ValueError(f"Missing required config key: {e}")
        
        # Create a mock hp object with the serializable attributes. Needs items() and __getitem__
        class MockHP:
            def __init__(self, data):
                if not isinstance(data, dict):
                    raise TypeError("HP data must be a dictionary")
                # store in dict form
                self._data = dict(data)
                # Also assign as attributes for legacy attribute access
                for k, v in self._data.items():
                    setattr(self, k, v)

            def items(self):
                return self._data.items()

            def __getitem__(self, key):
                if key not in self._data:
                    raise KeyError(f"HP key '{key}' not found")
                return self._data[key]

        hp = MockHP(serializable_hp)
        
        # Create the model
        model = cls(hp, **config)
        model._normalisation_fitted = normalisation_fitted
        
        # If normalisation was fitted, build the model
        if normalisation_fitted:
            try:
                model.build(input_shape=[(None, 7), (None,)])
                # Verify that normalization layers are actually fitted
                if not model.is_normalisation_fitted:
                    # Reset flag if layers aren't actually fitted
                    model._normalisation_fitted = False
            except Exception as e:
                # Model building failed, but we can still return the model
                # It will build on first call
                model._normalisation_fitted = False
        
        return model


class TrainingModel(tf.keras.Model):
    """Training wrapper for the global model that uses normalised scale for loss computation"""
    
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
    
    def get_config(self):
        """Get configuration for saving"""
        config = super().get_config()
        # Note: TrainingModel is just a wrapper, the GlobalModel holds all the state
        # We don't save the global_model reference as it should be reconstructed
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create training model from configuration"""
        # Note: This method should not be used directly as TrainingModel
        # requires a GlobalModel instance. Instead, create GlobalModel first,
        # then wrap it with TrainingModel.
        raise NotImplementedError(
            "TrainingModel cannot be created from config alone. "
            "Create GlobalModel first, then use TrainingModel(global_model)."
        )
        return cls(global_model, **config)
