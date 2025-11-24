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
    
    def __init__(self, name=None, mean=None, std=None, dtype="float32", _fitted=False, **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Handle restoration from saved config
        if mean is not None and std is not None:
            # Use original dtype or specified dtype
            try:
                restore_dtype = getattr(tf, dtype) if isinstance(dtype, str) else dtype
            except AttributeError:
                restore_dtype = tf.float32  # Fallback
                
            base_name = name if name else "normalisation"
            self.mean = tf.Variable(
                tf.cast(mean, restore_dtype), 
                trainable=False,
                name=f"{base_name}_mean"
            )
            self.std = tf.Variable(
                tf.cast(std, restore_dtype), 
                trainable=False,
                name=f"{base_name}_std"
            )
            self._fitted = _fitted
        else:
            self.mean = None
            self.std = None
            self._fitted = False
            
    def build(self, input_shape):
        super().build(input_shape)
        if self.mean is None:
            base_name = self.name if self.name else "normalisation"
            input_dim = input_shape[-1] if len(input_shape) > 1 else 1
            self.mean = self.add_weight(
                name=f"{base_name}_mean",
                shape=(input_dim,),
                initializer='zeros',
                trainable=False
            )
            self.std = self.add_weight(
                name=f"{base_name}_std", 
                shape=(input_dim,),
                initializer='ones',
                trainable=False
            )
            self._fitted = False
        else:
            self._fitted = True
        
    def fit_statistics(self, data):
        """Fit normalisation statistics on data"""
        # Validation
        if data is None or tf.size(data) == 0:
            raise ValueError("Cannot fit statistics on empty data")
        if tf.reduce_any(tf.math.is_nan(data)):
            raise ValueError("Data contains NaN values")
            
        # Use the same dtype as the input data
        dtype = data.dtype
        
        mean_value = tf.cast(tf.reduce_mean(data, axis=0), dtype)
        std_value = tf.cast(tf.math.reduce_std(data, axis=0) + 1e-8, dtype)
        
        # If weights don't exist yet, create them
        if self.mean is None or self.std is None:
            base_name = self.name if self.name else "normalisation"
            self.mean = self.add_weight(
                name=f"{base_name}_mean",
                shape=mean_value.shape,
                initializer='zeros',
                trainable=False,
                dtype=dtype
            )
            self.std = self.add_weight(
                name=f"{base_name}_std", 
                shape=std_value.shape,
                initializer='ones',
                trainable=False,
                dtype=dtype
            )
        
        # Set the values
        self.mean.assign(mean_value)
        self.std.assign(std_value)
        self._fitted = True
        
    def call(self, inputs):
        # Check if we need to restore from weights
        if self.mean is None or self.std is None:
            for weight in self.weights:
                if 'mean' in weight.name:
                    self.mean = weight
                elif 'std' in weight.name:
                    self.std = weight
                    
        # If statistics are not available, return inputs unchanged
        if self.mean is None or self.std is None:
            return inputs
            
        mean = tf.cast(self.mean, inputs.dtype)
        std = tf.cast(self.std, inputs.dtype)
        
        # Check if these are dummy/default values during loading - use tf.where for graph compatibility
        is_default = tf.logical_and(
            tf.reduce_all(tf.equal(mean, 0.0)),
            tf.reduce_all(tf.equal(std, 1.0))
        )
        
        normalized = (inputs - mean) / std
        return tf.where(is_default, inputs, normalized)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'fitted': getattr(self, '_fitted', False)
        })
        return config


class DenormalisationLayer(tf.keras.layers.Layer):
    """Layer that reverses z-score normalisation using fitted statistics"""
    
    def __init__(self, name=None, mean=None, std=None, dtype="float32", **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Handle restoration from saved config
        if mean is not None and std is not None:
            try:
                restore_dtype = getattr(tf, dtype) if isinstance(dtype, str) else dtype
            except AttributeError:
                restore_dtype = tf.float32
                
            base_name = name if name else "denormalisation"
            self.mean = tf.Variable(
                tf.cast(mean, restore_dtype), 
                trainable=False,
                name=f"{base_name}_mean"
            )
            self.std = tf.Variable(
                tf.cast(std, restore_dtype), 
                trainable=False,
                name=f"{base_name}_std"
            )
        else:
            self.mean = None
            self.std = None
            
    def build(self, input_shape):
        super().build(input_shape)
        if self.mean is None:
            base_name = self.name if self.name else "denormalisation"
            input_dim = input_shape[-1] if len(input_shape) > 1 else 1
            self.mean = self.add_weight(
                name=f"{base_name}_mean",
                shape=(input_dim,),
                initializer='zeros',
                trainable=False
            )
            self.std = self.add_weight(
                name=f"{base_name}_std", 
                shape=(input_dim,),
                initializer='ones',
                trainable=False
            )
            self._fitted = False
        else:
            self._fitted = True
        
    def fit_statistics(self, data):
        """Fit denormalisation statistics on data"""
        # Validation
        if data is None or tf.size(data) == 0:
            raise ValueError("Cannot fit statistics on empty data")
        if tf.reduce_any(tf.math.is_nan(data)):
            raise ValueError("Data contains NaN values")
            
        # Use the same dtype as the input data
        dtype = data.dtype
        
        mean_value = tf.cast(tf.reduce_mean(data, axis=0), dtype)
        std_value = tf.cast(tf.math.reduce_std(data, axis=0) + 1e-8, dtype)
        
        # If weights don't exist yet, create them
        if self.mean is None or self.std is None:
            base_name = self.name if self.name else "denormalisation"
            self.mean = self.add_weight(
                name=f"{base_name}_mean",
                shape=mean_value.shape,
                initializer='zeros',
                trainable=False,
                dtype=dtype
            )
            self.std = self.add_weight(
                name=f"{base_name}_std", 
                shape=std_value.shape,
                initializer='ones',
                trainable=False,
                dtype=dtype
            )
        
        # Set the values
        self.mean.assign(mean_value)
        self.std.assign(std_value)
        self._fitted = True
        
    def call(self, inputs):
        # Check if we need to restore from weights
        if self.mean is None or self.std is None:
            for weight in self.weights:
                if 'mean' in weight.name:
                    self.mean = weight
                elif 'std' in weight.name:
                    self.std = weight
            
        if self.mean is None or self.std is None:
            return inputs
            
        mean = tf.cast(self.mean, inputs.dtype)
        std = tf.cast(self.std, inputs.dtype)
        
        # Check if these are dummy/default values during loading - use tf.where for graph compatibility
        is_default = tf.logical_and(
            tf.reduce_all(tf.equal(mean, 0.0)),
            tf.reduce_all(tf.equal(std, 1.0))
        )
        
        denormalized = inputs * std + mean
        return tf.where(is_default, inputs, denormalized)
        
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
            
        # Define embedding parameters for 2D patch vector [one_idx, dropped_idx]
        # Each coordinate can be 0-4, representing which coord is set to 1 or dropped
        self.embedding_dim = self.hp["embedding_dim"]
        
        # Define architecture
        coord_input = tf.keras.layers.Input(shape=(7,), name="coord_input")
        patch_indices_input = tf.keras.layers.Input(shape=(2,), dtype=tf.int32, name="patch_indices_input")

        # Handle patch encoding based on embedding_dim
        if self.embedding_dim is None:
            # Use one-hot encoding: 2 indices × 5 classes = 10-dimensional vector
            one_idx_onehot = tf.keras.layers.Lambda(
                lambda x: tf.one_hot(x[:, 0], depth=5),
                output_shape=(5,)
            )(patch_indices_input)
            dropped_idx_onehot = tf.keras.layers.Lambda(
                lambda x: tf.one_hot(x[:, 1], depth=5),
                output_shape=(5,)
            )(patch_indices_input)
            patch_embed = tf.keras.layers.Concatenate()([one_idx_onehot, dropped_idx_onehot])
        else:
            # Use dense embedding layer
            self.patch_embedding = tf.keras.layers.Dense(
                self.embedding_dim,
                activation=None,
                use_bias=True,
                name="patch_embedding"
            )
            patch_indices_float = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(patch_indices_input)
            patch_embed = self.patch_embedding(patch_indices_float)
        
        # Concatenate coordinates with patch representation
        combined_input = tf.keras.layers.Concatenate()([coord_input, patch_embed])

        # Feedforward layers
        initializer = ScaledGlorotUniform(scale=self.hp["parameter_initialisation_scale"])
        
        # Get regularization parameters (default to 0 for backward compatibility)
        # Ensure they are float type (YAML may parse scientific notation as string)
        dropout_rate = float(self.hp.get("dropout_rate", 0.0))
        l2_reg = float(self.hp.get("l2_regularization", 0.0))
        regularizer = tf.keras.regularizers.L2(l2_reg) if l2_reg > 0 else None
        
        x = tf.keras.layers.Dense(
            self.hp["n_hidden"], 
            activation=self.hp["activations"], 
            use_bias=self.hp["use_bias"], 
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )(combined_input)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        for _ in range(self.hp["n_layers"] - 2):
            x = tf.keras.layers.Dense(
                self.hp["n_hidden"],
                activation=self.hp["activations"],
                use_bias=self.hp["use_bias"],
                kernel_initializer=initializer,
                kernel_regularizer=regularizer
            )(x)
            if dropout_rate > 0:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
            
        outputs = tf.keras.layers.Dense(
            self.n_out, 
            activation=None, 
            use_bias=False, 
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )(x)
           
        self.model = tf.keras.Model(
            inputs=[coord_input, patch_indices_input], 
            outputs=outputs
        )

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
            
            def get(self, key, default=None):
                return self._data.get(key, default)
        
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
        # During loading, check if the layers exist first
        if not hasattr(self, 'input_normaliser') or not hasattr(self, 'output_denormaliser'):
            return getattr(self, '_normalisation_fitted', False)
            
        input_fitted = (self.input_normaliser.mean is not None and 
                       self.input_normaliser.std is not None)
        output_fitted = (self.output_denormaliser.mean is not None and 
                        self.output_denormaliser.std is not None)
        
        fitted = self._normalisation_fitted and input_fitted and output_fitted
        
        # During loading, if we have a saved fitted state but layers aren't ready yet, trust the saved state
        if self._normalisation_fitted and not (input_fitted and output_fitted):
            return self._normalisation_fitted
        
        return fitted

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
            
            def get(self, key, default=None):
                return self._data.get(key, default)

        hp = MockHP(serializable_hp)
        
        # Create the model
        model = cls(hp, **config)
        model._normalisation_fitted = normalisation_fitted
        
        # For loaded models, we need to ensure the normalization layers
        # have their statistics properly restored before any build/call operations
        # This happens automatically through the layer's from_config methods
        
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
