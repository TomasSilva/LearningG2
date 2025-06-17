'''ML architecture for the G2-structure learning'''
# Import libraries
import tensorflow as tf
from math import comb

class GlobalModel(tf.keras.Model):
    def __init__(self, hp, **kwargs):
        super(GlobalModel, self).__init__(**kwargs)
        # Define hyperparameters
        self.hp = hp
        self.serializable_hp = None
        self.set_serializable_hp()
        self.metric = self.hp["metric"]

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
        x = tf.keras.layers.Dense(
            self.hp["n_hidden"], activation=self.hp["activations"], use_bias=self.hp["use_bias"]
        )(combined_input)
        for _ in range(self.hp["n_layers"] - 2):
            x = tf.keras.layers.Dense(
                self.hp["n_hidden"],
                activation=self.hp["activations"],
                use_bias=self.hp["use_bias"]
            )(x)
        outputs = tf.keras.layers.Dense(self.n_out, activation=None, use_bias=False)(x)
           
        self.model = tf.keras.Model(inputs=[coord_input, patch_input], outputs=outputs)

    def call(self, inputs):
        return self.model(inputs)

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


