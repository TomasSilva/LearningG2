'''Normalisation utilities for geometric ML data'''
# Import libraries
import numpy as np
import tensorflow as tf

###########################################################################
# Invertible normalisation class for input/output data
class Normaliser:
    """
    Invertible normaliser for numpy arrays or tf.Tensor objects. Supports fitting, transforming, and inverse transforming.
    Normalisation is performed feature-wise (per column).
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self._tf_mode = False

    def fit(self, data):
        """
        Compute mean and std for normalisation.
        Args:
            data (np.ndarray or tf.Tensor): Array of shape (N, D) to fit statistics on.
        """
        if isinstance(data, tf.Tensor):
            self.mean = tf.reduce_mean(data, axis=0)
            self.std = tf.math.reduce_std(data, axis=0) + 1e-8
            self._tf_mode = True
        else:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0) + 1e-8
            self._tf_mode = False

    def transform(self, data):
        """
        Normalise data using fitted mean and std.
        Args:
            data (np.ndarray or tf.Tensor): Array to normalise.
        Returns:
            Same type as input: Normalised array.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Invert normalisation to recover original scale.
        Args:
            data (np.ndarray or tf.Tensor): Normalised array.
        Returns:
            Same type as input: Array in original scale.
        """
        return data * self.std + self.mean 