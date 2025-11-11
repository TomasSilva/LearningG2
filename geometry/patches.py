'''Geometric functions for changing coordinate systems'''
# Import libraries
import numpy as np
import tensorflow as tf

###########################################################################
# Coordinate change functions
def CoordChange_C5R10(points, inverse=False):
    """
    Vectorized coordinate change between complex and real representations. 
    Real coordinates arranged: (x1, x2, ..., y1, y2, ...).
    
    Args:
        points (np.ndarray): 
            - If `inverse=False`: array of shape (batch_size, n_complex), dtype complex.
            - If `inverse=True`: array of shape (batch_size, 2 * n_complex), dtype float.
        inverse (bool): Direction of transformation.

    Returns:
        np.ndarray:
            - If `inverse=False`: shape (batch_size, 2 * n_complex), dtype float.
            - If `inverse=True`: shape (batch_size, n_complex), dtype complex.
    """
    if not inverse:
        #return np.stack((point.real, point.imag), axis=-1).reshape(points.shape[0], -1) #...form of (x1, y1, x2, y2, ...)
        return np.concatenate((np.real(points), np.imag(points)), axis=1)
    else:
        #return point[:,::2] + 1j * point[:,1::2] #...form of (x1, y1, x2, y2, ...)
        return points[:, :points.shape[1] // 2] + 1j * points[:, points.shape[1] // 2:]
    

###########################################################################
# Function to combine the patch indices into a unique scalar for model input
# DEPRECATED: Now using separate embeddings for one_idx and dropped_idx
# def patch_indices_to_scalar(i, j):
#     """
#     Maps (i, j) indices with i ≠ j, 0 <= i,j <= 4, to a unique ID in [0, 19].
#     """
#     i = tf.convert_to_tensor(i)
#     j = tf.convert_to_tensor(j)

#     # Assert i != j
#     assert_op = tf.debugging.assert_none_equal(i, j, message="Patch indices must satisfy i ≠ j")

#     with tf.control_dependencies([assert_op]):
#         offset = tf.cast(j > i, j.dtype)
#         j_pos = j - offset
#         patch_id = i * 4 + j_pos

#     return patch_id

