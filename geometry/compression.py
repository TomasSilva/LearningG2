'''Geometric functions for mapping between full geometric tensors of interest and vectors of their degree of freedom'''
# Import libraries
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from math import comb, factorial
from itertools import permutations, combinations

###########################################################################
# Compression functions for rank n antisymmetric form tensors
def form_to_vec(tensor):
    """
    Extracts unique components of a batch of forms represented as antisymmetric tensors.

    Args:
        tensor: tf.Tensor of shape (B, D, D, ..., D), where B=batch size and
                rank = number of antisymmetric tensor dimensions.

    Returns:
        tf.Tensor of shape (B, N) where N = number of unique components,
        containing the extracted unique antisymmetric components for each batch.
    """
    batch_size = tf.shape(tensor)[0]
    rank = tensor.shape.rank - 1
    D = tensor.shape[1]
    
    # Generate unique index combinations for antisymmetric components 
    unique_indices = tf.constant(
        list(combinations(range(D), rank))
    )  # shape (N, rank)

    # Repeat batch indices for each unique index
    batch_indices = tf.repeat(tf.range(batch_size), tf.shape(unique_indices)[0])  # shape (B*N,)

    # Tile unique indices for each batch item
    tiled_unique_indices = tf.tile(unique_indices, [batch_size, 1])  # shape (B*N, rank)

    # Concatenate batch indices and unique indices to form full indices
    full_indices = tf.concat([tf.expand_dims(batch_indices, axis=1), tiled_unique_indices], axis=1)  # shape (B*N, rank+1)

    # Gather the selected elements
    gathered = tf.gather_nd(tensor, full_indices)  # shape (B*N,)

    # Reshape to (B, N)
    result = tf.reshape(gathered, (batch_size, tf.shape(unique_indices)[0]))

    return result


def compute_parity(p):
    """
    Compute parity (+1 or -1) of each permutation in batch p.
    p: tf.Tensor of shape (batch_size, k), dtype int32 or int64
    Returns:
        tf.Tensor of shape (batch_size,), dtype float32 with +1 or -1 values
    """
    k = tf.shape(p)[1]

    # Expand dims to compare all pairs i,j
    p1 = tf.expand_dims(p, 2)  # (batch_size, k, 1)
    p2 = tf.expand_dims(p, 1)  # (batch_size, 1, k)

    # Create mask for upper triangle (i < j)
    mask = tf.linalg.band_part(tf.ones((k, k), dtype=tf.bool), 0, -1)  # upper triangle incl diagonal
    mask = tf.linalg.set_diag(mask, tf.zeros(k, dtype=tf.bool))       # zero diagonal
    # Now mask[i,j] == True iff i < j

    # Compare pairs where i < j and count inversions: p1 > p2
    inversions = tf.reduce_sum(
        tf.cast(tf.logical_and(p1 > p2, mask), tf.int32),
        axis=[1, 2]
    )

    parity = tf.where(inversions % 2 == 0, 1.0, -1.0)
    return parity


def vec_to_form(vectors, n, k):
    """
    Reconstruct a full antisymmetric tensor of shape 
    (batch_size, n, n, ..., n) (k times) from a batch of 1D vectors of their
    respective unique entries.
    
    Args:
        vector (np.ndarray): (batch_size, C(n,k)).
        n (int): dimension of each axis
        k (int): rank of the tensor (number of axes)
    
    Returns:
        np.ndarray: reconstructed antisymmetric tensor of shape (batch_size, n, n, ..., n) (k times).
    """
    batch_size = tf.shape(vectors)[0]
    expected_length = comb(n, k)
    
    # Check length dimension matches
    if vectors.shape[1] != expected_length:
        raise ValueError(f"Expected vector length {expected_length}, got {vectors.shape[1]}")

    # 1. Get all strictly increasing indices (C(n,k), k)
    unique_indices = np.array(list(combinations(range(n), k)))  # shape (C(n,k), k)

    # 2. Precompute all permutations of unique indices and parity signs
    perms_list = []
    base_indices_repeated = []
    for base_idx in unique_indices:
        perms_list.extend(permutations(base_idx))
        base_indices_repeated.extend([base_idx] * factorial(k))

    perms_np = np.array(perms_list, dtype=np.int32)  # (C(n,k)*k!, k)
    base_indices_np = np.array(base_indices_repeated, dtype=np.int32)  # same shape
    perms_tf = tf.constant(perms_np)

    # 3. Compute parity for all permutations
    parity = compute_parity(perms_tf)  # (C(n,k)*k!,)

    # 4. For each vector in batch, gather values for base indices, then multiply by parity for each perm
    # Map from base index to vector index:
    # create a lookup dict to map tuple(base_idx) to vector idx
    base_idx_to_vec_idx = {tuple(idx): i for i, idx in enumerate(unique_indices)}

    # Convert base_indices_tf to vector indices:
    base_idx_tuples = [tuple(idx) for idx in base_indices_np]  # just reuse from np array

    # Build vector indices array for base_indices_tf
    vec_indices_for_perms = tf.constant([base_idx_to_vec_idx[t] for t in base_idx_tuples], dtype=tf.int32)  # (C(n,k)*k!,)

    # Expand batch dimension:
    vec_indices_for_perms_batch = tf.tile(tf.expand_dims(vec_indices_for_perms, 0), [batch_size, 1])  # (batch_size, C(n,k)*k!)

    # Gather base values from vectors:
    base_values = tf.gather(vectors, vec_indices_for_perms_batch, batch_dims=1)  # (batch_size, C(n,k)*k!)

    # Multiply by parity:
    signed_values = base_values * parity  # broadcast parity over batch_size

    # 5. Scatter signed_values to final tensor shape:
    output_shape = tf.concat([[batch_size], [n]*k], axis=0)  # (batch_size, n, n, ..., n)

    # Flatten indices for scatter_nd:
    # perms_tf shape (C(n,k)*k!, k)
    # For batch indexing, prepend batch dimension indices:
    batch_indices = tf.repeat(tf.range(batch_size), repeats=tf.shape(perms_tf)[0])  # (batch_size * C(n,k)*k!)
    perms_tiled = tf.tile(perms_tf, [batch_size, 1])  # same shape

    scatter_indices = tf.stack([batch_indices, perms_tiled[:, 0], perms_tiled[:, 1], perms_tiled[:, 2]], axis=1) if k == 3 else \
                      tf.concat([tf.expand_dims(batch_indices, 1), perms_tiled], axis=1)
    # For general k, use the concat version above

    # Scatter the signed values:
    signed_values_flat = tf.reshape(signed_values, [-1])
    output_tensor = tf.scatter_nd(scatter_indices, signed_values_flat, output_shape)

    return output_tensor


###########################################################################
# Compression functions for the rank 2 symmetric metrics
def vec_to_metric(lower_triangular_vector):
    """
    Reconstructs a (batch of) full positive-definite matrix from its vectorized lower-triangular Cholesky factor.

    This function assumes the input vector represents a lower-triangular matrix in 
    compact (vectorized) form, such as one created by `tfp.math.fill_triangular_inverse`. 
    It then reconstructs the full matrix as L @ Lᵀ.

    Args:
        lower_triangular_vector (tf.Tensor): A 1D tensor containing the lower-triangular 
            elements of a matrix in a compact vectorized form.

    Returns:
        tf.Tensor: A full positive-definite matrix reconstructed from the Cholesky factor.
    """
    lower_triangular_matrix = tfp.math.fill_triangular(lower_triangular_vector)
    full_matrix = tf.matmul(
        lower_triangular_matrix, lower_triangular_matrix, transpose_b=True
    )

    return full_matrix


def metric_to_vec(full_matrix):
    """
    Converts a (batch of) full positive-definite matrix to a vectorized lower-triangular Cholesky factor.
    
    This function performs a Cholesky decomposition on the input matrix and then 
    returns the lower-triangular factor in vectorized (compact) form.
    
    Args:
        full_matrix (tf.Tensor): A symmetric positive-definite matrix (or batch of matrices).
    
    Returns:
        tf.Tensor: A 1D tensor containing the lower-triangular elements of the Cholesky 
        factor in vectorized form.
    """
    lower_triangular_matrices = tf.linalg.cholesky(full_matrix)
    lower_triangular_vector = tfp.math.fill_triangular_inverse(
        lower_triangular_matrices
    )

    return lower_triangular_vector

