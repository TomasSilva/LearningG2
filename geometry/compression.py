'''Geometric functions for mapping between full geometric tensors of interest and vectors of their degree of freedom'''
# Import libraries
import numpy as np
import tensorflow as tf

from math import comb, factorial
from itertools import permutations, combinations

###########################################################################
# Functions for fill_triangular and its inverse (adapted from tensorflow-probability)
def fill_triangular(x, upper=False):
    """Creates a (batch of) triangular matrix from a vector of inputs.

    Created matrix can be lower- or upper-triangular. (It is more efficient to
    create the matrix as upper or lower, rather than transpose.)

    Triangular matrix elements are filled in a clockwise spiral. See example,
    below.

    If `x.shape` is `[b1, b2, ..., bB, d]` then the output shape is
    `[b1, b2, ..., bB, n, n]` where `n` is such that `d = n(n+1)/2`, i.e.,
    `n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.

    Example:

    ```python
    fill_triangular([1, 2, 3, 4, 5, 6])
    # ==> [[4, 0, 0],
    #      [6, 5, 0],
    #      [3, 2, 1]]
    ```


    Args:
    x: `Tensor` representing lower (or upper) triangular elements.
    upper: Python `bool` representing whether output matrix should be upper
        triangular (`True`) or lower triangular (`False`, default).

    Returns:
    tril: `Tensor` with lower (or upper) triangular elements filled from `x`.

    Raises:
    ValueError: if `x` cannot be mapped to a triangular matrix.
    """

    x = tf.convert_to_tensor(x)

    # Get the last dimension size
    if x.shape.rank is not None and x.shape[-1] is not None:
        # Static shape case
        m = x.shape[-1]
        # Formula derived by solving for n: m = n(n+1)/2.
        n = np.sqrt(0.25 + 2. * m) - 0.5
        if n != np.floor(n):
            raise ValueError('Input right-most shape ({}) does not '
                            'correspond to a triangular matrix.'.format(m))
        n = int(n)
    else:
        # Dynamic shape case
        m = tf.shape(x)[-1]
        # For derivation, see above. Casting automatically lops off the 0.5, so we
        # omit it.  We don't validate n is an integer because this has
        # graph-execution cost; an error will be thrown from the reshape, below.
        n = tf.cast(
            tf.sqrt(0.25 + tf.cast(2 * m, dtype=tf.float32)), dtype=tf.int32)

    # Get the rank of x for axis indexing
    ndims = len(x.shape) if x.shape.rank is not None else tf.rank(x)

    if upper:
        x_list = [x, tf.reverse(x[..., n:], axis=[ndims - 1])]
    else:
        x_list = [x[..., n:], tf.reverse(x, axis=[ndims - 1])]

    # Create new shape for reshaping
    batch_shape = tf.shape(x)[:-1]
    new_shape = tf.concat([batch_shape, [n, n]], axis=0)

    x = tf.reshape(tf.concat(x_list, axis=-1), new_shape)
    x = tf.linalg.band_part(
        x, num_lower=(0 if upper else -1), num_upper=(-1 if upper else 0))

    return x


def fill_triangular_inverse(x, upper=False):
    """Creates a vector from a (batch of) triangular matrix.

    The vector is created from the lower-triangular or upper-triangular portion
    depending on the value of the parameter `upper`.

    If `x.shape` is `[b1, b2, ..., bB, n, n]` then the output shape is
    `[b1, b2, ..., bB, d]` where `d = n (n + 1) / 2`.

    Example:
    ```python
    fill_triangular_inverse(
    [[4, 0, 0],
        [6, 5, 0],
        [3, 2, 1]])

    # ==> [1, 2, 3, 4, 5, 6]
    ```

    Args:
    x: `Tensor` representing lower (or upper) triangular elements.
    upper: Python `bool` representing whether output matrix should be upper
        triangular (`True`) or lower triangular (`False`, default).

    Returns:
    flat_tril: (Batch of) vector-shaped `Tensor` representing vectorized lower
        (or upper) triangular elements from `x`.
    """

    x = tf.convert_to_tensor(x)

    # Get the matrix size n
    if x.shape.rank is not None and x.shape[-1] is not None:
        # Static shape case
        n = x.shape[-1]
        m = (n * (n + 1)) // 2
    else:
        # Dynamic shape case
        n = tf.shape(x)[-1]
        m = (n * (n + 1)) // 2

    # Get the rank of x for axis indexing
    ndims = len(x.shape) if x.shape.rank is not None else tf.rank(x)

    if upper:
        initial_elements = x[..., 0, :]
        triangular_portion = x[..., 1:, :]
    else:
        initial_elements = tf.reverse(x[..., -1, :], axis=[ndims - 2])
        triangular_portion = x[..., :-1, :]

    rotated_triangular_portion = tf.reverse(
        tf.reverse(triangular_portion, axis=[ndims - 1]), axis=[ndims - 2])
    consolidated_matrix = triangular_portion + rotated_triangular_portion

    # Create shape for reshaping
    batch_shape = tf.shape(x)[:-2]
    end_sequence_shape = tf.concat([batch_shape, [n * (n - 1)]], axis=0)
    end_sequence = tf.reshape(consolidated_matrix, end_sequence_shape)

    y = tf.concat([initial_elements, end_sequence[..., :m - n]], axis=-1)

    return y


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
    parity = tf.cast(compute_parity(perms_tf), dtype=vectors.dtype)

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
    Uses a pure TensorFlow fill_triangular implementation.
    """
    lower_triangular_matrix = fill_triangular(lower_triangular_vector)
    full_matrix = tf.matmul(
        lower_triangular_matrix, lower_triangular_matrix, transpose_b=True
    )
    return full_matrix


def metric_to_vec(full_matrix):
    """
    Converts a (batch of) full positive-definite matrix to a vectorized lower-triangular Cholesky factor.
    Uses a pure TensorFlow fill_triangular_inverse implementation.
    """
    lower_triangular_matrices = tf.linalg.cholesky(full_matrix)
    lower_triangular_vector = fill_triangular_inverse(
        lower_triangular_matrices
    )
    return lower_triangular_vector

