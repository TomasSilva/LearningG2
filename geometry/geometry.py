'''Geometric functions'''
# Import libraries
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from math import comb, pow, factorial
from itertools import product, permutations, combinations
from geometry.wedge_product import wedge_product

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
# Geometric functions to build the G2 3-form
def hermitian_to_real_symmetric(H_batch):
    """
    Vectorized conversion of a batch of 3x3 Hermitian matrices to 6x6 real symmetric matrices.
    Real coordinates arranged: (x1, x2, ..., y1, y2, ...).
    
    Args:
        H_batch (np.ndarray): Array of shape (batch_size, 3, 3) with complex Hermitian matrices.

    Returns:
        G_batch (np.ndarray): Array of shape (batch_size, 6, 6) with real symmetric matrices.
    """
    if H_batch.shape[-2:] != (3, 3):
        raise ValueError("Each matrix must be 3x3.")
    
    if not np.allclose(H_batch, H_batch.conj().transpose(0, 2, 1)):
        raise ValueError("All input matrices must be Hermitian.")

    A = H_batch.real  # shape: (batch_size, 3, 3)
    B = H_batch.imag  # shape: (batch_size, 3, 3)

    upper = np.concatenate((A, -B), axis=2)  # shape: (batch_size, 3, 6)
    lower = np.concatenate((B,  A), axis=2)  # shape: (batch_size, 3, 6)
    G_batch = np.concatenate((upper, lower), axis=1)  # shape: (batch_size, 6, 6)

    # Reorder axes: (x1, y1, x2, y2, x3, y3) → (x1, x2, x3, y1, y2, y3)
    perm = [0, 2, 4, 1, 3, 5]
    G_batch = G_batch[:, perm][:, :, perm]

    return G_batch


def kahler_form_real(H_batch):
    """
    Vectorized Kähler form computation for a batch of 3x3 Hermitian matrices.
    Real coordinates arranged: (x1, x2, ..., y1, y2, ...).
    
    Args:
        H_batch (np.ndarray): Array of shape (batch_size, 3, 3), complex Hermitian matrices.

    Returns:
        omega_batch (np.ndarray): Array of shape (batch_size, 6, 6), real antisymmetric matrices.
    """
    if H_batch.shape[-2:] != (3, 3):
        raise ValueError("Each input matrix must be 3x3.")

    if not np.allclose(H_batch, H_batch.conj().transpose(0, 2, 1)):
        raise ValueError("All matrices must be Hermitian.")

    A = H_batch.real  # shape: (batch_size, 3, 3)
    zero = np.zeros_like(A)

    # Assemble block antisymmetric matrix:
    # [[ 0,  A ],
    #  [ -A.T, 0 ]]
    upper = np.concatenate((zero, A), axis=2)          # shape: (batch_size, 3, 6)
    lower = np.concatenate((-A.transpose(0, 2, 1), zero), axis=2)  # shape: (batch_size, 3, 6)
    omega = np.concatenate((upper, lower), axis=1)     # shape: (batch_size, 6, 6)

    # Reorder (x1, y1, x2, y2, x3, y3) → (x1, x2, x3, y1, y2, y3)
    perm = [0, 2, 4, 1, 3, 5]
    omega_reordered = omega[:, perm][:, :, perm]

    return omega_reordered


def holomorphic_volume_form_to_real_tensor(c_batch):
    """
    Convert a complex coefficient c of dz^1 ^ dz^2 ^ dz^3 into a real 6x6x6 tensor 
    representing the real coordinate expression of the holomorphic volume form.

    Args:
        c_batch (np.ndarray): shape (batch,), complex dtype.

    Returns:
        Omega_real (np.ndarray): shape (batch, 6, 6, 6), real part of 3-form
        Omega_imag (np.ndarray): shape (batch, 6, 6, 6), imaginary part of 3-form
    """
    batch_size = c_batch.shape[0]
    Omega_real = np.zeros((batch_size, 6, 6, 6))
    Omega_imag = np.zeros((batch_size, 6, 6, 6))

    dx = [0, 1, 2]
    dy = [3, 4, 5]
    index_triplets = list(permutations(dx, 3))

    # Only one unique permutation set (since all permutations are included later)
    for (i, j, k) in index_triplets:
        iy, jy, ky = dy[i], dy[j], dy[k]
        for a, b, d in product([0, 1], repeat=3):
            idx = [i, j, k]
            idx[0] = [i, iy][a]
            idx[1] = [j, jy][b]
            idx[2] = [k, ky][d]

            coeff = ((1j) ** (a + b + d)) * c_batch[:, None] / 6  # shape: (batch, 1)
            for perm in set(permutations(idx)):
                # Determine the sign of the permutation
                sign = (
                    1 if list(perm) == sorted(perm) else
                    (-1) ** sum(p1 > p2 for p1, p2 in zip(perm, sorted(perm)))
                )
                Omega_real[:, perm[0], perm[1], perm[2]] += coeff.real[:, 0] * sign
                Omega_imag[:, perm[0], perm[1], perm[2]] += coeff.imag[:, 0] * sign

    return Omega_real, Omega_imag


def compute_gG2(G2_val):
    # TODO: this function must be vectorized!
    # Note that wedge_product is very slow
    """
    Compute the gG2 metric from the G2 structure 3-form.
    See: https://arxiv.org/pdf/math/0702077 EQ (2.3)
    """
    B = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            if i <= j:  # avoid double counting by only computing upper triangle       
                B[i,j] = wedge_product(G2_val[i,:,:], wedge_product(G2_val[j,:,:], G2_val))[0,1,2,3,4,5,6]
    # Make B symmetric
    B = B + B.T - np.diag(B.diagonal())             
    detB = np.linalg.det(B)

    
    ### 
    # Warning if excessive clipping
    if detB < -1e-6:
        raise ValueError("detB ({detB}) negative...")
    # Clip det(B) if negative (due to floating pt error)
    detB = max(detB, 0)
    
    #...below gives division by 0, basically shouldn't be getting singular B !?
    #factor = (1 / pow(36, 1 / 9)) * (1 / pow(detB, 1 / 9))
    #gG2 = factor * B
    
    gG2 = B ###delete when above fixed
    ###
    
    
    return gG2


###########################################################################
# Patch transformation functions
def PatchChange_Coords(coords, input_patch=0, output_patch=0):
    if input_patch == output_patch:
        return coords
    else:
        ### write this (patches labelled 0-4 with 0 the first patch used as input)
        return coords ###
    
def PatchChange_G2form(coords, forms, input_patch=0, output_patch=0):
    if input_patch == output_patch:
        return forms
    else:
        ### write this (patches labelled 0-4 with 0 the first patch used as input)
        return forms
    
def PatchChange_G2metric(coords, metrics, input_patch=0, output_patch=0):
    if input_patch == output_patch:
        return metrics
    else:
        ### write this (patches labelled 0-4 with 0 the first patch used as input)
        return metrics


###########################################################################
# Function to reduce forms tensors to their dof vectors 
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


###########################################################################
### OLD FUNCTIONS ###
def wedge_form2_with_form1(omega6_batch, alpha7_batch):
    """
    Vectorized wedge product of a batch of 6x6 antisymmetric 2-forms with a batch of 7D 1-forms.
    
    Args:
        omega6_batch (np.ndarray): shape (batch_size, 6, 6), antisymmetric 2-forms.
        alpha7_batch (np.ndarray): shape (batch_size, 7), 1-forms.

    Returns:
        form3_batch (np.ndarray): shape (batch_size, 7, 7, 7), antisymmetric 3-forms.
    """
    batch_size = omega6_batch.shape[0]
    
    assert omega6_batch.shape == (batch_size, 6, 6), "omega6 must be (N, 6, 6)"
    assert alpha7_batch.shape == (batch_size, 7), "alpha7 must be (N, 7)"

    # Check antisymmetry
    if not np.allclose(omega6_batch, -omega6_batch.transpose(0, 2, 1)):
        raise ValueError("omega6 must be antisymmetric")

    # Extend omega6 to (N, 7, 7)
    omega7_batch = np.zeros((batch_size, 7, 7), dtype=omega6_batch.dtype)
    omega7_batch[:, :6, :6] = omega6_batch

    # Broadcast and compute the 3-form wedge product
    i, j, k = np.meshgrid(np.arange(7), np.arange(7), np.arange(7), indexing='ij')

    # Compute each term: omega[i,j]*alpha[k], omega[j,k]*alpha[i], omega[k,i]*alpha[j]
    term1 = omega7_batch[:, i, j] * alpha7_batch[:, k]
    term2 = omega7_batch[:, j, k] * alpha7_batch[:, i]
    term3 = omega7_batch[:, k, i] * alpha7_batch[:, j]

    form3_batch = (1.0 / 3.0) * (term1 + term2 + term3)

    return form3_batch


