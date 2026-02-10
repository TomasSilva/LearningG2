# Import libraries
import numpy as np
from math import comb, factorial
from itertools import permutations, combinations


###########################################################################
# Compression functions for rank n antisymmetric form tensors

def form_to_vec(tensor):
    """
    Extract unique components from a form tensor (antisymmetric tensor).
    Works on single tensor or batch.
    
    Parameters
    ----------
    tensor : array_like, shape (D, D, ..., D) with rank dimensions or (B, D, D, ..., D)
        Fully antisymmetric form tensor(s)
        
    Returns
    -------
    ndarray, shape (C(D,rank),) or (B, C(D,rank))
        The unique independent components
    """
    tensor = np.asarray(tensor)
    
    # Determine if batch or single
    if tensor.ndim == 3:
        # Single 3-form (7,7,7) case
        is_single = True
        D, rank = 7, 3
        tensor = tensor[np.newaxis, ...]
    elif tensor.ndim == 4 and tensor.shape[1] == tensor.shape[2] == tensor.shape[3]:
        # Batch 3-form (B, 7,7,7) case
        is_single = False
        D, rank = tensor.shape[1], 3
    else:
        raise ValueError(f"Expected shape (7,7,7) or (B,7,7,7), got {tensor.shape}")
    
    # Extract components using unique index combinations
    unique_indices = list(combinations(range(D), rank))
    result = np.array([[t[i, j, k] for (i, j, k) in unique_indices] for t in tensor])
    
    return result[0] if is_single else result


def compute_parity_np(perm):
    """
    Compute the parity (+1 or -1) of a permutation.
    
    Parameters
    ----------
    perm : array_like, shape (k,)
        A permutation of integers
        
    Returns
    -------
    int
        +1 if even permutation, -1 if odd permutation
    """
    perm = np.asarray(perm)
    k = len(perm)
    inversions = 0
    for i in range(k):
        for j in range(i + 1, k):
            if perm[i] > perm[j]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def vec_to_form(vectors, n=7, k=3):
    """
    Reconstruct full antisymmetric tensor from unique components.
    Works on single vector or batch.
    
    Parameters
    ----------
    vectors : array_like, shape (C(n,k),) or (B, C(n,k))
        Vector(s) of unique components
    n : int
        Dimension of each axis (default: 7)
    k : int
        Rank of the tensor (default: 3)
    
    Returns
    -------
    ndarray, shape (n,n,...,n) (k times) or (B, n,n,...,n)
        Reconstructed antisymmetric tensor(s)
    """
    vectors = np.asarray(vectors)
    is_single = (vectors.ndim == 1)
    
    if is_single:
        vectors = vectors[np.newaxis, ...]
    
    batch_size = vectors.shape[0]
    expected_length = comb(n, k)
    
    if vectors.shape[1] != expected_length:
        raise ValueError(f"Expected vector length {expected_length}, got {vectors.shape[1]}")
    
    # Get unique index combinations
    unique_indices = list(combinations(range(n), k))
    
    # Create output tensor
    output_shape = (batch_size,) + (n,) * k
    output = np.zeros(output_shape, dtype=vectors.dtype)
    
    # Fill tensor with antisymmetric property
    for vec_idx, base_indices in enumerate(unique_indices):
        base_value = vectors[:, vec_idx]
        
        # Generate all permutations and their parities
        for perm in permutations(base_indices):
            parity = compute_parity_np(perm)
            output[(slice(None),) + perm] = parity * base_value
    
    return output[0] if is_single else output


###########################################################################
# Compression functions for rank 2 symmetric metrics

def metric_to_vec(matrix):
    """
    Compress a positive-definite symmetric matrix to its Cholesky lower-triangular components.
    Works on single matrix or batch.
    Returns flattened lower-triangular part (including diagonal).
    """
    matrix = np.asarray(matrix)
    is_single = (matrix.ndim == 2)
    if is_single:
        matrix = matrix[np.newaxis, ...]
    assert matrix.shape[1:] == (7, 7), f"Expected shape (B, 7, 7), got {matrix.shape}"
    chol = np.linalg.cholesky(matrix)
    idx = np.tril_indices(7)
    result = chol[:, idx[0], idx[1]]
    return result[0] if is_single else result

def vec_to_metric(vector):
    """
    Reconstruct symmetric positive-definite matrix from Cholesky lower-triangular components.
    Works on single vector or batch.
    """
    vector = np.asarray(vector)
    is_single = (vector.ndim == 1)
    if is_single:
        vector = vector[np.newaxis, ...]
    batch_size = vector.shape[0]
    assert vector.shape[1] == 28, f"Expected 28 components, got {vector.shape[1]}"
    chol = np.zeros((batch_size, 7, 7), dtype=vector.dtype)
    idx = np.tril_indices(7)
    chol[:, idx[0], idx[1]] = vector
    # Reconstruct metric: L @ L.T
    output = np.matmul(chol, np.transpose(chol, (0, 2, 1)))
    return output[0] if is_single else output
