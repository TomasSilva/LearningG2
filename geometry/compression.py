# Import libraries
import numpy as np
from math import comb, factorial
from itertools import permutations, combinations


###########################################################################
# Compression functions for rank n antisymmetric form tensors

def form_to_vec(tensor):
    """
    Extract unique components from a form tensor (antisymmetric tensor).
    Works on single tensor or batch, for any rank.
    
    Parameters
    ----------
    tensor : array_like
        Fully antisymmetric form tensor(s)
        - Single: shape (D, D, ..., D) with rank equal dimensions
        - Batch: shape (B, D, D, ..., D) with rank equal dimensions
        
    Returns
    -------
    ndarray, shape (C(D,rank),) or (B, C(D,rank))
        The unique independent components
    """
    tensor = np.asarray(tensor)
    
    # Determine rank and dimension
    if tensor.ndim < 2:
        raise ValueError(f"Tensor must have at least 2 dimensions, got shape {tensor.shape}")
    
    # Check if this is a batch or single tensor
    # Single: all dimensions are equal
    # Batch: all dimensions after the first are equal
    if len(set(tensor.shape)) == 1:
        # Single case: (D, D, ..., D)
        is_single = True
        D = tensor.shape[0]
        rank = tensor.ndim
        tensor = tensor[np.newaxis, ...]
    elif len(set(tensor.shape[1:])) == 1:
        # Batch case: (B, D, D, ..., D)
        is_single = False
        D = tensor.shape[1]
        rank = tensor.ndim - 1
    else:
        raise ValueError(f"Expected all dimensions equal or all but first equal, got shape {tensor.shape}")
    
    # Extract components using unique index combinations
    unique_indices = list(combinations(range(D), rank))
    
    # Build result array by extracting components
    result = np.zeros((tensor.shape[0], len(unique_indices)), dtype=tensor.dtype)
    for i, idx_tuple in enumerate(unique_indices):
        # Use advanced indexing to extract component
        result[:, i] = tensor[(slice(None),) + idx_tuple]
    
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
    Works on single matrix or batch, for any square dimension.
    Returns flattened lower-triangular part (including diagonal).
    
    Parameters
    ----------
    matrix : array_like, shape (n, n) or (B, n, n)
        Symmetric positive-definite matrix/matrices
        
    Returns
    -------
    ndarray, shape (n*(n+1)/2,) or (B, n*(n+1)/2)
        Cholesky lower-triangular components
    """
    matrix = np.asarray(matrix)
    is_single = (matrix.ndim == 2)
    if is_single:
        matrix = matrix[np.newaxis, ...]
    
    # Detect dimension from matrix shape
    n = matrix.shape[1]
    assert matrix.shape[1] == matrix.shape[2], f"Expected square matrices, got shape {matrix.shape}"
    
    chol = np.linalg.cholesky(matrix)
    idx = np.tril_indices(n)
    result = chol[:, idx[0], idx[1]]
    return result[0] if is_single else result

def vec_to_metric(vector):
    """
    Reconstruct symmetric positive-definite matrix from Cholesky lower-triangular components.
    Works on single vector or batch, for any dimension.
    
    Parameters
    ----------
    vector : array_like, shape (n*(n+1)/2,) or (B, n*(n+1)/2)
        Cholesky lower-triangular components
        
    Returns
    -------
    ndarray, shape (n, n) or (B, n, n)
        Reconstructed symmetric positive-definite matrix/matrices
    """
    vector = np.asarray(vector)
    is_single = (vector.ndim == 1)
    if is_single:
        vector = vector[np.newaxis, ...]
    batch_size = vector.shape[0]
    num_components = vector.shape[1]
    
    # Compute dimension n from number of components: n*(n+1)/2 = num_components
    # Solving: n^2 + n - 2*num_components = 0
    n = int((-1 + np.sqrt(1 + 8 * num_components)) / 2)
    assert n * (n + 1) // 2 == num_components, \
        f"Invalid number of components {num_components} for a symmetric matrix"
    
    chol = np.zeros((batch_size, n, n), dtype=vector.dtype)
    idx = np.tril_indices(n)
    chol[:, idx[0], idx[1]] = vector
    # Reconstruct metric: L @ L.T
    output = np.matmul(chol, np.transpose(chol, (0, 2, 1)))
    return output[0] if is_single else output
