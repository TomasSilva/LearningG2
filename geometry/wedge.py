import numpy as np
from itertools import combinations

def wedge(alpha, beta):
    """
    Wedge product of a p-form alpha and a q-form beta on R^n,
    both given as fully antisymmetric tensors.

    alpha: ndarray with shape (n,)*p
    beta:  ndarray with shape (n,)*q
    returns: ndarray with shape (n,)*(p+q)
    """
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)

    p = alpha.ndim
    q = beta.ndim
    if p == 0:
        return beta * alpha  # scalar times form
    if q == 0:
        return alpha * beta

    n = alpha.shape[0]
    assert all(s == n for s in alpha.shape), "alpha must be (n,)*p"
    assert all(s == n for s in beta.shape), "beta must be (n,)*q"
    r = p + q
    if r > n:
        # In dim n, any (p+q)-form with p+q>n is zero
        return np.zeros((n,)*r, dtype=np.result_type(alpha, beta))

    # Outer product: shape (n,)*r; first p axes = alpha, last q axes = beta
    T = np.multiply.outer(alpha, beta)

    out = np.zeros((n,)*r, dtype=np.result_type(alpha, beta))

    # We sum over (p,q)-shuffles: choose positions of the p alpha-indices
    # The rest are beta-indices. This is O(C(r, p)), which is small for r <= 7.
    all_positions = range(r)
    for posA in combinations(all_positions, p):
        posB = [i for i in all_positions if i not in posA]

        # Build axis permutation for np.transpose:
        # perm[j] = old_axis_index that should go to new position j.
        # - old 0..p-1 (alpha axes) go to posA (in order)
        # - old p..p+q-1 (beta axes) go to posB (in order)
        perm = [None]*r
        for k, j in enumerate(posA):
            perm[j] = k        # alpha axis k -> position j
        for k, j in enumerate(posB):
            perm[j] = p + k    # beta axis p+k -> position j

        # Sign of the (p,q)-shuffle:
        # number of swaps needed to move alpha axes from [0..p-1] into posA
        # formula: (-1)^(sum_j (posA[j] - j))
        swaps = sum(posA[j] - j for j in range(p))
        sign = -1 if (swaps % 2) else 1

        out += sign * np.transpose(T, perm)

    return out