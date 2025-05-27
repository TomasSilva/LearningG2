'''Geometric functions'''
# Import libraries
import numpy as np
import tensorflow as tf
from itertools import product, permutations
from geometry.wedge_product import wedge_product

###########################################################################
# Functions related to the CY-structure
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


###########################################################################
# Functions related to the G2-structure
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

