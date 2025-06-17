'''Geometric functions'''
# Import libraries
import numpy as np
import tensorflow as tf
from itertools import product, permutations
from geometry.compression import vec_to_form
from geometry.wedge_product import wedge_product

###########################################################################
# Functions related to the CY-structure
def hermitian_to_riemannian_real(H_batch):
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

    return G_batch


def hermitian_to_kahler_real(H_batch):
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

    A = H_batch.imag  # shape: (batch_size, 3, 3)
    zero = np.zeros_like(A)

    # Assemble block antisymmetric matrix:
    # [[ 0,  A ],
    #  [ -A.T, 0 ]]
    upper = np.concatenate((zero, A), axis=2)          # shape: (batch_size, 3, 6)
    lower = np.concatenate((-A.transpose(0, 2, 1), zero), axis=2)  # shape: (batch_size, 3, 6)
    omega = np.concatenate((upper, lower), axis=1)     # shape: (batch_size, 6, 6)

    return omega


def holomorphic_volume_form_to_real(c_batch):
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
    factor = (1 / pow(36, 1 / 9)) * (1 / pow(detB, 1 / 9))
    gG2 = factor * B  
    
    return gG2


###########################################################################
# Functions for computing the exterior derivatives of the trained NN geometric functions
def exterior_derivative(model, coords, patch_idxs, form_output=True):
    """
    Computes the exterior derivative (as a batched Jacobian) of a model's output 
    with respect to its inputs, and optionally repackages it into a fully antisymmetric 
    4-form tensor.

    This function calculates ∂fᵢ/∂xⱼ for each output component of the model with respect 
    to each input component using TensorFlow's batch Jacobian. If `form_output=True`, 
    it interprets the Jacobian as a 3-form valued 1-form and repackages it into a 
    fully antisymmetric tensor of shape (batch_size, 7, 7, 7, 7).

    Args:
        model (tf.keras.Model or callable): A model or function mapping inputs of shape 
            (batch_size, input_dim) to outputs of shape (batch_size, output_dim).
        inputs (tf.Tensor): A 2D tensor of shape (batch_size, input_dim), representing 
            a batch of input vectors.
        form_output (bool): If True, return the exterior derivative as an antisymmetric 
            4-form of shape (batch_size, 7, 7, 7, 7). If False, return the raw Jacobian 
            of shape (batch_size, output_dim, input_dim).

    Returns:
        tf.Tensor:
            - If `form_output=True`: Tensor of shape (batch_size, 7, 7, 7, 7), representing 
              the antisymmetric 4-form structure.
            - If `form_output=False`: Tensor of shape (batch_size, output_dim, input_dim), 
              the raw Jacobian.
    """
    coords = tf.convert_to_tensor(coords)

    with tf.GradientTape() as tape:
        tape.watch(coords)
        outputs = model([coords, patch_idxs])

    jacobians = tape.batch_jacobian(outputs, coords)  # Shape: (batch_size, output_dim, input_dim)
    
    # Repackage the data into the (batch, 7, 7, 7, 7) form
    if form_output:
        components = []
        for i in range(7):
            antisym_tensor = vec_to_form(jacobians[:, :, i], n=7, k=3)  # shape: (batch_size, 7, 7, 7)
            components.append(antisym_tensor)  # 7 of these
            
        # Stack them along a new final axis
        dg2_form = tf.stack(components, axis=-1)
        
        return dg2_form
        
    # Or just return the independent derivatives
    else:
        return jacobians

