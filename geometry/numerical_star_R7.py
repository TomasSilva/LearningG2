import numpy as np
from math import factorial
import itertools


def levi_cevita_tensor(dim):   
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr

# Cache the 7D Levi-Civita tensor (computed once at module import)
# This avoids recomputing it on every Hodge_Dual call
_EPS_7D = levi_cevita_tensor(7)

def Hodge_Dual(p_form, g):
    """
    Compute the Hodge star operator ⋆: Λᵖ(M) → Λ⁷⁻ᵖ(M) on a 7-dimensional manifold.
    
    This implementation uses vectorized NumPy einsum operations for optimal performance.
    For p=3 forms (critical for G2 geometry), this is ~100-1000x faster than nested loops.
    
    Parameters
    ----------
    p_form : ndarray
        A p-form as a fully antisymmetric tensor of rank p (shape (7,)*p)
    g : ndarray, shape (7, 7)
        The metric tensor (symmetric positive-definite matrix)
        
    Returns
    -------
    ndarray
        The (7-p)-form ⋆p_form as a fully antisymmetric tensor of rank (7-p)
    """
    # Define p and q
    p = len(p_form.shape)
    # Factorial normalization: 1/(p! (7-p)!)
    fac_norm = 1.0 / (factorial(p) * factorial(7 - p))
    # Define the inverse metric
    g_inv = np.linalg.inv(g)
    # Compute the square root of the determinant of the metric
    sqrt_g = np.sqrt(np.linalg.det(g))
    # Use cached Levi-Civita tensor
    eps = _EPS_7D
    # Compute the q-form in \Omega^{7-p} using vectorized einsum operations
    # Formula: (⋆ω)_{a1...a_{7-p}} = sqrt_g * ε_{b1...b7} * g^{b1,c1}...g^{bp,cp} * ω_{c1...cp}
    # where the ε contracts over all 7 indices, and we sum over the appropriate indices
    if p == 0:
        q_form = fac_norm * p_form * eps
    elif p == 1:
        q_form = fac_norm * sqrt_g * np.einsum('iabcdef,ij,j->abcdef', eps, g_inv, p_form, optimize=True)
    elif p == 2:
        q_form = fac_norm * sqrt_g * np.einsum('ijabcde,ik,jl,kl->abcde', eps, g_inv, g_inv, p_form, optimize=True)
    elif p == 3:
        q_form = fac_norm * sqrt_g * np.einsum('ijkabcd,im,jn,ko,mno->abcd', eps, g_inv, g_inv, g_inv, p_form, optimize=True)
    elif p == 4:
        q_form = fac_norm * sqrt_g * np.einsum('ijklabc,im,jn,ko,lp,mnop->abc', eps, g_inv, g_inv, g_inv, g_inv, p_form, optimize=True)
    elif p == 5:
        q_form = fac_norm * sqrt_g * np.einsum('ijklmab,in,jo,kp,lq,mr,nopqr->ab', eps, g_inv, g_inv, g_inv, g_inv, g_inv, p_form, optimize=True)
    elif p == 6:
        q_form = fac_norm * sqrt_g * np.einsum('ijklmna,io,jp,kq,lr,ms,nt,opqrst->a', eps, g_inv, g_inv, g_inv, g_inv, g_inv, g_inv, p_form, optimize=True)
    elif p == 7:
        q_form = fac_norm * sqrt_g * np.einsum('ijklmno,ip,jq,kr,ls,mt,nu,ov,pqrstuv->', eps, g_inv, g_inv, g_inv, g_inv, g_inv, g_inv, g_inv, p_form, optimize=True)
    else:
        raise ValueError("The Hodge dual is not defined for p-forms with p > 7.") 
    return q_form