'''Geometric functions'''
# Import libraries
import numpy as np
import itertools
import tensorflow as tf
from itertools import product, permutations
from geometry.compression import vec_to_form
from geometry.wedge import wedge
from functools import lru_cache

def find_max_dQ_coords(points, BASIS):
    r"""Finds the coordinates for which |dQ/dz| is largest.

    Args:
        points (ndarray[(n_p, ncoords), np.complex128]): Points.

    Returns:
        ndarray[(n_p), np.int64]: max(dQdz) indices
    """
    dQdz = np.abs(_compute_dQdz(points, BASIS))
    dQdz = dQdz * (~np.isclose(points, complex(1, 0)))
    return np.argmax(dQdz, axis=-1)

def _compute_dQdz(points, BASIS):
        r"""Computes dQdz at each point.

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.

        Returns:
            ndarray([n_p, ncoords], np.complex): dQdz at each point.
        """
        p_exp = np.expand_dims(np.expand_dims(points, 1), 1)
        dQdz = np.power(p_exp, BASIS['DQDZB0'])
        dQdz = np.multiply.reduce(dQdz, axis=-1)
        dQdz = np.multiply(BASIS['DQDZF0'], dQdz)
        dQdz = np.add.reduce(dQdz, axis=-1)
        return dQdz

def riemannian_metric_real_matrix(g):
    """
    g: (n, n) complex Hermitian matrix g_{j \bar{k}}
    returns: (2n, 2n) real symmetric matrix G representing the
             Riemannian metric in the basis (x1,...,xn,y1,...,yn),
             where z^j = x^j + i y^j.

    Convention: ds^2 = 2 * sum_{j,k} g_{j\bar k} dz^j d\bar z^k
                (common in Kähler geometry). With this convention,
                G is exactly the real metric on (x,y).
    """
    g = np.asarray(g)
    n = g.shape[0]
    A = np.real(g)
    B = np.imag(g)

    G = np.zeros((2*n, 2*n), dtype=float)

    # Block form in (x..., y...) order:
    # G_xx = A,  G_xy = -B
    # G_yx = B,  G_yy = A
    G[:n, :n] = A
    G[:n, n:] = -B
    G[n:, :n] = B
    G[n:, n:] = A

    # numerical hygiene: force symmetry
    G = 0.5 * (G + G.T)
    return G

def kahler_form_real_matrix(g, half=False):
    """
    g: (n, n) complex Hermitian matrix g_{j \bar{k}}
    returns: (2n, 2n) real antisymmetric matrix W for ω
             in the basis (x1,...,xn,y1,...,yn)
    Convention: ω = i * sum g_{j\bar k} dz^j ∧ d\bar z^k
                (set half=True for ω = (i/2) ... )
    """
    n = g.shape[0]
    W = np.zeros((2*n, 2*n), dtype=float)

    for j in range(n):
        for k in range(n):
            a = g[j, k].real
            b = g[j, k].imag
            xj, yj = j, n + j
            xk, yk = k, n + k

            # Re(g): a (dx^j∧dy^k - dy^j∧dx^k)
            W[xj, yk] += a
            W[yk, xj] -= a
            W[yj, xk] -= a
            W[xk, yj] += a

            # Im(g): -b (dx^j∧dx^k + dy^j∧dy^k)
            W[xj, xk] += -b
            W[xk, xj] -= -b
            W[yj, yk] += -b
            W[yk, yj] -= -b

    if half:
        W *= 0.5

    # optional numerical hygiene:
    W = 0.5 * (W - W.T)

    return W

def holomorphic_volume_real_imag(c):
    """
    Omega = c * dz1 ∧ dz2 ∧ dz3, with z_i = x_i + i y_i.
    Returns ReOmega, ImOmega as real (6,6,6) arrays in basis
    (x1, x2, x3, y1, y2, y3).

    Convention: Omega = (1/3!) * Omega_{ijk} e^i∧e^j∧e^k
    so Omega_{ijk} is fully antisymmetric.
    """
    dz1 = np.zeros(6, dtype=np.complex128)
    dz2 = np.zeros(6, dtype=np.complex128)
    dz3 = np.zeros(6, dtype=np.complex128)

    dz1[0] = 1.0; dz1[3] = 1.0j  # dx1 + i dy1
    dz2[1] = 1.0; dz2[4] = 1.0j  # dx2 + i dy2
    dz3[2] = 1.0; dz3[5] = 1.0j  # dx3 + i dy3

    T = np.einsum("i,j,k->ijk", dz1, dz2, dz3)

    # 6 * antisymmetrization: sum_{σ∈S3} sgn(σ) T_{σ(i)σ(j)σ(k)}
    Omega = c * (
        T
        + np.transpose(T, (1, 2, 0))
        + np.transpose(T, (2, 0, 1))
        - np.transpose(T, (0, 2, 1))
        - np.transpose(T, (2, 1, 0))
        - np.transpose(T, (1, 0, 2))
    )

    return Omega.real, Omega.imag


###########################################################################
# Functions related to the G2-structure

@lru_cache(None)
def levi_civita_7(dtype=np.float64):
    eps = np.zeros((7,)*7, dtype=dtype)
    for p in itertools.permutations(range(7)):
        s = 1
        for i in range(7):
            for j in range(i+1, 7):
                if p[i] > p[j]:
                    s *= -1
        eps[p] = s
    return eps

def compute_gG2(phi, tol_eig=1e-12):
    phi = np.asarray(phi, dtype=np.float64)
    eps = levi_civita_7(phi.dtype)

    # B_ij = (1/24) * phi_{i a b} phi_{j c d} phi_{e f g} eps^{a b c d e f g}
    B = (1.0/24.0) * np.einsum('iab,jcd,efg,abcdefg->ij', phi, phi, phi, eps)
    B = 0.5 * (B + B.T)

    # Use slogdet (stable) + abs root like your original
    sign, logdet = np.linalg.slogdet(B)
    if sign == 0:
        raise ValueError("B is singular (slogdet sign=0). φ likely degenerate / numerical issues.")
    root = np.exp(logdet / 9.0)  # = |detB|^(1/9) up to sign tracked separately

    factor = (6.0 ** (-2.0/9.0)) / root
    g = factor * B
    g = 0.5 * (g + g.T)

    # Match your orientation-fix behavior
    if np.linalg.eigvalsh(g).min() <= 0:
        g = -g

    return g

