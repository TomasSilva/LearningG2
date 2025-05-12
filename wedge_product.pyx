# cython: language_level=3
import numpy as np
cimport numpy as np
from math import factorial  # Use Python's version
from itertools import permutations

cdef int permutation_sign(tuple p):
    cdef int i, j, inv = 0
    cdef int d = len(p)
    for i in range(d):
        for j in range(i + 1, d):
            if p[i] > p[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1

def wedge_product(np.ndarray A, np.ndarray B):
    cdef int p = A.ndim
    cdef int q = B.ndim
    cdef int n = A.shape[0]
    cdef int d = p + q
    cdef int i

    cdef np.ndarray tensor_prod = np.tensordot(A, B, axes=0).reshape((n,) * d)
    cdef np.ndarray result = np.zeros_like(tensor_prod)

    cdef list perms = list(permutations(range(d)))
    cdef int num_perms = len(perms)

    cdef tuple perm
    cdef int sign
    for i in range(num_perms):
        perm = perms[i]
        sign = permutation_sign(perm)
        permuted = np.transpose(tensor_prod, axes=perm)
        if sign == 1:
            result += permuted
        else:
            result -= permuted

    factor = 1.0 / (factorial(p) * factorial(q))
    return result * factor