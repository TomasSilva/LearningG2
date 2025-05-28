import numpy as np
import itertools


def levi_cevita_tensor(dim):   
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr

def Hodge_Dual(p_form, g):
    # Define p and q
    p = len(p_form.shape)
    
    # Define the inverse metric
    g_inv = np.linalg.inv(g)
    
    # Compute the square root of the determinant of the metric
    sqrt_g = np.sqrt(np.linalg.det(g))
    
    # Define the Levi-Civita tensor
    eps = levi_cevita_tensor(7)
    
    # Compute the q-form in \Omega^{7-p}
    if p == 0:
        q_form = p_form * eps
    elif p == 1:
        q_form = sqrt_g * sum(sum(g_inv[a, t] * p_form[t] * eps[a] for a in range(7)) for t in range(7))
    elif p == 2:
        q_form = sqrt_g * sum(sum(sum(sum(g_inv[a, t] * g_inv[b, u] * p_form[t, u] * eps[a, b] for a in range(7)) for b in range(7)) for t in range(7)) for u in range(7))
    elif p == 3:
        q_form = sqrt_g * sum(sum(sum(sum(sum(sum(g_inv[a, t] * g_inv[b, u] * g_inv[c, v] * p_form[t, u, v] * eps[a, b, c] for a in range(7)) for b in range(7)) for c in range(7)) for t in range(7)) for u in range(7)) for v in range(7))
    elif p == 4:
        q_form = sqrt_g * sum(sum(sum(sum(sum(sum(sum(sum(g_inv[a, t] * g_inv[b, u] * g_inv[c, v] * g_inv[d, w] * p_form[t, u, v, w] * eps[a, b, c, d] for a in range(7)) for b in range(7)) for c in range(7)) for d in range(7)) for t in range(7)) for u in range(7)) for v in range(7)) for w in range(7))
    elif p == 5:
        q_form = sqrt_g * sum(sum(sum(sum(sum(sum(sum(sum(sum(sum(g_inv[a, t] * g_inv[b, u] * g_inv[c, v] * g_inv[d, w] * g_inv[e, x] * p_form[t, u, v, w, x] * eps[a, b, c, d, e] for a in range(7)) for b in range(7)) for c in range(7)) for d in range(7)) for e in range(7)) for t in range(7)) for u in range(7)) for v in range(7)) for w in range(7)) for x in range(7))
    elif p == 6:
        q_form = sqrt_g * sum(sum(sum(sum(sum(sum(sum(sum(sum(sum(sum(sum(g_inv[a,t] * g_inv[b,u] * g_inv[c,v] * g_inv[d,w] * g_inv[e,x] * g_inv[f,y] * p_form[t,u,v,w,x,y] * eps[a,b,c,d,e,f] for a in range(7)) for b in range(7)) for c in range(7)) for d in range(7)) for e in range(7)) for f in range(7)) for t in range(7)) for u in range(7)) for v in range(7)) for w in range(7)) for x in range(7)) for y in range(7))
    elif p == 7:
        q_form = sqrt_g * sum(sum(sum(sum(sum(sum(sum(sum(sum(sum(sum(sum(sum(sum(g_inv[a,t] * g_inv[b,u] * g_inv[c,v] * g_inv[d,w] * g_inv[e,x] * g_inv[f,y] * g_inv[h,z] * p_form[t,u,v,w,x,y,z] * eps[a,b,c,d,e,f,h] for a in range(7)) for b in range(7)) for c in range(7)) for d in range(7)) for e in range(7)) for f in range(7)) for h in range(7)) for t in range(7)) for u in range(7)) for v in range(7)) for w in range(7)) for x in range(7)) for y in range(7)) for z in range(7))
    else:
        raise ValueError("The Hodge dual is not defined for p-forms with p > 7.") 
    return q_form