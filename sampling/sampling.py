'''Sampling the Link manifold'''
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # .../LearningG2
sys.path.insert(0, str(PROJECT_ROOT))
print("PROJECT_ROOT:", PROJECT_ROOT)
# Import libraries
import tensorflow as tf
import numpy as np
import os
import yaml
import pickle as pickle
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
tfk = tf.keras
# Setup path for cymetric package

_parent_dir = pathlib.Path.cwd().parent.parent
_cymetric_dir = _parent_dir / "cymetric"

if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_cymetric_dir) not in sys.path:
    sys.path.insert(0, str(_cymetric_dir))

# Create alias to fix cymetric internal imports
import cymetric
if hasattr(cymetric, 'cymetric'):
    sys.modules['cymetric'] = cymetric.cymetric

# Import functions
from geometry.geometry import kahler_form_real_matrix, holomorphic_volume_real_imag, riemannian_metric_real_matrix, compute_gG2, find_max_dQ_coords
# # from ..geometry.patches import CoordChange_C5R10
from geometry.wedge import wedge
# from models.model import get_model_path

# Import cymetric functions
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.helper import prepare_basis



from cymetric.models.models import MultFSModel


# def _find_max_dQ_coords(points, BASIS):
#     r"""Finds the coordinates for which |dQ/dz| is largest.

#     Args:
#         points (ndarray[(n_p, ncoords), np.complex128]): Points.

#     Returns:
#         ndarray[(n_p), np.int64]: max(dQdz) indices
#     """
#     dQdz = np.abs(_compute_dQdz(points, BASIS))
#     dQdz = dQdz * (~np.isclose(points, complex(1, 0)))
#     return np.argmax(dQdz, axis=-1)

# def _compute_dQdz(points, BASIS):
#         r"""Computes dQdz at each point.

#         Args:
#             points (ndarray([n_p, ncoords], np.complex128)): Points.

#         Returns:
#             ndarray([n_p, ncoords], np.complex): dQdz at each point.
#         """
#         p_exp = np.expand_dims(np.expand_dims(points, 1), 1)
#         dQdz = np.power(p_exp, BASIS['DQDZB0'])
#         dQdz = np.multiply.reduce(dQdz, axis=-1)
#         dQdz = np.multiply(BASIS['DQDZF0'], dQdz)
#         dQdz = np.add.reduce(dQdz, axis=-1)
#         return dQdz

def sampler_g2_package_R7(p, fmodel, BASIS, rotation=0):
        base_point = p
        applied_rotation = rotation
        
        point_cc = p[0:5] + 1.j*p[5:]
        drop_max = int(find_max_dQ_coords(point_cc, BASIS))
        drop_one = int(np.argmin(np.abs(point_cc - 1)))

        model_out = np.array(fmodel(np.expand_dims(p, axis=0))[0])
        
        riemannian_metric = riemannian_metric_real_matrix(model_out)
        
        w = kahler_form_real_matrix(model_out)
        
        
        w_R7 = np.pad(w, ((0, 1), (0, 1)), mode='constant')

        holomorphic_volume_form = 1/(5*point_cc[drop_max]**4)

        ReOmega, ImOmega = holomorphic_volume_real_imag(holomorphic_volume_form)

        ReOmega_R7 = np.pad(ReOmega, ((0, 1), (0, 1), (0, 1)), mode='constant')
        ImOmega_R7 = np.pad(ImOmega, ((0, 1), (0, 1), (0, 1)), mode='constant')

        u_coords = [i for i in range(5) if i != drop_max and i != drop_one]

        eta = np.array([0,0,0,0,0,0,1], dtype=np.float64)
        
        point_cc = np.exp(1.j*rotation)*point_cc
        
        u_count = 0
        for i in u_coords:
                factor = point_cc[i].conjugate() - (point_cc[drop_max].conjugate()*(point_cc[i]**4/point_cc[drop_max]**4))
                factor = factor / np.linalg.norm(point_cc)**2
                
                eta[u_count] = factor.imag
                eta[u_count+3] = factor.real
                
                u_count += 1
                
        link_pt = np.concatenate([(point_cc/np.linalg.norm(point_cc)).real, (point_cc/np.linalg.norm(point_cc)).imag])
        
        g2 = wedge(w_R7, eta) + ReOmega_R7

        star_g2 = (1/2)*wedge(w_R7, w_R7) + wedge(eta, ImOmega_R7)
        
        g2_metric = compute_gG2(g2)
        
        return base_point, link_pt, applied_rotation, g2, star_g2, riemannian_metric, g2_metric, drop_max, drop_one, eta
    

if __name__ == "__main__":
    
    dirname = './models/cy_models/train_data'

    data = np.load(os.path.join(dirname, 'dataset.npz'))
    BASIS = np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True)
    BASIS = prepare_basis(BASIS)
    
    print("Loaded data and basis.")
    
    nlayer = 3
    nHidden = 64
    act = 'gelu'
    nEpochs = 500
    bSizes = [64, 50000]
    alpha = [1., 1., 1., 1., 1.]
    nfold = 3
    n_in = 2*5
    n_out = nfold**2
    
    nn = tf.keras.Sequential()
    nn.add(tf.keras.Input(shape=(n_in,)))
    for i in range(nlayer):
        nn.add(tf.keras.layers.Dense(nHidden, activation=act))
    nn.add(tf.keras.layers.Dense(n_out, use_bias=False))
    
    loaded_nn = tf.keras.models.load_model("./models/cy_models/cy_metric_model.keras")
    fmodel = MultFSModel(loaded_nn, BASIS, alpha=alpha)
    
    print("Loaded model.")
    
  

    def compute_sample(point, rotation):
        return sampler_g2_package_R7(point, fmodel, BASIS, rotation=rotation)

    X = data["X_train"][:40000]

    tasks = []
    for pt in X:
        tasks.append((pt, 0.0))
        for _ in range(4):
            tasks.append((pt, np.random.uniform(0, 2*np.pi)))

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(compute_sample)(pt, rot)
        for pt, rot in tqdm(tasks, desc="Sampling G2 points")
    )

    base_points = np.stack([r[0] for r in results])
    link_points = np.stack([r[1] for r in results])
    rotations = np.stack([r[2] for r in results])
    phis = np.stack([r[3] for r in results])
    psis =  np.stack([r[4] for r in results])
    riemannian_metrics = np.stack([r[5] for r in results])
    g2_metrics = np.stack([r[6] for r in results])
    drop_maxs = np.stack([r[7] for r in results])
    drop_ones = np.stack([r[8] for r in results])
    etas = np.stack([r[9] for r in results])

    np.savez_compressed("./sampling/g2_dataset.npz", base_points=base_points, 
                        link_points=link_points, 
                        rotations=rotations, 
                        phis=phis, 
                        psis=psis, 
                        riemannian_metrics=riemannian_metrics, 
                        g2_metrics=g2_metrics,
                        drop_maxs=drop_maxs,
                        drop_ones=drop_ones,
                        etas=etas)

    
    print("Saved G2 dataset.")
    
    