'''Sampling the Link manifold'''
# Import libraries
import tensorflow as tf
import numpy as np
import os
import sys
import yaml
import pickle as pickle
import itertools

# Setup path for cymetric package
import pathlib
_parent_dir = pathlib.Path(__file__).parent.parent.parent
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
from geometry.geometry import kahler_form_real_matrix, holomorphic_volume_real_imag, compute_gG2
from geometry.patches import CoordChange_C5R10
from geometry.wedge import wedge
from models.model import get_model_path

# Import cymetric functions
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.helper import prepare_basis
from cymetric.models.models import MultFSModel

###########################################################################
# Class of the generated Link data (i.e. ambient pt coordinates, the G2 3-form, etc)
class LinkSample:
    def __init__(self, n_pts, cymodel_name='', target_patch=None, dataset_type='train'):
        """
        Generate Link manifold samples.
        
        Args:
            n_pts: Target number of points to generate
            cymodel_name: Name suffix for CY model files
            target_patch: Optional tuple (one_idx, dropped_idx) to filter for specific patch.
                         If provided, will generate points until n_pts matching this patch are found.
                         If None, generates n_pts points across all patches.
            dataset_type: String label for logging ('train', 'val', etc.)
        """
        # Set up the data paths
        self.n_pts = n_pts
        self.target_patch = target_patch
        self.dataset_type = dataset_type
        self.cymodel_name = cymodel_name
        self.dirname = os.path.dirname(os.path.dirname(__file__)) + '/models/cy_models/link_data'
        self.config_path = os.path.dirname(os.path.dirname(__file__)) + f'/models/cy_models/cy_model_config{cymodel_name}.yaml'
        cymodel_base = os.path.dirname(os.path.dirname(__file__)) + f'/models/cy_models/cy_metric_model{cymodel_name}'
        self.cymodel_path = get_model_path(cymodel_base + '.keras')
        
        # Run the data generation
        self._load_config()
        self._generate_points()
        self._prepare_cymodel()
        self._compute_geometry()

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.unsafe_load(f)

    def _generate_points(self):
        # Set up the point generator for the CY
        self.pg = PointGenerator(self.config['monomials'], self.config['coefficients'], self.config['kmoduli'], self.config['ambient'])
        
        if self.target_patch is None:
            # Standard generation: n_pts points across all patches
            kappa = self.pg.prepare_dataset(self.n_pts, self.dirname, val_split=0.)
            self.pg.prepare_basis(self.dirname, kappa=kappa)
            data = np.load(os.path.join(self.dirname, 'dataset.npz'))
            self.points = data['X_train']
            self.cy_points_C5 = CoordChange_C5R10(self.points, inverse=True)
            BASIS = np.load(os.path.join(self.dirname, 'basis.pickle'), allow_pickle=True)
            self.BASIS = prepare_basis(BASIS)
            
            # Identify coordinates dropped in the patching
            self.one_idxs = np.argmax(np.isclose(self.cy_points_C5, complex(1, 0)), axis=1)
            self.dropped_idxs = self.pg._find_max_dQ_coords(self.cy_points_C5)
        else:
            # Filtered generation: keep generating until we have n_pts from target_patch
            target_one_idx, target_dropped_idx = self.target_patch
            
            # Initial generation (oversample to reduce iterations)
            # Expect ~5% per patch (20 patches), so 5x oversample should usually suffice
            batch_size = max(self.n_pts * 5, 1000)
            collected_points = []
            collected_one_idxs = []
            collected_dropped_idxs = []
            
            iteration = 0
            total_collected = 0
            while total_collected < self.n_pts:
                iteration += 1
                # Generate batch
                kappa = self.pg.prepare_dataset(batch_size, self.dirname, val_split=0.)
                if iteration == 1:  # Only prepare basis once
                    self.pg.prepare_basis(self.dirname, kappa=kappa)
                data = np.load(os.path.join(self.dirname, 'dataset.npz'))
                batch_points = data['X_train']
                batch_cy_points = CoordChange_C5R10(batch_points, inverse=True)
                
                # Identify patches
                batch_one_idxs = np.argmax(np.isclose(batch_cy_points, complex(1, 0)), axis=1)
                batch_dropped_idxs = self.pg._find_max_dQ_coords(batch_cy_points)
                
                # Filter for target patch
                mask = (batch_one_idxs == target_one_idx) & (batch_dropped_idxs == target_dropped_idx)
                n_matched = np.sum(mask)
                
                if n_matched > 0:
                    collected_points.append(batch_points[mask])
                    collected_one_idxs.append(batch_one_idxs[mask])
                    collected_dropped_idxs.append(batch_dropped_idxs[mask])
                    total_collected += n_matched
                    if iteration == 1:
                        print(f"  Filtering {self.dataset_type} for patch [{target_one_idx}, {target_dropped_idx}]: "
                              f"{n_matched}/{batch_size} → {total_collected}/{self.n_pts}")
                    else:
                        print(f"    Iteration {iteration}: {n_matched}/{batch_size} → {total_collected}/{self.n_pts}")
            
            # Combine and truncate to exactly n_pts
            self.points = np.concatenate(collected_points, axis=0)[:self.n_pts]
            self.cy_points_C5 = CoordChange_C5R10(self.points, inverse=True)
            self.one_idxs = np.concatenate(collected_one_idxs, axis=0)[:self.n_pts]
            self.dropped_idxs = np.concatenate(collected_dropped_idxs, axis=0)[:self.n_pts]
            
            # Load basis (already prepared in first iteration)
            BASIS = np.load(os.path.join(self.dirname, 'basis.pickle'), allow_pickle=True)
            self.BASIS = prepare_basis(BASIS)
        
        # Generate the link points in the local \mathbb{R}^7 coordinate system
        self.thetas = np.random.uniform(low=0., high=2*np.pi, size=self.cy_points_C5.shape[0]) #...sample a random angle
        mask = np.ones(self.cy_points_C5.shape, dtype=bool)
        samples = np.arange(self.cy_points_C5.shape[0])
        mask[samples, self.one_idxs] = False
        mask[samples, self.dropped_idxs] = False
        c3_coords = self.cy_points_C5[mask].reshape(self.cy_points_C5.shape[0], -1)
        self.link_points_local = np.concatenate((np.real(c3_coords), np.imag(c3_coords), self.thetas.reshape(-1, 1)), axis=1) 

    def _prepare_cymodel(self):
        nn = tf.keras.Sequential()
        nn.add(tf.keras.Input(shape=(self.config['n_in'],)))
        for i in range(self.config['nlayer']):
            nn.add(tf.keras.layers.Dense(self.config['nHidden'], activation=self.config['act']))
        nn.add(tf.keras.layers.Dense(self.config['n_out'], use_bias=False))
        loaded_nn = tf.keras.models.load_model(self.cymodel_path)
        self.cymetric_model = MultFSModel(loaded_nn, self.BASIS, alpha=self.config['alpha'])

        # nn_phi = tf.keras.Sequential()
        # nn_phi.add(tf.keras.Input(shape=(self.config['n_in'],)))
        # for _ in range(self.config['nlayer']):
        #     nn_phi.add(tf.keras.layers.Dense(self.config['nHidden'], activation=self.config['act']))
        # nn_phi.add(tf.keras.layers.Dense(self.config['n_out'], use_bias=False))

        # self.cymetric_model = MultFSModel(nn_phi, self.BASIS, alpha=self.config['alpha'])
        # self.cymetric_model.nn_phi = tf.keras.models.load_model(self.cymodel_path)

    def _compute_geometry(self):
        self.holomorphic_volume_form = self.pg.holomorphic_volume_form(self.cy_points_C5)
        hvf_r = []
        hvf_i = []

        for c in self.holomorphic_volume_form:
            ReOmega, ImOmega = holomorphic_volume_real_imag(c)
            hvf_r.append(ReOmega)
            hvf_i.append(ImOmega)
        hvf_r = np.array(hvf_r)
        hvf_i = np.array(hvf_i)

        hvf_r = np.pad(hvf_r, pad_width=((0,0), (0,1), (0,1), (0,1)), mode='constant',constant_values=0)
        hvf_i = np.pad(hvf_i, pad_width=((0,0), (0,1), (0,1), (0,1)), mode='constant',constant_values=0)

        self.hermitian_metric = self.cymetric_model(CoordChange_C5R10(self.cy_points_C5)).numpy()
        # Force hermitian symmetrization to handle GPU numerical precision
        self.hermitian_metric = 0.5 * (self.hermitian_metric + self.hermitian_metric.conj().transpose(0, 2, 1))
        self.kahler_form_R6 = np.array([kahler_form_real_matrix(self.hermitian_metric[i]) for i in range(len(self.hermitian_metric))])
        self.kahler_form_R7 = np.pad(self.kahler_form_R6, ((0,0), (0,1), (0,1)), mode='constant')

        # TODO: This should change to the non-trivial fibration case later
        self.dthetas = np.concatenate((np.zeros((self.cy_points_C5.shape[0], 6)), np.ones((self.cy_points_C5.shape[0], 1))), axis=1)

        self.g2form_R7 = np.array([wedge(self.kahler_form_R7[i], self.dthetas[i]) + hvf_r[i] for i in range(len(self.kahler_form_R7))])

        self.star_g2form_R7 = np.array([(1/2)*wedge(self.kahler_form_R7[i], self.kahler_form_R7[i]) + wedge(self.dthetas[i], hvf_i[i]) for i in range(len(self.kahler_form_R7))])
        
        # self.holomorphic_volume_form = self.pg.holomorphic_volume_form(self.cy_points_C5)
        # hvf_r, hvf_i = holomorphic_volume_form_to_real(self.holomorphic_volume_form)

        # self.hermitian_metric = self.cymetric_model(CoordChange_C5R10(self.cy_points_C5)).numpy()
        # # Force hermitian symmetrization to handle GPU numerical precision
        # self.hermitian_metric = 0.5 * (self.hermitian_metric + self.hermitian_metric.conj().transpose(0, 2, 1))
        # kahler_form_R6 = hermitian_to_kahler_real(self.hermitian_metric)
        # self.kahler_form_R7 = np.pad(kahler_form_R6, ((0,0), (0,1), (0,1)), mode='constant')

        # self.dthetas = np.concatenate((np.zeros((self.cy_points_C5.shape[0], 6)), np.ones((self.cy_points_C5.shape[0], 1))), axis=1)
        # self.g2form_R7 = np.array([wedge(self.kahler_form_R7[i], self.dthetas[i]) for i in range(self.cy_points_C5.shape[0])])
        # self.g2form_R7[:, :6, :6, :6] += hvf_r

    def link_points(self, local=True):
        if local:
            return self.link_points_local
        else:
            if not hasattr(self, 'link_points_R10'):
                self.cy_points_on_S9 = self.cy_points_C5/np.linalg.norm(self.cy_points_C5, axis=1).reshape(-1, 1)
                self.link_points_C5 = self.cy_points_on_S9 * np.exp(1j * self.thetas.reshape(-1, 1)) #...multiple that angle in, the plane origin is the normalised cy point and orientation is implicitly set
                self.link_points_R10 = CoordChange_C5R10(self.link_points_C5)
            return self.link_points_R10
        
    @property
    def kahler_form(self):
        return self.kahler_form_R7
        
    @property
    def g2_form(self):
        return self.g2form_R7

    @property
    def g2_metric(self):
        if not hasattr(self, '_g2_metric'):
            self._g2_metric = np.array([compute_gG2(g2) for g2 in self.g2form_R7])
        return self._g2_metric

def levi_civita_6():
    """Return ε_{abcdef} for dimension 6 as an int8 array of shape (6,6,6,6,6,6)."""
    eps = np.zeros((6, 6, 6, 6, 6, 6), dtype=np.int8)

    def sign_of_perm(p):
        inv = 0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] > p[j]:
                    inv += 1
        return 1 if (inv % 2 == 0) else -1

    for p in itertools.permutations(range(6)):
        eps[p] = sign_of_perm(p)

    return eps

def holomorphic_volume_real_imag(c):
    """
    Given a complex scalar c (np.complex128) such that
    Omega = c * dz1 ∧ dz2 ∧ dz3 with z_i = x_i + i y_i,
    return (ReOmega, ImOmega) as real (6,6,6) tensors in the basis
    (x1, x2, x3, y1, y2, y3).
    """
    # basis order: 0:x1, 1:x2, 2:x3, 3:y1, 4:y2, 5:y3
    dz1 = np.zeros(6, dtype=np.complex128)
    dz2 = np.zeros(6, dtype=np.complex128)
    dz3 = np.zeros(6, dtype=np.complex128)

    # dz1 = dx1 + i dy1
    dz1[0] = 1.0
    dz1[3] = 1.0j

    # dz2 = dx2 + i dy2
    dz2[1] = 1.0
    dz2[4] = 1.0j

    # dz3 = dx3 + i dy3
    dz3[2] = 1.0
    dz3[5] = 1.0j

    eps6 = levi_civita_6()

    # Omega_{def} = c * ε_{abcdef} dz1_a dz2_b dz3_c
    Omega = c * np.einsum("abcdef,a,b,c->def", eps6, dz1, dz2, dz3)

    ReOmega = np.real(Omega)
    ImOmega = np.imag(Omega)

    return ReOmega, ImOmega

def kahler_form_real_matrix(g):
    """
    g: (n, n) complex Hermitian matrix g_{j \bar{k}}
    returns: (2n, 2n) real antisymmetric matrix W
             representing the Kähler form ω in the basis
             (x1,...,xn,y1,...,yn)
    """
    n = g.shape[0]
    W = np.zeros((2*n, 2*n), dtype=float)

    for j in range(n):
        for k in range(n):
            a = g[j, k].real  # Re g_{j\bar{k}}
            b = g[j, k].imag  # Im g_{j\bar{k}}
            xj, yj = 2*j, 2*j + 1
            xk, yk = 2*k, 2*k + 1

            # Terms with Re(g): a (dx^j∧dy^k - dy^j∧dx^k)
            W[xj, yk] += a
            W[yk, xj] -= a
            W[yj, xk] -= a
            W[xk, yj] += a

            # Terms with Im(g): -b (dx^j∧dx^k + dy^j∧dy^k)
            W[xj, xk] += -b
            W[xk, xj] -= -b
            W[yj, yk] += -b
            W[yk, yj] -= -b

    return W