'''Sampling the Link manifold'''
# Import libraries
import tensorflow as tf
import numpy as np
import os
import sys
import yaml
import pickle as pickle

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
from geometry.geometry import hermitian_to_kahler_real, holomorphic_volume_form_to_real, compute_gG2
from geometry.patches import CoordChange_C5R10
from geometry.wedge_product import wedge_product

# Import cymetric functions
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel

###########################################################################
# Class of the generated Link data (i.e. ambient pt coordinates, the G2 3-form, etc)
class LinkSample:
    def __init__(self, n_pts, cymodel_name=''):
        # Set up the data paths
        self.n_pts = n_pts
        self.cymodel_name = cymodel_name
        self.dirname = os.path.dirname(os.path.dirname(__file__)) + '/models/cy_models/link_data'
        self.config_path = os.path.dirname(os.path.dirname(__file__)) + f'/models/cy_models/cy_model_config{cymodel_name}.yaml'
        self.cymodel_path = os.path.dirname(os.path.dirname(__file__)) + f'/models/cy_models/cy_metric_model{cymodel_name}.keras'
        
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
        kappa = self.pg.prepare_dataset(self.n_pts, self.dirname, val_split=0.)
        self.pg.prepare_basis(self.dirname, kappa=kappa)
        data = np.load(os.path.join(self.dirname, 'dataset.npz'))
        self.points = data['X_train']
        self.cy_points_C5 = CoordChange_C5R10(self.points, inverse=True)
        BASIS = np.load(os.path.join(self.dirname, 'basis.pickle'), allow_pickle=True)
        self.BASIS = prepare_tf_basis(BASIS)
        
        # Identify coordinates dropped in the patching
        self.one_idxs = np.argmax(np.isclose(self.cy_points_C5, complex(1, 0)), axis=1) #...find the indices set to 1 to define the wps patch
        self.dropped_idxs = self.pg._find_max_dQ_coords(self.cy_points_C5) #...find the indices set to 1 to define the cy patch
        
        # Generate the link points in the local \mathbb{R}^7 coordinate system
        self.thetas = np.random.uniform(low=0., high=2*np.pi, size=self.cy_points_C5.shape[0]) #...sample a random angle
        mask = np.ones(self.cy_points_C5.shape, dtype=bool)
        samples = np.arange(self.cy_points_C5.shape[0])
        mask[samples, self.one_idxs] = False
        mask[samples, self.dropped_idxs] = False
        c3_coords = self.cy_points_C5[mask].reshape(self.cy_points_C5.shape[0], -1)
        self.link_points_local = np.concatenate((np.real(c3_coords), np.imag(c3_coords), self.thetas.reshape(-1, 1)), axis=1) 

    def _prepare_cymodel(self):
        nn_phi = tf.keras.Sequential()
        nn_phi.add(tf.keras.Input(shape=(self.config['n_in'],)))
        for _ in range(self.config['nlayer']):
            nn_phi.add(tf.keras.layers.Dense(self.config['nHidden'], activation=self.config['act']))
        nn_phi.add(tf.keras.layers.Dense(self.config['n_out'], use_bias=False))

        self.cymetric_model = PhiFSModel(nn_phi, self.BASIS, alpha=self.config['alpha'])
        self.cymetric_model.nn_phi = tf.keras.models.load_model(self.cymodel_path)

    def _compute_geometry(self):
        self.holomorphic_volume_form = self.pg.holomorphic_volume_form(self.cy_points_C5)
        hvf_r, hvf_i = holomorphic_volume_form_to_real(self.holomorphic_volume_form)

        hermitian_metric = self.cymetric_model(CoordChange_C5R10(self.cy_points_C5)).numpy()
        kahler_form_R6 = hermitian_to_kahler_real(hermitian_metric)
        self.kahler_form_R7 = np.pad(kahler_form_R6, ((0,0), (0,1), (0,1)), mode='constant')

        self.dthetas = np.concatenate((np.zeros((self.cy_points_C5.shape[0], 6)), np.ones((self.cy_points_C5.shape[0], 1))), axis=1)
        self.g2form_R7 = np.array([wedge_product(self.kahler_form_R7[i], self.dthetas[i]) for i in range(self.cy_points_C5.shape[0])])
        self.g2form_R7[:, :6, :6, :6] += hvf_r

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

    

###########################################################################
if __name__ == '__main__':
    
    # Generate a link data sample
    n_pts = int(1e1)
    link_dataset = LinkSample(n_pts=n_pts)
    link_coords = link_dataset.link_points()
    g2_form = link_dataset.g2_form
    #g2_metric = link_dataset.g2_metric
    #kahler_form = link_dataset.kahler_form
    
    print(f'Link pts shape:     {link_coords.shape}\nLink 3-forms shape: {g2_form.shape}')
    
    
    
    
    
    
    
    
    
    

