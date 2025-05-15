'''Sampling the Link manifold'''
# Import libraries
import tensorflow as tf
import numpy as np
import os
import yaml
import pickle as pickle

# Import functions
from geometry.geometry import CoordChange_C5R10, kahler_form_real, holomorphic_volume_form_to_real_tensor
from geometry.wedge_product import wedge_product

# Import cymetric functions
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel

###########################################################################
# Function to generate the Link data (local pt coordinates and the G2 3-form)
def LinkSample(n_pts): ###change whats passed here?
    # Import the cymodel config info
    with open(os.path.dirname(os.path.dirname(__file__))+'/models/cy_models/cy_model_config.yaml', 'r') as f:
        config = yaml.unsafe_load(f)
    
    #Set up point generator
    pg = PointGenerator(config['monomials'], config['coefficients'], config['kmoduli'], config['ambient']) ###...better to import directly?

    #Generate points
    dirname = os.path.dirname(os.path.dirname(__file__))+'/models/cy_models/link_data'
    kappa = pg.prepare_dataset(n_pts, dirname, val_split=0.)
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    pg.prepare_basis(dirname, kappa=kappa)
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    points = data['X_train']
    BASIS = np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True)
    BASIS = prepare_tf_basis(BASIS)
    cy_points = CoordChange_C5R10(points,inverse=True)
        
    # Reimport the cymetric model
    nn_phi = tf.keras.Sequential()
    nn_phi.add(tf.keras.Input(shape=(config['n_in'])))
    for i in range(config['nlayer']):
        nn_phi.add(tf.keras.layers.Dense(config['nHidden'], activation=config['act']))
    nn_phi.add(tf.keras.layers.Dense(config['n_out'], use_bias=False))
    cymodel_filepath = os.path.dirname(os.path.dirname(__file__))+'/models/cy_models/cy_metric_model.keras'
    cymetric_model = PhiFSModel(nn_phi, BASIS, alpha=config['alpha'])
    cymetric_model.nn_phi = tf.keras.models.load_model(cymodel_filepath)
    
    # Transform to the link coordinate system
    thetas = np.random.uniform(low=0., high=2*np.pi, size=cy_points.shape[0]) #...sample a random angle
    c3_coords = cy_points[:, 2:] #..remove the 0th index as setting patch 0, then the 1st index with the CY eqn
    link_points_local = np.concatenate((np.real(c3_coords), np.imag(c3_coords), thetas.reshape(-1, 1)), axis=1) #...generate the link local coordinates
    
    # Define the geometric components
    holomorphic_volume_form = pg.holomorphic_volume_form(cy_points)
    hvf_r, hvf_i = holomorphic_volume_form_to_real_tensor(holomorphic_volume_form)
    hermitian_metric = cymetric_model(CoordChange_C5R10(cy_points)).numpy()
    kahler_form_R6 = kahler_form_real(hermitian_metric)
    kahler_form_R7 = np.pad(kahler_form_R6, ((0,0), (0,1), (0,1)), mode='constant')
    
    ### for TOMAS
    #...currently functional code:
    dthetas = np.concatenate((np.zeros((cy_points.shape[0], 6)), thetas.reshape(-1, 1)), axis=1)
    g2form_R7 = np.array([wedge_product(kahler_form_R7[idx], dthetas[idx]) for idx in range(cy_points.shape[0])])
    #...to replace with batch compatible code below:
    #g2form_R7 = wedge_product(kahler_form_R7, np.concatenate((np.zeros((cy_points.shape[0], 6)), thetas.reshape(-1, 1)), axis=1)) 
    #...below is the old code (delete when happy)
    #g2form_R7 = wedge_form2_with_form1(kahler_form_R6, np.concatenate((np.zeros((cy_points.shape[0], 6)), thetas.reshape(-1, 1)), axis=1)) 
    ###
    
    g2form_R7[:, :6, :6, :6] += hvf_i

    return (link_points_local, g2form_R7)

###########################################################################
if __name__ == '__main__':
    
    # Generate a link data sample
    num_pts = int(1e2)
    link_pts, link_phis = LinkSample(num_pts)
    
    print(f'Link pts shape:     {link_pts.shape}\nLink 3-forms shape: {link_phis.shape}')
    
    
    
    
    
    
    
    
    
    

