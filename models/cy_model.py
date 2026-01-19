'''Code to train the CY metric'''
# Import libraries
import numpy as np
import os
import sys
import yaml
import pickle
import tensorflow as tf

# Setup path for cymetric package  
import pathlib
_parent_dir = pathlib.Path(__file__).parent.parent.parent
_cymetric_dir = _parent_dir / "cymetric"
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_cymetric_dir) not in sys.path:
    sys.path.insert(0, str(_cymetric_dir))

# Create alias to fix cymetric internal imports
# import cymetric
# if hasattr(cymetric, 'cymetric'):
#     sys.modules['cymetric'] = cymetric.cymetric

# Import cymetric functions
from cymetric.pointgen.pointgen import PointGenerator
# from cymetric.models.callbacks import SigmaCallback
# from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.callbacks import RicciCallback, SigmaCallback, VolkCallback, KaehlerCallback, TransitionCallback
from cymetric.models.models import MultFSModel
# from cymetric.models.metrics import SigmaLoss
# from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.helper import prepare_basis, train_model

###########################################################################
if __name__ == '__main__':
    # Generate the training data
    monomials = 5*np.eye(5, dtype=np.int64)
    coefficients = np.ones(5)
    kmoduli = np.ones(1)
    ambient = np.array([4])
    
    pg = PointGenerator(monomials, coefficients, kmoduli, ambient)
    #points = pg.generate_points(100)
    
    # Save the training data
    dirname = os.path.dirname(__file__)+'/cy_models/train_data'
    n_p = 100000
    kappa = pg.prepare_dataset(n_p, dirname)
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    pg.prepare_basis(dirname, kappa=kappa)
    
    # Load the training data
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    BASIS = np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True)
    BASIS = prepare_basis(BASIS)
    
    rcb = RicciCallback((data['X_val'], data['y_val']), data['val_pullbacks'])
    scb = SigmaCallback((data['X_val'], data['y_val']))
    volkcb = VolkCallback((data['X_val'], data['y_val']))
    kcb = KaehlerCallback((data['X_val'], data['y_val']))
    tcb = TransitionCallback((data['X_val'], data['y_val']))
    cb_list = [rcb, scb, kcb, tcb, volkcb]
    
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
    
    fmodel = MultFSModel(nn, BASIS, alpha=alpha)
    
    opt = tf.keras.optimizers.Adam()
    
    fmodel, training_history = train_model(fmodel, data, optimizer=opt, epochs=nEpochs, batch_sizes=bSizes, verbose=1, callbacks=cb_list)
        
   
    
    # Save the trained model
    save_filepath = os.path.dirname(__file__)+'/cy_models/'
    # cymodel_name = '_test'
    cymodel_name = ''
    fmodel.save(save_filepath+'cy_metric_model'+cymodel_name+'.keras')
    
    # Save the cymetric hyperparameters
    config = {
        'n_in': n_in,
        'n_out': n_out,
        'nlayer': nlayer,
        'nHidden': nHidden,
        'act': act,
        'alpha': alpha,
        'monomials': monomials,
        'coefficients': coefficients,
        'kmoduli': kmoduli,
        'ambient': ambient
    }
    with open(save_filepath+'cy_model_config'+cymodel_name+'.yaml', 'w') as f:
        yaml.dump(config, f)
