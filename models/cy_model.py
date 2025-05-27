'''Code to train the CY metric'''
# Import libraries
import numpy as np
import os
import yaml
import pickle
import tensorflow as tf

# Import cymetric functions
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.callbacks import SigmaCallback
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.metrics import SigmaLoss
from cymetric.models.tfhelper import prepare_tf_basis, train_model

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
    n_p = 200000
    kappa = pg.prepare_dataset(n_p, dirname)
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    pg.prepare_basis(dirname, kappa=kappa)
    
    # Load the training data
    data = np.load(os.path.join(dirname, 'dataset.npz'))
    BASIS = np.load(os.path.join(dirname, 'basis.pickle'), allow_pickle=True)
    BASIS = prepare_tf_basis(BASIS)
    
    # Define the cymetric NN hyperparameters
    nfold = 3
    alpha = [1., 1., 1., 1., 1.]
    n_in = 2*5
    n_out = 1
    nlayer = 3
    nHidden = 64
    act = 'gelu'
    nEpochs = 200
    bSizes = [64, 50000]

    # Define the cymetric NN model (learnt via the Kahler potential)
    nn_phi = tf.keras.Sequential()
    nn_phi.add(tf.keras.Input(shape=(n_in)))
    for i in range(nlayer):
        nn_phi.add(tf.keras.layers.Dense(nHidden, activation=act))
    nn_phi.add(tf.keras.layers.Dense(n_out, use_bias=False))
    phimodel = PhiFSModel(nn_phi, BASIS, alpha=alpha)
    opt_phi = tf.keras.optimizers.Adam()
    scb = SigmaCallback((data['X_val'], data['y_val']))
    cb_list = [scb]
    cmetrics = [SigmaLoss()]
    
    # Train the model
    phimodel, training_history = train_model(phimodel, 
                                             data, 
                                             optimizer=opt_phi, 
                                             epochs=nEpochs, 
                                             batch_sizes=bSizes, 
                                             verbose=1, 
                                             custom_metrics=cmetrics, 
                                             callbacks=cb_list
                                             )
    
    # Save the trained model
    save_filepath = os.path.dirname(__file__)+'/cy_models/'
    cymodel_name = '_test'
    phimodel.save(save_filepath+'cy_metric_model'+cymodel_name+'.keras')
    
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
