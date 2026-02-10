import sys
sys.path.insert(0, '.')
import numpy as np
import tensorflow as tf
from geometry.compression import vec_to_form

# Load the 4form model
model = tf.keras.models.load_model('models/link_models/4form_run1.keras')

# Load test data
data = np.load('samples/link_data/g2_test.npz')
link_points = data['link_points'][:1]
etas = data['etas'][:1]
drop_maxs = data['drop_maxs'][:1]
drop_ones = data['drop_ones'][:1]

X_input = np.concatenate([link_points[0], etas[0], [drop_maxs[0], drop_ones[0]]])
X_input = np.expand_dims(X_input, axis=0)

# Predict
psi_vec = model.predict(X_input, verbose=0)[0]
print(f'Predicted psi_vec shape: {psi_vec.shape}')
print(f'Predicted psi_vec min/mean/max: {psi_vec.min():.6e}, {psi_vec.mean():.6e}, {psi_vec.max():.6e}')

# Load normalization stats
norm_data = np.load('models/link_models/4form_run1_norm_stats.npz')
y_mean = norm_data['y_mean']
y_std = np.sqrt(norm_data['y_variance'])
psi_vec_denorm = psi_vec * y_std + y_mean

print(f'Denormalized psi_vec min/mean/max: {psi_vec_denorm.min():.6e}, {psi_vec_denorm.mean():.6e}, {psi_vec_denorm.max():.6e}')

# Expand to 35 dims
try:
    index_map = np.load('models/link_models/4form_run1_index_map.npz')
    nonzero_indices = index_map['psi_nonzero_indices']
except FileNotFoundError:
    # Use default indices
    zero_indices = np.array([6, 8, 12, 15, 17, 18, 24, 25, 27, 29, 32, 33])
    nonzero_indices = np.array([i for i in range(35) if i not in zero_indices])

psi_vec_expanded = np.zeros(35)
psi_vec_expanded[nonzero_indices] = psi_vec_denorm
print(f'Expanded psi_vec shape: {psi_vec_expanded.shape}')

# Convert to 4-form tensor
psi = vec_to_form(psi_vec_expanded, n=7, k=4)
print(f'Psi tensor shape: {psi.shape}')
print(f'Psi norm: {np.linalg.norm(psi)}')
