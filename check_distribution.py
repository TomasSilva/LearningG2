import numpy as np

train_data = np.load('samples/link_data/g2_train.npz')
phi_train = train_data['phis']
# Check which key exists
if 'psis_vec' in train_data:
    psi_train = train_data['psis_vec']
else:
    psi_train = train_data['psis']

phi_norms = np.linalg.norm(phi_train, axis=1)
psi_norms = np.linalg.norm(psi_train, axis=1)

print('Training data distribution:')
print(f'  ||φ|| range: [{phi_norms.min():.6f}, {phi_norms.max():.6f}]')
print(f'  ||φ|| mean±std: {phi_norms.mean():.6f} ± {phi_norms.std():.6f}')
print(f'  ||ψ|| range: [{psi_norms.min():.6f}, {psi_norms.max():.6f}]')
print(f'  ||ψ|| mean±std: {psi_norms.mean():.6f} ± {psi_norms.std():.6f}')

print('\nOutliers have ||φ|| ~ 0.5-0.8, ||ψ|| ~ 0.4-0.7')
print(f'Outlier phi is {100*0.6/phi_norms.mean():.1f}% of training mean')
print(f'Outlier psi is {100*0.5/psi_norms.mean():.1f}% of training mean')

# Check percentiles
print(f'\n||φ|| percentiles:')
print(f'  1%: {np.percentile(phi_norms, 1):.6f}')
print(f'  5%: {np.percentile(phi_norms, 5):.6f}')
print(f' 10%: {np.percentile(phi_norms, 10):.6f}')

print(f'\n||ψ|| percentiles:')
print(f'  1%: {np.percentile(psi_norms, 1):.6f}')
print(f'  5%: {np.percentile(psi_norms, 5):.6f}')
print(f' 10%: {np.percentile(psi_norms, 10):.6f}')
