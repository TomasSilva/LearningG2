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
from geometry.geometry import kahler_form_real_matrix, holomorphic_volume_real_imag, compute_gG2
from geometry.patches import CoordChange_C5R10
from geometry.wedge_product import wedge_product
from models.model import get_model_path

# Import cymetric functions
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel

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
            self.BASIS = prepare_tf_basis(BASIS)
            
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
            self.BASIS = prepare_tf_basis(BASIS)
        
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
        # New function takes single complex number, so we need to loop for batch
        hvf_r_list = []
        hvf_i_list = []
        for hvf in self.holomorphic_volume_form:
            hvf_r_single, hvf_i_single = holomorphic_volume_real_imag(hvf)
            hvf_r_list.append(hvf_r_single)
            hvf_i_list.append(hvf_i_single)
        hvf_r = np.array(hvf_r_list)
        hvf_i = np.array(hvf_i_list)

        hermitian_metric = self.cymetric_model(CoordChange_C5R10(self.cy_points_C5)).numpy()
        # Force hermitian symmetrization to handle GPU numerical precision
        hermitian_metric = 0.5 * (hermitian_metric + hermitian_metric.conj().transpose(0, 2, 1))
        
        # New function takes single matrix, so we need to loop for batch
        kahler_form_R6 = np.array([kahler_form_real_matrix(hm) for hm in hermitian_metric])
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
# Data loading function for pre-computed g2_dataset.npz
def load_g2_dataset(n_samples, use_10d_input=False, metric=False, dataset_type='train', 
                     split_ratio=0.9, random_seed=42, dataset_path=None):
    """
    Load data from g2_dataset.npz file.
    
    Args:
        n_samples: Number of samples to load
        use_10d_input: If True, use 10D ambient coordinates as input.
                       If False, use 7D representation with 2D patch indices.
        metric: If True, output G2 metric (28D vector). If False, output 3-form (35D vector).
        dataset_type: 'train' or 'val' for train/validation split
        split_ratio: Ratio for train/val split (default 0.9)
        random_seed: Random seed for reproducible shuffling
        dataset_path: Path to g2_dataset.npz file. If None, uses default location.
    
    Returns:
        tuple: (input_coords, patch_indices, output_vecs)
               - input_coords: (n_samples, 10) or (n_samples, 7) depending on use_10d_input
               - patch_indices: (n_samples, 2) if use_10d_input=False, else None
               - output_vecs: (n_samples, 28) if metric=True, else (n_samples, 35)
    """
    # Import here to avoid circular dependency
    from geometry.compression import form_to_vec, metric_to_vec
    
    # Determine default dataset path (go up to workspace root)
    if dataset_path is None:
        current_dir = os.path.dirname(os.path.dirname(__file__))
        dataset_path = os.path.join(os.path.dirname(current_dir), 'g2_dataset.npz')
    
    # Load the dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    print(f"  Loading data from: {dataset_path}")
    # Use memory mapping to avoid loading entire dataset into RAM
    # Only the selected indices will be loaded into memory
    data = np.load(dataset_path, mmap_mode='r')
    
    # Get total samples without loading data into memory
    total_samples = data['base_points'].shape[0]
    
    # Create reproducible shuffle
    np.random.seed(random_seed)
    indices = np.random.permutation(total_samples)
    
    # Split into train/val
    split_idx = int(total_samples * split_ratio)
    if dataset_type == 'train':
        selected_indices = indices[:split_idx]
    elif dataset_type == 'val':
        selected_indices = indices[split_idx:]
    else:
        raise ValueError(f"dataset_type must be 'train' or 'val', got {dataset_type}")
    
    # Handle n_samples selection
    available_samples = len(selected_indices)
    if n_samples is None or n_samples >= available_samples:
        # Use all available data for this split
        n_samples_to_use = available_samples
        if n_samples is None:
            print(f"  Using all {available_samples} available samples ({dataset_type})")
        else:
            print(f"  Requested {n_samples} samples, using all {available_samples} available ({dataset_type})")
    else:
        # Use random subsample of requested size
        n_samples_to_use = n_samples
    
    selected_indices = selected_indices[:n_samples_to_use]
    
    # NOW load only the selected data into memory (memory efficient)
    print(f"  Loading {n_samples_to_use} samples into memory...")
    base_points_selected = data['base_points'][selected_indices]
    link_points_selected = data['link_points'][selected_indices]
    rotations_selected = data['rotations'][selected_indices]
    phis_selected = data['phis'][selected_indices]
    g2_metrics_selected = data['g2_metrics'][selected_indices]
    
    # Convert to output vectors based on metric flag
    import tensorflow as tf
    if metric:
        # Convert g2_metrics to 28D vectors
        output_vecs = metric_to_vec(tf.convert_to_tensor(g2_metrics_selected, dtype=tf.float32)).numpy()
    else:
        # Convert phis (3-forms) to 35D vectors
        output_vecs = form_to_vec(tf.convert_to_tensor(phis_selected, dtype=tf.float32)).numpy()
    
    if use_10d_input:
        # Use full 10D ambient coordinates (link points in R^10)
        input_coords = link_points_selected
        patch_indices = None
        print(f"  Loaded {n_samples} samples ({dataset_type}): 10D input mode")
    else:
        # Use 7D local representation: [Re(C^3), Im(C^3), theta]
        # This matches the original LinkSample.link_points(local=True) format
        
        # Convert R^10 to C^5 for base points (CY points)
        from geometry.patches import CoordChange_C5R10
        base_points_C5 = CoordChange_C5R10(base_points_selected, inverse=True)
        
        # Find which coordinate is set to 1 (the "one" coordinate in patching)
        # This is the coordinate with magnitude closest to 1
        one_idxs = np.argmax(np.abs(base_points_C5 - 1.0) < 0.1, axis=1)
        
        # Find dropped coordinate using heuristic: smallest magnitude (excluding one_idx)
        # This approximates the maximum dQ coordinate that would be dropped in patching
        magnitudes = np.abs(base_points_C5)
        dropped_idxs = np.zeros(n_samples, dtype=np.int32)
        for i in range(n_samples):
            mags_copy = magnitudes[i].copy()
            mags_copy[one_idxs[i]] = np.inf  # Exclude the one coordinate
            dropped_idxs[i] = np.argmin(mags_copy)
        
        # Create 7D representation: [Re(C^3), Im(C^3), theta]
        # Extract C^3 by removing one_idx and dropped_idx from C^5
        input_coords = np.zeros((n_samples, 7), dtype=np.float32)
        for i in range(n_samples):
            # Create mask to extract the 3 remaining complex coordinates
            mask = np.ones(5, dtype=bool)
            mask[one_idxs[i]] = False
            mask[dropped_idxs[i]] = False
            
            # Extract C^3 coordinates
            c3_coords = base_points_C5[i][mask]  # 3 complex numbers
            
            # Build 7D vector: [Re(c3), Im(c3), theta]
            input_coords[i, :3] = np.real(c3_coords)
            input_coords[i, 3:6] = np.imag(c3_coords)
            input_coords[i, 6] = rotations_selected[i]
        
        # Stack patch indices
        patch_indices = np.stack([one_idxs, dropped_idxs], axis=1).astype(np.int32)
        
        print(f"  Loaded {n_samples} samples ({dataset_type}): 7D input mode with patch info")
    
    return input_coords, patch_indices, output_vecs

    

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
    
    
    
    
    
    
    
    
    
    

