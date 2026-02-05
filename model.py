"""
Model architecture and training utilities for learning G2 structures.
Supports both 3-form and metric learning tasks.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import itertools
from pathlib import Path


# -----------------------
# Data preprocessing
# -----------------------
def split_train_val_test(X, Y, train=0.90, val=0.05, seed=42):
    """
    Split data into train/validation/test sets.
    
    Parameters
    ----------
    X : array_like
        Input features
    Y : array_like
        Target values
    train : float
        Fraction for training set
    val : float
        Fraction for validation set
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple of tuples
        ((X_train, Y_train), (X_val, Y_val), (X_test, Y_test))
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    assert X.shape[0] == Y.shape[0]
    N = X.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_tr = int(train * N)
    n_va = int(val * N)
    tr = idx[:n_tr]
    va = idx[n_tr:n_tr + n_va]
    te = idx[n_tr + n_va:]

    return (X[tr], Y[tr]), (X[va], Y[va]), (X[te], Y[te])


def oriented35_components(T):
    """
    Extract 35 independent components from a 3-form tensor.
    
    Parameters
    ----------
    T : array_like, shape (7,7,7)
        Fully antisymmetric 3-form tensor
        
    Returns
    -------
    ndarray, shape (35,)
        The 35 = C(7,3) independent components
    """
    T = np.asarray(T)
    assert T.shape == (7, 7, 7)
    triples = list(itertools.combinations(range(7), 3))
    vals = np.array([T[i, j, k] for (i, j, k) in triples], dtype=T.dtype)
    return vals


def upper_triangular_part(A, include_diagonal=True):
    """
    Extract the upper triangular part of a 7x7 symmetric matrix.
    
    Parameters
    ----------
    A : array_like, shape (7,7)
        Symmetric matrix
    include_diagonal : bool
        Whether to include diagonal entries
        
    Returns
    -------
    ndarray, shape (28,) if include_diagonal else (21,)
        Upper triangular entries in row-major order
    """
    A = np.asarray(A)
    assert A.shape == (7, 7), "Input must be a 7x7 matrix"

    if include_diagonal:
        idx = np.triu_indices(7)
    else:
        idx = np.triu_indices(7, k=1)

    return A[idx]


def prepare_data(data, task='3form'):
    """
    Prepare X and Y from loaded npz data or dict.
    
    Parameters
    ----------
    data : dict-like or NpzFile
        Loaded .npz data or dict containing link_points, etas, drop_maxs, drop_ones,
        and either phis (3-form) or g2_metrics (metric)
    task : str
        Either '3form' or 'metric'
        
    Returns
    -------
    X : ndarray, shape (N, 19)
        Input features
    Y : ndarray, shape (N, 35) or (N, 28)
        Target outputs (35 for 3-form, 28 for metric)
    """
    # Handle both dict and NpzFile
    if isinstance(data, dict):
        link_points = data['link_points']
        etas = data['etas']
        drop_maxs = data['drop_maxs']
        drop_ones = data['drop_ones']
    else:
        # NpzFile object
        link_points = data['link_points']
        etas = data['etas']
        drop_maxs = data['drop_maxs']
        drop_ones = data['drop_ones']
    
    # Build input features: [link_points(10), etas(7), drop_max(1), drop_one(1)] = 19
    X = np.concatenate([
        link_points, 
        etas, 
        drop_maxs[:, None], 
        drop_ones[:, None]
    ], axis=1)
    
    if task == '3form':
        # Extract 35 independent components from 3-form
        if isinstance(data, dict):
            phis = data['phis']
        else:
            phis = data['phis']
        # Check if phis are already in component form or tensor form
        if phis.ndim == 2 and phis.shape[1] == 35:
            # Already in component form
            Y = phis
        else:
            # Full tensor form - extract components
            Y = np.array([oriented35_components(phi) for phi in phis])
    elif task == 'metric':
        # Extract 28 upper triangular components from symmetric metric
        if isinstance(data, dict):
            G = data["g2_metrics"]
        else:
            G = data["g2_metrics"]
        # Check if metrics are already in flattened form or full matrix form
        if G.ndim == 2 and G.shape[1] == 28:
            # Already in flattened form
            Y = G
        else:
            # Full matrix form - extract upper triangular components
            idx = np.triu_indices(7)
            Y = G[:, idx[0], idx[1]]
    else:
        raise ValueError(f"Unknown task: {task}. Must be '3form' or 'metric'")
    
    return X, Y


# -----------------------
# Model architecture
# -----------------------
def build_regressor(input_dim, output_dim, hidden=(512, 512, 256, 256), 
                   dropout=0.0, activation='gelu', use_bias=True, 
                   init_scale=1.0, l2_reg=0.0):
    """
    Build a feedforward neural network regressor.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    output_dim : int
        Dimension of output (35 for 3-form, 28 for metric)
    hidden : tuple of int
        Hidden layer sizes
    dropout : float
        Dropout rate (0.0 means no dropout)
    activation : str
        Activation function ('gelu', 'relu', etc.)
    use_bias : bool
        Whether to use bias terms in layers
    init_scale : float
        Scale for uniform initialization ([-scale, scale])
    l2_reg : float
        L2 regularization strength (0.0 means no regularization)
        
    Returns
    -------
    keras.Model
        The constructed model
    """
    # Kernel initializer
    if init_scale != 1.0:
        initializer = keras.initializers.RandomUniform(
            minval=-init_scale, maxval=init_scale
        )
    else:
        initializer = 'glorot_uniform'  # Default Keras initializer
    
    # Regularizer
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    inp = keras.Input(shape=(input_dim,), dtype=tf.float32)
    
    x = inp
    for w in hidden:
        x = layers.Dense(
            w, 
            activation=activation, 
            use_bias=use_bias,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
    
    out = layers.Dense(
        output_dim, 
        dtype=tf.float32, 
        use_bias=use_bias,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )(x)
    return keras.Model(inp, out)


# -----------------------
# Training
# -----------------------
def train_regressor(
    X, Y,
    X_val=None, Y_val=None,
    pretrained_model=None,
    task='3form',
    seed=42,
    batch=2048,
    val_batch=None,
    epochs=200,
    lr=1e-3,
    hidden=(512, 512, 256, 256),
    dropout=0.0,
    activation='gelu',
    use_bias=True,
    init_scale=1.0,
    l2_reg=0.0,
    huber_delta=None,
    lr_reduce_factor=0.5,
    lr_reduce_patience=8,
    min_lr=1e-6,
    early_stop_patience=20,
    validate=True,
    verbosity=1,
    checkpoint_path="regressor.keras",
):
    """
    Train a regressor for G2 structure learning.
    
    Parameters
    ----------
    X : ndarray, shape (N, input_dim)
        Input features (training data)
    Y : ndarray, shape (N, output_dim)
        Target values (training data)
    X_val : ndarray, shape (N_val, input_dim), optional
        Validation input features (if None and validate=True, uses split from X)
    Y_val : ndarray, shape (N_val, output_dim), optional
        Validation target values (if None and validate=True, uses split from Y)
    pretrained_model : keras.Model, optional
        Pre-trained model to continue training (if None, builds new model)
    task : str
        Task name ('3form' or 'metric') for naming
    seed : int
        Random seed
    batch : int
        Batch size for training
    val_batch : int, optional
        Batch size for validation (defaults to batch if None)
    epochs : int
        Maximum number of epochs
    lr : float
        Initial learning rate
    hidden : tuple of int
        Hidden layer sizes
    dropout : float
        Dropout rate
    activation : str
        Activation function
    use_bias : bool
        Whether to use bias in layers
    init_scale : float
        Parameter initialization scale
    l2_reg : float
        L2 regularization strength
    huber_delta : float or None
        Huber loss delta (None or inf for MSE)
    lr_reduce_factor : float
        Factor to reduce learning rate
    lr_reduce_patience : int
        Patience for learning rate reduction
    min_lr : float
        Minimum learning rate
    early_stop_patience : int
        Patience for early stopping
    validate : bool
        Whether to use validation
    verbosity : int
        Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
    checkpoint_path : str or Path
        Path to save the best model
        
    Returns
    -------
    model : keras.Model
        Trained model (outputs normalized predictions)
    history : keras.callbacks.History
        Training history
    test_data : tuple
        (X_test, Y_test, Y_pred) - predictions are in original scale
    normalizers : tuple
        (x_norm, y_norm) - the normalization layers
    test_metrics : dict
        Test metrics in normalized space
    """
    if val_batch is None:
        val_batch = batch
    
    # Handle validation data
    if validate and X_val is not None and Y_val is not None:
        # Use provided validation data
        Xtr, Ytr = X, Y
        Xva, Yva = X_val, Y_val
        # Still need test set - use small split from training
        n_train = int(0.95 * len(Xtr))
        Xte, Yte = Xtr[n_train:], Ytr[n_train:]
        Xtr, Ytr = Xtr[:n_train], Ytr[:n_train]
    elif validate:
        # Split training data into train/val/test
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = split_train_val_test(X, Y, seed=seed)
    else:
        # No validation - use all for training, small test split
        n_train = int(0.95 * len(X))
        Xtr, Ytr = X[:n_train], Y[:n_train]
        Xte, Yte = X[n_train:], Y[n_train:]
        Xva, Yva = None, None

    # Build or load model
    if pretrained_model is not None:
        # Use loaded model and extract normalizers
        model = pretrained_model
        x_norm = model.get_layer('x_norm')
        y_norm = model.get_layer('y_norm')
        print("Using pretrained model with existing normalizers")
    else:
        # Normalization layers (fit on train only)
        x_norm = layers.Normalization(axis=-1, name="x_norm")
        y_norm = layers.Normalization(axis=-1, name="y_norm")
        x_norm.adapt(Xtr)
        y_norm.adapt(Ytr)

        # Build model that outputs normalized Y
        base = build_regressor(
            input_dim=Xtr.shape[1],
            output_dim=Ytr.shape[1],
            hidden=hidden,
            dropout=dropout,
            activation=activation,
            use_bias=use_bias,
            init_scale=init_scale,
            l2_reg=l2_reg,
        )

        inp = keras.Input(shape=(Xtr.shape[1],), dtype=tf.float32)
        x = x_norm(inp)
        yhat_norm = base(x)
        model = keras.Model(inp, yhat_norm, name=f"regressor_{task}")

    # Choose loss function
    if huber_delta is not None and np.isfinite(huber_delta):
        loss = keras.losses.Huber(delta=huber_delta)
    else:
        loss = "mse"

    # Loss/metrics in normalized Y space
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss,
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    # tf.data for speed
    train_ds = tf.data.Dataset.from_tensor_slices((Xtr, y_norm(Ytr))) \
        .shuffle(min(len(Xtr), 200_000), seed=seed, reshuffle_each_iteration=True) \
        .batch(batch) \
        .prefetch(tf.data.AUTOTUNE)

    # Setup callbacks
    cb = []
    
    if validate and Xva is not None:
        val_ds = tf.data.Dataset.from_tensor_slices((Xva, y_norm(Yva))) \
            .batch(val_batch) \
            .prefetch(tf.data.AUTOTUNE)
        
        cb.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=lr_reduce_factor, 
            patience=lr_reduce_patience, 
            min_lr=min_lr
        ))
        cb.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=early_stop_patience, 
            restore_best_weights=True
        ))
        cb.append(keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), 
            monitor="val_loss", 
            save_best_only=True
        ))
    else:
        val_ds = None
        cb.append(keras.callbacks.ReduceLROnPlateau(
            monitor="loss", 
            factor=lr_reduce_factor, 
            patience=lr_reduce_patience, 
            min_lr=min_lr
        ))
        cb.append(keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), 
            monitor="loss", 
            save_best_only=True
        ))

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb,
        verbose=verbosity
    )

    # Predict in normalized space, then unnormalize back to original scale
    Ypred_norm = model.predict(Xte, batch_size=8192, verbose=0)

    y_mean = y_norm.mean.numpy()
    y_std = np.sqrt(y_norm.variance.numpy())
    Ypred = y_mean + Ypred_norm * y_std

    # Get model metrics on normalized test targets
    test_metrics = model.evaluate(
        Xte, y_norm(Yte), batch_size=8192, verbose=0, return_dict=True
    )

    return model, hist, (Xte, Yte, Ypred), (x_norm, y_norm), test_metrics


# -----------------------
# Evaluation metrics
# -----------------------
def evaluate(Y_true, Y_pred):
    """
    Compute and print evaluation metrics.
    
    Parameters
    ----------
    Y_true : ndarray
        Ground truth values
    Y_pred : ndarray
        Predicted values
    """
    err = Y_pred - Y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))

    print("Per-component MAE (min/median/max):", 
          float(mae.min()), float(np.median(mae)), float(mae.max()))
    print("Per-component RMSE (min/median/max):", 
          float(rmse.min()), float(np.median(rmse)), float(rmse.max()))
    print("Global MAE:", float(np.mean(np.abs(err))))
    print("Global RMSE:", float(np.sqrt(np.mean(err**2))))


# -----------------------
# Plotting
# -----------------------
def plot_history(hist, save_path=None):
    """
    Plot training history.
    
    Parameters
    ----------
    hist : keras.callbacks.History
        Training history object
    save_path : str or Path, optional
        If provided, save figure to this path
    """
    # Loss plot
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE (normalized)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss plot to {save_path}")
    
    plt.show()
    plt.close()

    # MAE/RMSE plots if present
    for k in ["mae", "rmse"]:
        if k in hist.history:
            plt.figure(figsize=(6, 4))
            plt.plot(hist.history[k], label=f"train {k}")
            val_key = f"val_{k}"
            if val_key in hist.history:
                plt.plot(hist.history[val_key], label=f"val {k}")
            plt.yscale("log")
            plt.xlabel("epoch")
            plt.ylabel(f"{k} (normalized)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            if save_path:
                metric_path = save_path.parent / f"{save_path.stem}_{k}{save_path.suffix}"
                plt.savefig(metric_path, dpi=150, bbox_inches='tight')
                print(f"Saved {k} plot to {metric_path}")
            
            plt.show()
            plt.close()


def plot_true_vs_pred(Y_true, Y_pred, n_points=30000, seed=0, save_path=None):
    """
    Plot predicted vs true values.
    
    Parameters
    ----------
    Y_true : ndarray
        Ground truth values
    Y_pred : ndarray
        Predicted values
    n_points : int
        Number of points to plot (randomly sampled)
    seed : int
        Random seed for sampling
    save_path : str or Path, optional
        If provided, save figure to this path
    """
    rng = np.random.default_rng(seed)
    N = Y_true.shape[0]
    idx = rng.choice(N, min(N, n_points), replace=False)
    yt = Y_true[idx].reshape(-1)
    yp = Y_pred[idx].reshape(-1)

    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, s=2, alpha=0.25)
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Predicted vs True Values")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction plot to {save_path}")
    
    plt.show()
    plt.close()
