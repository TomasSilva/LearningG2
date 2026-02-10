# Learning G2-Structure 3-forms

Neural network approach to learning G2 structures on Calabi-Yau Links. The package trains models to predict the defining 3-form $\varphi$ and associated metric on the total space of the Link, enabling efficient computation of G2 geometry.

## Setup

Set up the Python environment following the instructions in [`environment/README.md`](./environment/README.md).

## Workflow

**Interactive Alternative:** If you prefer not to use the command line, all scripts can be run from the Jupyter notebook [`run_interactive.ipynb`](./run_interactive.ipynb). Alternatively, run the entire pipeline automatically using the bash script [`run_all.sh`](./run_all.sh).

### 1. Train CY Metric Model
Train a neural network to learn the Ricci-flat Kähler metric on the Calabi-Yau threefold using the [cymetric](https://github.com/pythoncymetric/cymetric) package:

```bash
python run_cy.py --n-points 100000
```

This generates training data for the quintic CY threefold and trains a model that outputs the Hermitian metric components. The trained model is saved to `models/cy_models/cy_metric_model_run{N}.keras`.

### 2. Generate G2 Sample Data
Create training data for G2 structure learning by sampling points on the CY and computing the analytical G2 forms:

```bash
python sampling.py
```

This produces datasets in `samples/link_data/` containing:
- Base points and link points on the CY
- Analytically computed $\varphi$ (3-form) and $\psi$ (4-form)  
- G2 metrics and local Reeb vector (eta) components
- CY base patch coordinate indices for each point

### 3. Train G2 Models
Train neural networks to predict the 3-form and G2 metric. Edit `hyperparameters/hps.yaml` to configure training parameters, then run:

```bash
# Train 3-form predictor
python run_g2.py --task 3form

# Train G2 metric predictor
python run_g2.py --task metric
```

Models are saved to `models/link_models/3form_run{N}.keras` and `metric_run{N}.keras`. The hyperparameters file controls:
- Network architecture (layers, units, activation)
- Training parameters (epochs, batch size, learning rate)
- Data splits and validation settings

### 4. Validate Models
Check that learned models satisfy the G2 structure identities: $\varphi \wedge \psi = 7·Vol(g)$, $d\psi = 0$, and $d\varphi = \omega^2$.
 
```bash
# Check Kählerity of learned CY metric (dω = 0)
python analysis/cy_kahlerity.py --cy-run-number 1

# Check G2 identities using analytical construction
python analysis/g2_identities_analytic.py --cy-run-number 1

# Check G2 identities using trained model predictions
# Use --psi-method star (Hodge star) or --psi-method model (4form model)
python analysis/g2_identities_model_v2.py --cy-run-number 1 --g2-run-number 1 --psi-method star
```

All scripts output statistics and save diagnostic plots to `plots/` directory.

## Additional Functionality

- The `run_g2.py` script also accepts '4form' as a task argument, which will train an NN on the 4-form $\psi$. This can then be used in the `g2_identities_model.py` checks (instead of building $\psi$ as $\ast\varphi$).  
- The G2 identities checks can be performed at the level of the data (without the trained NN models) using the `g2_identities_analytic.py` script.  

### Run Numbering System
All training scripts use an automatic run numbering system to organize experiments:

- Each training run is assigned an incrementing integer: run 1, run 2, run 3, etc.
- Run numbers are detected automatically from existing model files in the relevant save directory
- Scripts can load specific runs using `--cy-run-number` or `--g2-run-number` argumentsmv
- Without specifying a run number, scripts auto-detect and use the most recent run
- This enables easy experiment tracking and comparison without manual file management

## References

Based on numerical exterior derivative methods from [arXiv:2510.00999](https://arxiv.org/abs/2510.00999).

## BibTeX Citation

```
raise NotImplementedError("Paper yet to be published.")
```
