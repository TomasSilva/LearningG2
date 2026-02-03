# Environment Setup

This repository uses [Git LFS](https://git-lfs.com/). Before cloning:
```bash
git lfs install
```

## Quick Setup

```bash
# 1. Create conda environment
conda create -n g2_ml python=3.11 -y
conda activate g2_ml

# 2. Install packages
python -m pip install -r environment/requirements.txt

# 3. Install Jupyter kernel
python -m ipykernel install --user --name g2_ml --display-name "G2 ML Environment"

# 4. Install cymetric (in parent directory)
cd ..
git clone https://github.com/ruehlef/cymetric.git
cd LearningG2

# 5. Build Cython extensions
cd environment
python setup.py build_ext --inplace
cd ..
```

## Important Notes

- **Always use `python -m pip`** (not just `pip`) to ensure packages install to the conda environment
- Activate the environment before working: `conda activate g2_ml`
- The "G2 ML Environment" kernel will be available in Jupyter notebooks
- Cymetric must be in the parent directory

