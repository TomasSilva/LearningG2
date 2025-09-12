## Setup:

We recommend the use of a virtual environment. Below demonstrated with conda:
```bash
# Create new conda environment with Python 3.11
conda create -n g2_ml python=3.11 -y
conda activate g2_ml

# Install packages from requirements
pip install -r environment/requirements.txt

# Set up Jupyter kernel
python -m ipykernel install --user --name g2_ml --display-name "G2 ML Environment"
```
The environment includes a dedicated Jupyter kernel named "G2 ML Environment", and compatibility with [cymetric](https://github.com/pythoncymetric/cymetric) for learning of the metric on the base Calab-Yau manifold. Download the package and place in the same parent directory as this repository. Then run the below check to ensure it is correctly accessible by the package.
```bash
# Check cymetric is in parent directory alongside LearningG2/
python -c "
import os
if os.path.exists('../cymetric/cymetric/__init__.py'):
    print('✓ Cymetric package found in correct location')
else:
    print('✗ Cymetric package not found in parent directory')
    print('   Ensure cymetric/ is in the same directory as github/')
    print('   Clone from: https://github.com/pythoncymetric/cymetric')
"
```

Finally, build the cython functionality:
```bash
python3 environment/setup.py build_ext --inplace
```

