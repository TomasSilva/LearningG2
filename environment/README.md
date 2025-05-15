## Setup:

We recommend the use of a virtual environment. For example, with conda:
```
conda create -n g2_ml
```

Then, activate conda and install the required packages:
```
conda activate g2_ml && pip install -r environment/requirements.txt
```
Follow the instructions at the [cymetric](https://github.com/pythoncymetric/cymetric) repository to set up the functionality for learning the Calabi-Yau metric used in the Calabi-Yau link construction in this virtual environment.

Finally, build the cython functionality:
```
python3 environment/setup.py build_ext --inplace
```

