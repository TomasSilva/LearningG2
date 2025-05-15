## Setup:

We recommend the use of a virtual environment. For example, with conda:
```
conda create -n g2_ml
```

Then, activate conda and install the required packages:
```
conda activate g2_ml && pip install -r environment/requirements.txt
```
...and build the cython functionality:

```
python3 environment/setup.py build_ext --inplace
```

