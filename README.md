# LearningG2
Setup:
```
python setup.py build_ext --inplace
```

To train the CY metric model run in the CLI in this repo:  
```
python3 -m models.cy_model
```
...then to train a model of the $G_2$ 3-form, first specify the run hyperparameters in the `hyperparameters/hps.yaml` file, then run:  
```
python3 -m run
```
...which outputs a saved model into the `runs` folder.  
