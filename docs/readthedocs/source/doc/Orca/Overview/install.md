# Installation


## To use Distributed Data processing, training, and/or inference
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment.
```bash
conda create -n py37 python=3.7  # "py37" is conda environment name, you can use any name you like.
conda activate py37
pip install bigdl-orca
```

You can install bigdl-orca nightly build version using
```bash
pip install --pre --upgrade bigdl-orca
```

## To use RayOnSpark

There're some additional dependencies required for running [RayOnSpark](ray.md). Use extra key `[ray]` to install.

```bash
pip install bigdl-orca[ray]
```

or to install nightly build, use
```bash
pip install --pre --upgrade bigdl-orca[ray]
```

## To use Orca AutoML

There're some additional dependencies required for Orca AutoML support. Use extra key `[automl]` to install.

```bash
pip install bigdl-orca[automl]
````


_Note that with extra key of [automl], `pip` will automatically install the additional dependencies for distributed hyper-parameter tuning,
including `ray[tune]==1.9.2`, `scikit-learn`, `tensorboard`, `xgboost`._

To use [Pytorch Estimator](#pytorch-autoestimator), you need to install Pytorch with `pip install torch==1.8.1`.

To use [TensorFlow/Keras AutoEstimator](#tensorflow-keras-autoestimator), you need to install Tensorflow with `pip install tensorflow==1.15.0`.

