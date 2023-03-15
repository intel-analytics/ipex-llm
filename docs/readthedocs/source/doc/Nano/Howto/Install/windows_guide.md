# Install BigDL-Nano on Windows

## Step 1: Install WSL2


Follow [BigDL Windows User guide](../../UserGuide/win.md) to install WSL2.


## Step 2: Install conda in WSL2

It is highly recommended to use conda to manage the python environment for BigDL-Nano. Follow [BigDL Windows User Guide/Conda Install](../../UserGuide/win.md#install-conda) to install conda.

## Step 3: Create a BigDL-Nano env

Use conda to create a new environment. For example, use `bigdl-nano` as the new environment name:

```bash
conda create -n bigdl-nano
conda activate bigdl-nano
```


## Step 4: Install BigDL-Nano from Pypi

You can install BigDL-Nano from Pypi with `pip`. Specifically, for PyTorch extensions, please run:

```
pip install bigdl-nano[pytorch]
source bigdl-nano-init
```

For Tensorflow:

```
pip install bigdl-nano[tensorflow]
source bigdl-nano-init
```