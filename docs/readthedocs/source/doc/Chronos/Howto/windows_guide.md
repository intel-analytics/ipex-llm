# Install Chronos on Windows

There are 2 ways to install Chronos on Windows: install using WSL2 and install on native Windows. With WSL2, all the feaatures of Chronos are available, while on native Windows, there are some limitations now.

## Install using WSL2
### Step 1: Install WSL2

Follow [BigDL Windows User guide](../../UserGuide/win.md) to install WSL2.


### Step 2: Install Chronos

Follow the [Chronos Installation guide](../Overview/chronos.md#install) to install Chronos.

## Install on native Windows

### Step1: Install conda

We recommend using conda to manage the Chronos python environment, for more information on install conda on Windows, you can refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

When conda is successfully installed, open the Anaconda Powershell Prompt, then you can create a conda environment using the following command:

```
# create a conda environment for chronos
conda create -n my_env python=3.7 setuptools=58.0.4  # you could change my_env to any name you want
```

### Step2: Install Chronos from PyPI
You can simply install Chronos from PyPI using the following command:

```
# activate your conda environment
conda activate my_env

# install Chronos nightly build version
pip install --pre --upgrade bigdl-chronos[pytorch]
```

You can use the [install panel](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/install.html#install-using-conda) to select the proper install options based on your need, but there are some limitations now:

- bigdl-chronos[distributed] is not supported.

- `Prophet Forecaster` will raise RuntimeError on Windows, so the related feature is not supported.

- `intel_extension_for_pytorch (ipex)` is unavailable for Windows now, so the related feature is not supported.
