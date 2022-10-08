# Windows User Guide
## Prerequisite


### Confirm your windows version

To use BigDL on Windows, we recommend using [Windows Subsystem for Linux 2 (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/about#what-is-wsl-2).

The recommended windows version is Windows 10 version 2004 or higher (Build 19041 and higher), or Windows 11.



### Install WSL2

To install WSL2, simply open a PowerShell or Windows Command Prompt as **administrator** and enter the below command. Restart your machine and wait till WSL2 is successfully installed.

```powershell
wsl --install
```

By default, the command installs the latest required components for WSL2 and install **Ubuntu** as default distribution.

```eval_rst
.. note::
    This command requires Windows 10 version 2004 or higher. If you're using older versions of Window, or prefer not using command line for installation, or upgrading from WSL1 to WSL2, please refer to `WSL installation guide <https://learn.microsoft.com/en-us/windows/wsl/install>`_.
```

## Installation Guide

You can treat WSL2 shell window as a normal Linux shell and run commands in it. If you're using WSL2 shell for the first time, it may require you to set up some user information.


### Install Conda

Conda is the recommend way to manage the BigDL environment. Download and install conda using below commands.

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
./Miniconda3-4.5.4-Linux-x86_64.sh
```

```eval_rst
.. note::
    On WSL, you need to use a conda Linux version intead of a Windows version. For more available conda versions, refer to `conda install <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_, or `miniconda install <https://docs.conda.io/en/main/miniconda.html>`_.
```
### Install BigDL

After installing conda, create and activate an environment for bigdl.

```bash
conda create -n bigdl-env
conda activate bigdl-env
```

Then install BigDL, or BigDL-Nano, BigDL-Chronos, etc.

```bash
pip install bigdl
```

```eval_rst
.. card::

    **Related Readings**
    ^^^
    * `BigDL Installation Guide <../UserGuide/python>`_
    * `Nano Installation Guide <../Nano/Overview/nano.html#install>`_
    * `Chronos Installation Guide <../Chronos/Overview/chronos.html#install>`_
```

### Setup Jupyter Notebook Environment

Install JupyterLab with pip:

```bash
pip install jupyterlab
```
Once installed, launch JupyterLab with:

```bash
jupyter lab --no-browser
```
Note that the default workspace of jupyter is located at the directory where you ran this command.

Then you can copy and paste the full URL shown on the terminal to open the Jupyter GUI.


## Tips and Known Issues

### `ImportError: libgomp.so.1: cannot open shared object file: No such file or directory`

This error may appear when you try to import torch. This is caused by Ubuntu 14.04 or later not installing libgomp1 by default. Just install libgomp1 to resolve it:

```bash
sudo apt-get install libgomp1
```

### Slow PyTorch training with BF16

Using BFloat16 mixed precision in PyTorch or PyTorch-Lightning training may be much slower than FP32.


### `ERROR: Could not build wheels for pycocotools, which is required to install pyproject.toml-based projects`

pycocotools is a dependency of Intel neural-compressor which is used for inference quantization in BigDL-Nano. This error is usually caused by GCC library not installed in system.  Just install gcc to resolve it:

```bash
sudo apt-get install gcc
```

### `ValueError: After taking into account object store and redis memory usage, the amount of memory on this node available for tasks and actors is less than -75% of total.`

When running ray applications, you need to set the `memory` and `object_store_memory` properly according to your system memory capacity. This error indicates you have used too large memory configurations and you need to decrease them. For example on a laptop with 8G memory, you may set the memory configurations as below:

```bash
python yoloV3.py --memory 2g --object_store_memory 1g
```
