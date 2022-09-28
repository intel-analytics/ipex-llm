# Windows User Guide
## Prerequisite


### Confirm your windows version

To use BigDL on Windows, we recommend using [Windows Subsystem for Linux 2 (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/about#what-is-wsl-2).

The recommended windows version is Windows 10 version 2004 or higher (Build 19041 and higher), or Windows 11.



### Install WSL2

To install WSL2, simply open a PowerShell or Windows Command Prompt as **administrator** and enter the below command. Remember to restart your machine after the command completes.

```powershell
wsl --install
```

By default, the command installs the latest required components for WSL2 and install **Ubuntu** as default distribution.

```eval_rst
.. note::
    This command requires Windows 10 version 2004 or higher. If you're using older versions of Window, or prefer not using command line for installation, or upgrading from WSL1 to WSL2, please refer to `WSL installation guide <https://learn.microsoft.com/en-us/windows/wsl/install>`_.
```

## Installation Guide

You can treat WSL2 shell window as a nomral Linux shell and follow BigDL Linux guides to install BigDL.


### Install Conda

Fist, start a new WSL2 shell. If you're using WSL2 shell for the first time, it may require you to set up some user information.

Then, download and install conda (conda is the recommend way to manage the BigDL environment). On WSL, you need to use a conda Linux version as shown in the example command below. For more available conda versions, refer to [conda install](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), or [miniconda install](https://docs.conda.io/en/main/miniconda.html).


```bash
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
./Miniconda3-4.5.4-Linux-x86_64.sh
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
    * `Chronos Installation Guide` <../Chronos/Overview/chronos.html#install>
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

Then you can copy and paste the full URL listed in the terminal to open the GUI interface.


## Developer Guide


## Tips and Known Issues

### ImportError: libgomp.so.1: cannot open shared object file: No such file or directory

This error may appear when you try to import torch, which is due to Ubuntu 14.04 or later does not install libgomp1 by default. Fix it by running:

```bash
sudo apt-get install libgomp1
```

### Extremely slow training when BF16 is on

Using BFloat16 mixed precision in PyTorch or PyTorch-Lightning training may result in around 150x slower per step than the traditional way.

### WARNING:root:avx512 disabled, fall back to non-ipex mode.

The IPEX extension delivers optimizations for PyTorch on Intel hardware which has AVX-512 Vector Neural Network Instructions. Devices whose CPU does not support AVX-512 will fall back to non-ipex mode.

### ERROR: Could not build wheels for pycocotools, which is required to install pyproject.toml-based projects

This error is usually caused by lacking GCC library to build pycocotools, which is a dependency of neural-compressor for quantization inference, which can be solved by:

```bash
sudo apt-get install gcc
```
