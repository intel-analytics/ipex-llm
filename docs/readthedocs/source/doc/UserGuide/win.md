# Windows User Guide
## Prerequisite


### Confirm your windows version

To use BigDL on Windows, we recommend using [Windows Subsystem for Linux 2 (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/about#what-is-wsl-2). The recommended Windows versions are Windows 10 version 2004 or higher (Build 19041 and higher), or Windows 11.


### Install WSL2

To install WSL2, simply open a PowerShell or Windows Command Prompt as **administrator** and enter the below command. Restart your machine and wait until WSL2 is successfully installed.

```powershell
wsl --install
```

```eval_rst
.. note::
    By default, the above command installs the latest required components for WSL2 and **Ubuntu** as default Linux distribution, and it requires Windows 10 version 2004 or higher. If you're using older versions of Windows or need customization, please refer to `WSL installation guide <https://learn.microsoft.com/en-us/windows/wsl/install>`_.
```

## Installation Guide

You can treat WSL2 shell as a normal Linux shell and run normal bash commands in it. If you're using WSL2 shell for the first time, it may require you to set up some user information. Using WSL2, you can install BigDL the same way as you do on a Linux system.


### Install Conda

Conda is the recommend way to manage the BigDL environment. Download and install conda using below commands.

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
./Miniconda3-4.5.4-Linux-x86_64.sh
```

```eval_rst
.. note::
    On WSL2, you need to use a Linux version of Conda intead of a Windows version. For other available conda versions, refer to `conda install <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_, or `miniconda install <https://docs.conda.io/en/main/miniconda.html>`_.
```

### Install BigDL

After installing conda, use conda to create and activate an environment for bigdl.

```bash
conda create -n bigdl-env
conda activate bigdl-env
```

Then install BigDL as a whole, or specific bigdl library the same way as you do on a Linux system. For example,

```bash
pip install bigdl
```

```eval_rst
.. card::

    **Related Readings**
    ^^^
    * `BigDL Installation Guide <./python.html>`_
    * `Nano Installation Guide <../Nano/Overview/install.html>`_
    * `Chronos Installation Guide <../Chronos/Overview/install.html>`_
```

### Setup Jupyter Notebook Environment

Fist, install JupyterLab using pip:

```bash
pip install jupyterlab
```

Then start JupyterLab using:

```bash
jupyter lab
```

```eval_rst
.. note::
    Once you started Juypterlab, it will open automatically in your browser. If it does not open automatically, you can manually enter the notebook server’s URL into the browser (The URL is shown on the terminal where you run the command). The default workspace of jupyter is located at the directory where you start the jupyterlab. For more information about JupyterLab installation and usage, refer to `JupyterLab User Guide <https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html#>`_.
```

## Tips and Known Issues

### 1. ImportError: libgomp.so.1: cannot open shared object file: No such file or directory

This error may appear when you try to import torch. This is caused by Ubuntu 14.04 or later not installing libgomp1 by default. Just install libgomp1 to resolve it:

```bash
sudo apt-get install libgomp1
```

### 2. ERROR: Could not build wheels for pycocotools, which is required to install pyproject.toml-based projects

pycocotools is a dependency of Intel neural-compressor which is used for inference quantization in BigDL-Nano. This error is usually caused by GCC library not installed in system.  Just install gcc to resolve it:

```bash
sudo apt-get install gcc
```

### 3. ValueError: After taking into account object store and redis memory usage, the amount of memory on this node available for tasks and actors is less than -75% of total.

When running ray applications, you need to set the `memory` and `object_store_memory` properly according to your system memory capacity. This error indicates you have used too large memory configurations and you need to decrease them. For example on a laptop with 8G memory, you may set the memory configurations as below:

```bash
python yoloV3.py --memory 2g --object_store_memory 1g
```
