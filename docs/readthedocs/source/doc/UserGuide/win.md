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

By default, the command installs the latest required components for WSL2 and install Ubuntu as default distribution for you.

```eval_rst
.. note::
    This command requires Windows 10 version 2004 or higher. If you're using older versions of Window, or prefer not using command line for installation, or upgrading from WSL1 to WSL2, please refer to [WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).
```

## Python

### Install Conda

Fist, start a new WSL2 shell. If you're using WSL2 shell for the first time, it may require you to set up some user information.

Then, download and install conda (it is the recommend way to manage the BigDL environment). On WSL, you need to use a Linux version as shown below. For more available conda versions, refer to [conda install](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), or [miniconda install](https://docs.conda.io/en/main/miniconda.html).


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

To install nightly builds, et to for details.

```eval_rst
.. card::

    **Related Readings**
    ^^^
    * `BigDL Installation Guide <../PyTorchLightning/accelerate_pytorch_lightning_training_multi_instance.html>`_
    * `Nano Installation Guide <../Nano/Overview/nano.html>`_
```

### Setup Jupyter Notebook Environment


## Developer Guide


## Tips and Known Issues
