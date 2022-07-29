# Windows User Guide

## Step 1: Install WSL2

For Windows, we recommend using Windows Subsystem for Linux 2 (WSL2) to run BigDL-Chronos. 

To install WSL2, you can now install everything you need by entering this command in an administrator PowerShell or Windows Command Prompt and then restarting your machine.  

```powershell
wsl --install
```

By default, this command should install the latest required components for WSL2 and install Ubuntu as default distribution for you. 

To run this command, you must be running Windows 10 version 2004 and higher (Build 19041 and higher) or Windows 11. If you're running an older build, or just prefer not to use the install command and would like step-by-step directions, see WSL manual installation steps for older versions.

To learn more about installation of WSL2, please Follow [this guide](https://docs.microsoft.com/en-us/windows/wsl/install-win10). 

## Step 2: Install conda in WSL2

 Start a new WSL2 window and setup the user information. Then download and install the conda. 
 
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh
chmod +x Miniconda3-4.5.4-Linux-x86_64.sh
./Miniconda3-4.5.4-Linux-x86_64.sh
```

## Step 3: Create a BigDL-Chronos env 

Use conda to create a new environment. For example, use `bigdl-chronos` as the new environemnt name: 

```bash
conda create -n bigdl-chronos python=3.7 setuptools=58.0.4
conda activate bigdl-chronos
```


## Step 4: Install BigDL Chronos from Pypi

You can install BigDL chronos from Pypi with `pip`. Please run:

```
pip install bigdl-chronos
```

You may also install `bigdl-chronos` with target `[all]` to install the additional dependencies for _Chronos_. This will enable distributed tuning with AutoTS.
```bash
# stable version
pip install bigdl-chronos[all]
# nightly built version
pip install --pre --upgrade bigdl-chronos[all]
# set env variables for your conda environment
source bigdl-nano-init
```
