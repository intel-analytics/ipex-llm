# Installation guide

This project showcases a [video stylization](https://github.com/rnwzd/FSPBT-Image-Translation) use case on Intel laptop/desktop, and uses [BigDL-Nano](https://github.com/intel-analytics/bigdl#nano) to accelerate AI on Intel CPU. Before start, you have to finish some necessary installation.

## Nano installation

### For windows User

If you are a Windows user, we recommend using Windows Subsystem for Linux 2 (WSL2) to run BigDL-Nano. Please refer to [Nano Windows install guide](https://bigdl.readthedocs.io/en/latest/doc/Nano/Howto/windows_guide.html) for instructions.

### For Linux User

If you are a Linux user, We recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to prepare the environment. You can install bigdl-nano along with some dependencies specific to PyTorch using the following commands:
```bash
conda create -n env python=3.7 setuptools=58.0.4  # "env" is conda environment name, you can use any name you like.
conda activate env
pip install --pre --upgrade bigdl-nano[pytorch]
pip install jupyter
```
After installing bigdl-nano, you can run the following command to setup a few environment variables:

`source bigdl-nano-init`

The bigdl-nano-init scripts will export a few environment variables according to your hardware to maximize performance.

## Install other dependencies

Install [Intel Neural Compressor (INC)](https://github.com/intel/neural-compressor), which is used by BigDL-Nano:

`pip install neural-compressor==1.13.1`

Then install `ffmpeg` to deal with video input:

```bash
sudo apt install ffmpeg # for Linux
```

## Launch jupyter notebook

After installation, just launch the jupyter notebook by `jupyter notebook VideoStylization.ipynb` .
