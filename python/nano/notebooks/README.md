---
## Bigdl-nano Resnet example on CIFAR10 dataset
This example illustrates how to apply bigdl-nano optimizations on a image recognition case based on pytorch-lightning framework. The basic image recognition module is implemented with Lightning and trained on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) image recognition Benchmark dataset.
### Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment.
```bash
conda create -n nano python==3.7 # "nano" is conda environment name, you can use any name you like.
conda activate nano
```
#### Bigdl-nano
```bash
pip install bigdl-nano[pytorch]
```
Initialize environment variables with script `bigdl-nano-init` installed with bigdl-nano.
```bash
source bigdl-nano-init
```
You may find environment variables set like follows:
```
Setting OMP_NUM_THREADS...
Setting OMP_NUM_THREADS specified for pytorch...
Setting KMP_AFFINITY...
Setting KMP_BLOCKTIME...
Setting MALLOC_CONF...
+++++ Env Variables +++++
LD_PRELOAD=./../lib/libjemalloc.so
MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1
OMP_NUM_THREADS=48
KMP_AFFINITY=granularity=fine,compact,1,0
KMP_BLOCKTIME=1
TF_ENABLE_ONEDNN_OPTS=
+++++++++++++++++++++++++
Complete.
```
#### Ipex
Install ipex if you want to use ipex as as accelerator for nano trainer.
- Install IPEX via wheel file

```bash
python -m pip install torch_ipex==1.9.0 -f https://software.intel.com/ipex-whl-stable
```

- Install IPEX by compiling from source

```bash
git clone --recursive https://github.com/intel/intel-extension-for-pytorch
cd intel-extension-for-pytorch

# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# run setup.py to compile IPEX and install the binaries
python setup.py install
```
