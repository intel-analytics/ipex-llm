---
## Running BigDL-Nano on Jupyter Notebook
These examples illustrate how to apply bigdl-nano optimizations on machine learning models based on a variety of frameworks. The examples are presented as notebooks, so you need to prepare a general BigDL-Nano and Jupyter Notebook environment.
### Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment.
```bash
conda create -n nano python==3.7 # "nano" is conda environment name, you can use any name you like.
conda activate nano
```
#### BigDL-Nano
Installation
```bash
pip install bigdl-nano[pytorch]
```
Initialize environment variables
- **Note**
	*You need to perform this before starting Jupyter Notebook.*
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
#### Jupyter Notebook
Install jupyter notebook
```bash
pip install jupyter
```
Start jupyter notebook
```bash
jupyter notebook
```