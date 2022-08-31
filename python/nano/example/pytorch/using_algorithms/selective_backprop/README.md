# Using Selective_Backprop Algorithm in BigDL-nano 

This example illustrates how to use [selective_backprop](https://arxiv.org/abs/1910.00762) in training.

For the sake of this example, we train the proposed network(by default, a ResNet18 is used) on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)


## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment.
```
conda create -n nano python=3.7  # "nano" is conda environment name, you can use any name you like.
conda activate nano

pip install bigdl-nano[pytorch]
```
Initialize environment variables with script `bigdl-nano-init` installed with bigdl-nano.
```
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

## Prepare Dataset
By default the dataset will be auto-downloaded.
You could access [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) for a view of the whole dataset.

## Run example
You can run this example with command line:

```bash
python selective_backprop.py
```