# Bigdl-nano InferenceOptimizer example about how to export the optimized model to standard format

This example illustrates how to export the optimized model which is found by InferenceOptimizer to a standard format(PyTorch, ONNX, OpenVINO, etc.).

For the sake of this example, we take ResNet50 for an example. First, by calling `optimize()` and `get_best_model()`, we could obtain an accelerated model with the minimum latency. Then, by calling `save()`, we could export the optimized model to a standard format.


## Prepare the environment
We recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to prepare the environment.
**Note**: during your installation, there may be some warnings or errors about version, just ignore them.
```bash
conda create -n nano python=3.7 setuptools=58.0.4  # "nano" is conda environment name, you can use any name you like.
conda activate nano
pip install --pre --upgrade bigdl-nano[pytorch,inference]  # install the nightly-bulit version
```
Initialize environment variables with script `bigdl-nano-init` installed with bigdl-nano.
```
source bigdl-nano-init
``` 
You may find environment variables set like follows:
```
OpenMP library found...
Setting OMP_NUM_THREADS...
Setting OMP_NUM_THREADS specified for pytorch...
Setting KMP_AFFINITY...
Setting KMP_BLOCKTIME...
Setting MALLOC_CONF...
Setting LD_PRELOAD...
nano_vars.sh already exists
+++++ Env Variables +++++
LD_PRELOAD=/opt/anaconda3/envs/nano/bin/../lib/libiomp5.so /opt/anaconda3/envs/nano/lib/python3.7/site-packages/bigdl/nano//libs/libtcmalloc.so
MALLOC_CONF=
OMP_NUM_THREADS=112
KMP_AFFINITY=granularity=fine
KMP_BLOCKTIME=1
TF_ENABLE_ONEDNN_OPTS=1
ENABLE_TF_OPTS=1
NANO_TF_INTER_OP=1
+++++++++++++++++++++++++
Complete.
```

## Run example
You can run this example with command line:

```bash
python export_example.py
```
