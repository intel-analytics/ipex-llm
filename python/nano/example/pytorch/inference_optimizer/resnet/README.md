# Bigdl-nano InferenceOptimizer example on Cat vs. Dog dataset

This example illustrates how to apply InferenceOptimizer to quickly find acceleration method with the minimum inference latency under specific restrictions or without restrictions for a trained model. 
For the sake of this example, we first train the proposed network(by default, a ResNet18 is used) on the [cats and dogs dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip), which consists both [frozen and unfrozen stages](https://github.com/PyTorchLightning/pytorch-lightning/blob/495812878dfe2e31ec2143c071127990afbb082b/pl_examples/domain_templates/computer_vision_fine_tuning.py#L21-L35). Then, by calling `optimize()`, we can obtain all available accelaration combinations provided by BigDL-Nano for inference. By calling `get_best_mdoel()` , we could get an accelerated model whose inference is 8x times faster.


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
+++++++++++++++++++++++++
Complete.
```

## Prepare Dataset
By default the dataset will be auto-downloaded.
You could access [cats and dogs dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip) for a view of the whole dataset.

## Run example
You can run this example with command line:

```bash
python inference_pipeline.py
```

## Results
Then you may find the result for inference as follows. The actual number may vary based on your hardware.
```
==========================Optimization Results==========================
 -------------------------------- ---------------------- -------------- ----------------------
|             method             |        status        | latency(ms)  |     metric value     |
 -------------------------------- ---------------------- -------------- ----------------------
|            original            |      successful      |    45.145    |        0.975         |
|              bf16              |      successful      |    27.549    |        0.975         |
|          static_int8           |      successful      |    11.339    |        0.975         |
|         jit_fp32_ipex          |      successful      |    40.618    |        0.975*        |
|  jit_fp32_ipex_channels_last   |      successful      |    19.247    |        0.975*        |
|         jit_bf16_ipex          |      successful      |    10.149    |        0.975         |
|  jit_bf16_ipex_channels_last   |      successful      |    9.782     |        0.975         |
|         openvino_fp32          |      successful      |    22.721    |        0.975*        |
|         openvino_int8          |      successful      |    5.846     |        0.962         |
|        onnxruntime_fp32        |      successful      |    20.838    |        0.975*        |
|    onnxruntime_int8_qlinear    |      successful      |    7.123     |        0.981         |
 -------------------------------- ---------------------- -------------- ----------------------
* means we assume the metric value of the traced model does not change, so we don't recompute metric value to save time.
Optimization cost 60.8s in total.
===========================Stop Optimization===========================
When accuracy drop less than 5%, the model with minimal latency is:  openvino + int8
```