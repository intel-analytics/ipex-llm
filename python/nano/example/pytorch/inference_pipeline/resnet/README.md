# Bigdl-nano InferenceOptimizer example on Cat vs. Dog dataset

This example illustrates how to apply InferenceOptimizer to quickly find acceleration method with the minimum inference latency under specific restrictions or without restrictions for a trained model. 
For the sake of this example, we first train the proposed network(by default, a ResNet18 is used) on the [cats and dogs dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip), which consists both [frozen and unfrozen stages](https://github.com/PyTorchLightning/pytorch-lightning/blob/495812878dfe2e31ec2143c071127990afbb082b/pl_examples/domain_templates/computer_vision_fine_tuning.py#L21-L35). Then, by calling `optimize()`, we can obtain all available accelaration combinations provided by BigDL-Nano for inference. By calling `get_best_mdoel()` , we could get an accelerated model whose inference is 5.5x times faster.


## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment.
**Note**: during your installation, there may be some warnings or errors about version, just ignore them.
```
conda create -n nano python=3.7  # "nano" is conda environment name, you can use any name you like.
conda activate nano
pip install jsonargparse[signatures]
pip install --pre --upgrade bigdl-nano[pytorch]

# bf16 is available only on torch1.12
pip install torch==1.12.0 torchvision --extra-index-url https://download.pytorch.org/whl/cpu 
# Necessary packages for inference accelaration
pip install --upgrade intel-extension-for-pytorch
pip install onnx==1.12.0 onnxruntime==1.12.1 onnxruntime-extensions
pip install openvino-dev
pip install neural-compressor==1.12
pip install --upgrade numpy==1.21.6
```
Initialize environment variables with script `bigdl-nano-init` installed with bigdl-nano.
```
source bigdl-nano-init
unset KMP_AFFINITY
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
OMP_NUM_THREADS=112
KMP_AFFINITY=granularity=fine,compact,1,0
KMP_BLOCKTIME=1
TF_ENABLE_ONEDNN_OPTS=
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
It will take about 1 minute to run inference optimization. Then you may find the result for inference as follows:
```
==========================Optimization Results==========================
 -------------------------------- ---------------------- -------------- ------------
|             method             |        status        | latency(ms)  |  accuracy  |
 -------------------------------- ---------------------- -------------- ------------
|            original            |      successful      |    43.447    |   0.994    |
|           fp32_ipex            |      successful      |    32.827    |   0.994    |
|              bf16              |   fail to forward    |     None     |    None    |
|           bf16_ipex            |        pruned        |   201.702    |    None    |
|              int8              |      successful      |    10.992    |   0.994    |
|            jit_fp32            |      successful      |    36.741    |   0.994    |
|         jit_fp32_ipex          |      successful      |    33.293    |   0.994    |
|  jit_fp32_ipex_channels_last   |      successful      |    19.523    |   0.994    |
|         openvino_fp32          |      successful      |    10.51     |   0.994    |
|         openvino_int8          |      successful      |    6.637     |   0.994    |
|        onnxruntime_fp32        |      successful      |    20.55     |   0.994    |
|    onnxruntime_int8_qlinear    |      successful      |     8.15     |   0.994    |
|    onnxruntime_int8_integer    |   fail to convert    |     None     |    None    |
 -------------------------------- ---------------------- -------------- ------------

Optimization cost 64.3s at all.
===========================Stop Optimization===========================
When accelerator is onnxruntime, the model with minimal latency is:  inc + onnxruntime + qlinear 
When accuracy drop less than 5%, the model with minimal latency is:  openvino + pot 
The model with minimal latency is:  openvino + pot 
```