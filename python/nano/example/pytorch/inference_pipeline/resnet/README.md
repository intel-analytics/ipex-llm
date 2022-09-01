# Bigdl-nano InferenceOptimizer example on Cat vs. Dog dataset

This example illustrates how to apply InferenceOptimizer to quickly find acceleration method with the minimum inference latency under specific restrictions or without restrictions for a trained model. 
For the sake of this example, we first train the proposed network(by default, a ResNet18 is used) on the [cats and dogs dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip), which consists both [frozen and unfrozen stages](https://github.com/PyTorchLightning/pytorch-lightning/blob/495812878dfe2e31ec2143c071127990afbb082b/pl_examples/domain_templates/computer_vision_fine_tuning.py#L21-L35). Then, by calling `optimize()`, we can obtain all available accelaration combinations provided by BigDL-Nano for inference. By calling `get_best_mdoel()` , we could get an accelerated model whose inference is 7.8x times faster.


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

It will take about 4 minutes to run inference optimization. Then you may find the result for inference as follows:
```
accleration option: original, latency: 54.2669ms, accuracy: 0.9937
accleration option: fp32_ipex, latency: 40.3075ms, accuracy: 0.9937
accleration option: bf16_ipex, latency: 115.6182ms, accuracy: 0.9937
accleration option: int8, latency: 14.4857ms, accuracy: 0.4750
accleration option: jit_fp32, latency: 39.3361ms, accuracy: 0.9937
accleration option: jit_fp32_ipex, latency: 39.2949ms, accuracy: 0.9937
accleration option: jit_fp32_ipex_clast, latency: 24.5715ms, accuracy: 0.9937
accleration option: openvino_fp32, latency: 14.5771ms, accuracy: 0.9937
accleration option: openvino_int8, latency: 7.2186ms, accuracy: 0.9937
accleration option: onnxruntime_fp32, latency: 44.3872ms, accuracy: 0.9937
accleration option: onnxruntime_int8_qlinear, latency: 10.1866ms, accuracy: 0.9937
accleration option: onnxruntime_int8_integer, latency: 18.8731ms, accuracy: 0.9875
When accelerator is onnxruntime, the model with minimal latency is:  inc + onnxruntime + qlinear 
When accuracy drop less than 5%, the model with minimal latency is:  openvino + pot 
The model with minimal latency is:  openvino + pot 
```
