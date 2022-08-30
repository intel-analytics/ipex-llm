# Bigdl-nano InferenceOptimizer example on Cat vs. Dog dataset

This example illustrates how to apply InferenceOptimizer to quickly find acceleration method with the minimum inference latency under specific restrictions or without restrictions for a trained model. 
For the sake of this example, we first train the proposed network(by default, a ResNet50 is used) on the [cats and dogs dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip), which consists both [frozen and unfrozen stages](https://github.com/PyTorchLightning/pytorch-lightning/blob/495812878dfe2e31ec2143c071127990afbb082b/pl_examples/domain_templates/computer_vision_fine_tuning.py#L21-L35). Then, by calling `optimize()`, we can obtain all avaliable accelaration combinations provided by BigDL-Nano for inference. By calling `get_best_mdoel()` , we could get an accelerated model whose inference is 7.6x times faster.


## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environment.
```
conda create -n nano python=3.7  # "nano" is conda environment name, you can use any name you like.
conda activate nano
pip install jsonargparse[signatures]
pip install --pre --upgrade bigdl-nano[pytorch]

# bf16 is avaliable only on torch1.12
pip install torch==1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu 
# Necessary packages for inference accelaration
pip install --upgrade intel-extension-for-pytorch
pip install onnx onnxruntime onnxruntime-extensions
pip install openvino-dev
pip install --upgrade neural-compressor
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
You could access [cats and dogs dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip) for a view of the whole dataset.

## Run example
You can run this example with command line:

```bash
python inference_pipeline.py
```

## Results

You can find the result for inference as follows:
```
accleration option: original, latency: 133.7490ms, accuracy: 0.9890
accleration option: fp32_ipex, latency: 102.1587ms, accuracy: 0.9890
accleration option: bf16_ipex, latency: 628.2516ms, accuracy: 0.9890
accleration option: int8, latency: 34.6394ms, accuracy: 0.9880
accleration option: openvino_fp32, latency: 31.9775ms, accuracy: 0.9890
accleration option: openvino_int8, latency: 17.0428ms, accuracy: 0.9870
accleration option: onnxruntime_fp32, latency: 121.4465ms, accuracy: 0.9890
accleration option: onnxruntime_int8_qlinear, latency: 19.3709ms, accuracy: 0.9870
accleration option: onnxruntime_int8_integer, latency: 43.3232ms, accuracy: 0.9860
accleration option: jit_fp32, latency: 103.0402ms, accuracy: 0.9890
accleration option: jit_fp32_ipex, latency: 102.9166ms, accuracy: 0.9890
accleration option: jit_fp32_ipex_clast, latency: 55.8827ms, accuracy: 0.9890
inc + onnxruntime + qlinear
openvino + pot
```
