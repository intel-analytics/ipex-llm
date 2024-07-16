# Frequently Asked Questions (FAQ)

## General Info & Concepts

### GGUF format usage with IPEX-LLM?

IPEX-LLM supports running GGUF/AWQ/GPTQ models on both [CPU](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/CPU/HF-Transformers-AutoModels/Advanced-Quantizations) and [GPU](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/example/GPU/HuggingFace/Advanced-Quantizations).

Please also refer to [here](https://github.com/intel-analytics/ipex-llm?tab=readme-ov-file#latest-update-) for our latest support.

## How to Resolve Errors

### Fail to install `ipex-llm` via `pip` on Intel GPU

If you encounter errors when installing `ipex-llm` on Intel GPU using either

```python
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
``` 
or 
```python
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```

You can try install `ipex-llm` dependencies from source archives:
- For Windows system, refer to [here](../install_gpu.md#install-ipex-llm-from-wheel) for the steps.
- For Linux system, refer to [here](../install_gpu.md#prerequisites-1) for the steps.

### PyTorch is not linked with support for xpu devices

1. Before running on Intel GPUs, please make sure you've prepared environment follwing [installation instruction](../install_gpu.md).
2. If you are using an older version of `ipex-llm` (specifically, older than 2.5.0b20240104), you need to manually add `import intel_extension_for_pytorch as ipex` at the beginning of your code.
3. After optimizing the model with IPEX-LLM, you need to move model to GPU through `model = model.to('xpu')`.
4. If you have mutil GPUs, you could refer to [here](../KeyFeatures/multi_gpus_selection.md) for details about GPU selection.
5. If you do inference using the optimized model on Intel GPUs, you also need to set `to('xpu')` for input tensors.

### Import `intel_extension_for_pytorch` error on Windows GPU

Please refer to [here](../install_gpu.md#1-error-loading-intel_extension_for_pytorch)
for detailed guide. We list the possible missing requirements in environment which could lead to this error.

### XPU device count is zero

It's recommended to reinstall driver:
- For Windows system, refer to [here](../install_gpu.md#windows) for the steps.
- For Linux system, refer to [here](../install_gpu.md#prerequisites-1) for the steps.

### Error such as `The size of tensor a (33) must match the size of tensor b (17) at non-singleton dimension 2` duing attention forward function

If you are using IPEX-LLM PyTorch API, please try to set `optimize_llm=False` manually when call `optimize_model` function to work around. As for IPEX-LLM `transformers`-style API, you could try to set `optimize_model=False` manually when call `from_pretrained` function to work around.

### ValueError: Unrecognized configuration class

This error is not quite relevant to IPEX-LLM. It could be that you're using the incorrect AutoClass, or the transformers version is not updated, or transformers does not support using AutoClasses to load this model. You need to refer to the model card in huggingface to confirm these information. Besides, if you load the model from local path, please also make sure you download the complete model files.

### `mixed dtype (CPU): expect input to have scalar type of BFloat16` during inference

You could solve this error by converting the optimized model to `bf16` through `model.to(torch.bfloat16)` before inference.

### Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES) -5 (PI_ERROR_OUT_OF_RESOURCES)

This error is caused by out of GPU memory. Some possible solutions to decrease GPU memory uage:
1. If you run several models continuously, please make sure you have released GPU memory of previous model through `del model` timely.
2. You could try `model = model.float16()` or `model = model.bfloat16()` before moving model to GPU to use less GPU memory.
3. You could try set `cpu_embedding=True` when call `from_pretrained` of AutoClass or `optimize_model` function.

### Failed to enable AMX

You could use `export BIGDL_LLM_AMX_DISABLED=1` to disable AMX manually and solve this error.

### oneCCL: comm_selector.cpp:57 create_comm_impl: EXCEPTION: ze_data was not initialized

You may encounter this error during finetuning on multi GPUs. Please try `sudo apt install level-zero-dev` to fix it.

### Random and unreadable output of Gemma-7b-it on Arc770 ubuntu 22.04 due to driver and OneAPI missmatching.

If driver and OneAPI missmatching, it will lead to some error when IPEX-LLM uses XMX(short prompts) for speeding up.
The output of `What's AI?` may like below:
```
wiedzy Artificial Intelligence meliti: Artificial Intelligence undenti beng beng beng beng beng beng beng beng beng beng beng beng beng beng beng beng beng beng beng beng beng beng
```
If you meet this error. Please check your driver version and OneAPI version. Commnad is `sudo apt list --installed | egrep "intel-basekit|intel-level-zero-gpu"`. 
Make sure intel-basekit>=2024.0.1-43 and intel-level-zero-gpu>=1.3.27191.42-775~22.04.

### Too many open files

You may encounter this error during finetuning, expecially when run 70B model. Please raise the system open file limit using `ulimit -n 1048576`.

### `RuntimeError: could not create a primitive` on Windows

This error may happen when multiple GPUs exists for Windows Users. To solve this error, you can open Device Manager (search "Device Manager" in your start menu). Then click the "Display adapter" option, and disable all the GPU device you do not want to use. Restart your computer and try again. IPEX-LLM should work fine this time.

### The first time to run model on Meteor Lake's iGPU(Intel Core™ Ultra integrated GPU) will takes 5-10 minutes.
This is caused by intel-extension-for-pytorch's compilation. For windows user, you can install this [intel-extension-for-pytorch](https://github.com/Nuullll/intel-extension-for-pytorch/releases) with Meteor Lake's [ahead-of-time-compilation](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/ahead-of-time-compilation.html).
Please notice the packages only support python 3.10, you can use below commands after you have installed ipex-llm[xpu]:
```bash
pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.20%2Bmtl%2Boneapi/intel_extension_for_pytorch-2.1.20+git4849f3b-cp310-cp310-win_amd64.whl
pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.20%2Bmtl%2Boneapi/torch-2.1.0a0+git7bcf7da-cp310-cp310-win_amd64.whl
pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.20%2Bmtl%2Boneapi/torchvision-0.16.0+fbb4cc5-cp310-cp310-win_amd64.whl
```
