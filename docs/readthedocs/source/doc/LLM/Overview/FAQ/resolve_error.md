# FAQ: How to Resolve Errors

Refer to this section for common issues faced while using BigDL-LLM.

## Installation Error

### Q: Fail to install `bigdl-llm` through `pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu`

You could try to install BigDL-LLM dependencies for Intel XPU from source archives:
- On Windows system, refer to [here](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#install-bigdl-llm-from-wheel) for steps.
- On Linux system, refer to [here](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#id3) for steps.


## Runtime Error

### Q: Get error message `PyTorch is not linked with support for xpu devices`

1. Before running on Intel GPUs, please make sure you've prepared environment follwing [installation instruction](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html).
2. If you are using an older version of `bigdl-llm` (specifically, older than 2.5.0b20240104), you need to manually add `import intel_extension_for_pytorch as ipex` at the beginning of your code.
3. After optimizing the model with BigDL-LLM, you need to move model to GPU through `model = model.to('xpu')`.
4. If you have mutil GPUs, you could refer to [here](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/KeyFeatures/multi_gpus_selection.html) for details about GPU selection.
5. If you do inference using the optimized model on Intel GPUs, you also need to set `to('xpu')` for input tensors.

### Q: import `intel_extension_for_pytorch` error on Windows GPU

Please refer to [here](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#error-loading-intel-extension-for-pytorch) for detailed guide. We list the possible missing requirements in environment which could lead to this error.

### Q: XPU device count is zero

It's recommended to reinstall driver:
- On Windows system, refer to [here](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#prerequisites) for steps.
- On Linux system, refer to [here](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#id1) for steps.

### Q: Error such as `The size of tensor a (33) must match the size of tensor b (17) at non-singleton dimension 2` duing attention forward function

If you are using BigDL-LLM PyTorch API, please try to set `optimize_llm=False` manually when call `optimize_model` function to work around. As for BigDL-LLM `transformers`-style API, you could try to set `optimize_model=False` manually when call `from_pretrained` function to work around.

### Q: Get error message `ValueError: Unrecognized configuration class`

This error is not quite relevant to BigDL-LLM. It could be that you're using the incorrect AutoClass, or the transformers version is not updated, or transformers does not support using AutoClasses to load this model. You need to refer to the model card in huggingface to confirm these information. Besides, if you load the model from local path, please also make sure you download the complete model files.

### Q: Get error message `mixed dtype (CPU): expect input to have scalar type of BFloat16` during inference

You could solve this error by converting the optimized model to `bf16` through `model.to(torch.bfloat16)` before inference.

### Q: Get error message `Native API failed. Native API returns: -5 (PI_ERROR_OUT_OF_RESOURCES) -5 (PI_ERROR_OUT_OF_RESOURCES)`

This error is caused by out of GPU memory. Some possible solutions to decrease GPU memory uage:
1. If you run several models continuously, please make sure you have released GPU memory of previous model through `del model` timely.
2. You could try `model = model.float16()` or `model = model.bfloat16()` before moving model to GPU to use less GPU memory.
3. You could try set `cpu_embedding=True` when call `from_pretrained` of AutoClass or `optimize_model` function.

### Q: Get error message `failed to enable AMX`

You could use `export BIGDL_LLM_AMX_DISABLED=1` to disable AMX manually and solve this error.
