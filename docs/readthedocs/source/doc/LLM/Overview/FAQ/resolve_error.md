# FAQ: How to Resolve Errors

Refer to this section for common issues faced while using BigDL-LLM.

## Runtime Error

### PyTorch is not linked with support for xpu devices

1. Before running on Intel GPUs, please make sure you've prepared environment follwing [installation instruction](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html).
2. If you are using an older version of `bigdl-llm` (specifically, older than 2.5.0b20240104), you need to manually add `import intel_extension_for_pytorch as ipex` at the beginning of your code.
3. After optimizing the model with BigDL-LLM, you need to move model to GPU through `model = model.to('xpu')`.
4. If you have mutil GPUs, you could refer to [here](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/KeyFeatures/multi_gpus_selection.html) for details about GPU selection.
