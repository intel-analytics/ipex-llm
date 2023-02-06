# Troubleshooting Guide for BigDL-Nano

Refer to this section for common issues faced while using BigDL-Nano.

## Installation
### Why I fail to install openvino-dev==2022.2 when ``pip install bigdl-nano[inference]``?

Please check your system first as openvino-dev 2022.2 does not support centos. Refer [this](https://pypi.org/project/openvino-dev/) for more details. You can install bigdl-nano[inference] >= 2.2 instead, as bigdl-nano[inference] >= 2.2 use openvino-dev >= 2022.3 which supports centos again.

## Inference

### ``could not create a primitive descriptor iterator`` when using bf16 related methods

Please make sure you use context manager provided by ``InferenceOptimizer.get_context``, you can refer this [howto guide for context manager]() for more details.

### ``assert precision in list(self.cur_config['ops'].keys())`` when using ipex quantization with inc on machine with BF16 instruction set

It's known issue for [Intel® Neural Compressor](https://github.com/intel/neural-compressor) that they don't deal with BF16 op well at version 1.13.1 . This has been fixed in version 2.0. You can install bigdl-nano[inference] >= 2.2 to fix this problem.

### Why my output is not bf16 dtype when using bf16+ipex related methods?

Please check your torch version and ipex version first. Now we only have CI/CD for ipex>=1.12, and we can't guarantee 100% normal operation when the version is lower than this.

### ``TypeError: cannot serialize xxx object`` when ``InferenceOptimizer.optimize()`` calling all ipex related methods or when ``InferenceOptimizer.trace(use_ipex=True)`` / ``InferenceOptimizer.quantize(use_ipex=True)``

In ``InferenceOptimizer.optimize()``, actually we use ``ipex.optimize(model, dtype=torch.bfloat16, inplace=False)`` to make sure not change original model. If your model can't be deepcopy, you should use ``InferenceOptimizer.trace(use_ipex=True, xxx, inplace=True)`` or ``InferenceOptimizer.quatize(use_ipex=True, xxx, inplace=True)`` instead and make sure setting ``inplace=True``.

### error message like ``set strict=False`` when ``InferenceOptimizer.trace(accelerator='jit')`` or ``InferenceOptimizer.quantize(accelerator='jit')``

You can set ``strict=False`` for ``torch.jit.trace`` by setting ``jit_strict=False`` in ``InferenceOptimizer.trace(accelerator='jit', xxx, jit_strict=False)`` or ``InferenceOptimizer.quantize(accelerator='jit', xxx, jit_strict=False)``. 
Refer [API usage of torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace) for more details.


### Why ``channels_last`` option fails for my computer vision model?

Please check the shape of your input data first, we don't support ``channels_last`` for 3D input now. If your model is a 3D model or your input data is not 4D Tensor, normally ``channels_last`` option will fail.

### Why ``InferenceOptimizer.load(dir)`` fails to load my model saved by ``InferenceOptimizer.save(model, dir)``

If you accelerate the model with ``accelerator=None`` by ``InferenceOptimizer.trace``/``InferenceOptimizer.quantize`` or it's just a normal torch.nn.Module, you have to pass original FP32 model to load pytorch model by ``InferenceOptimizer.load(dir, model=model)``.

### Why my bf16 model is slower than fp32 model?

You can first check whether your machine supports the bf16 instruction set first by ``lscpu | grep "bf16"``. If there is no ``avx512_bf16`` or ``amx_bf16`` in the output, then, without instruction set support, the performance of bf16 cannot be guaranteed, and generally, its performance will deteriorate.

### ``INVALID_ARGUMENT : Got invalid dimensions for input`` or ``[ PARAMETER_MISMATCH ] Can not clone with new dims.`` when do inference with OpenVINO / ONNXRuntime accelerated model

This error usually occurs when your dataset produces data with dynamic shape, and such case needs you to manually set ``dynamic_axes`` parameter and pass ``dynamic_axes`` to ``trace/quantize``. 

For examples, if your forward function looks like ``def forward(x: torch.Tensor):``, and it recieves 4d Tensor as input. However, your input data's shape will vary during inference, it will be (1, 3, 224, 224) or (1, 3, 256, 256), then in such case, you should:
```
dynamic_axes['x'] = {0: 'batch_size', 2: 'width', 3: 'height'}  # this means the 0/2/3 dim of your input data may vary during inference
input_sample = torch.randn(1, 3, 224, 224)
acce_model = trace(model=model, input_sample=x, dynamic_axes=dynamic_axes)
```

You can refer to [API usage of torch.onnx.export](https://pytorch.org/docs/stable/onnx.html#functions) for more details.

### Why jit didn't work on my model?

Please check first if you use `patch_cuda(disable_jit=True)` command of Nano, if you have used it to disable cuda operation, it will disable jit at the same time by `torch.jit._state.disable()`, so jit has no effect now.

### How to cope with out-of-memory during workload with Intel® Extension for PyTorch*

If you found the workload runs with Intel® Extension for PyTorch* occupies a remarkably large amount of memory, you can try to reduce the occupied memory size by setting `weights_prepack=False` when calling `InferenceOptimizer.trace` \ `InferenceOptimizer.quantize`.

### RuntimeError: Check 'false' failed at src/frontends/common/src/frontend.cpp

You may see this error when you do inference with accelerator=`OpenVINO` in keras. It only occurs when you use `intel-tensorflow` >= 2.8 and you forget `source bigdl-nano-init`. The way to solve this problem is just `source bigdl-nano-init` or `source bigdl-nano-init -j`.

### TypeError: deprecated() got an unexpected keyword argument 'name'

If a version problem caused by too low cryptography version. You can fix it by just `pip install cryptography==38.0.0` .
