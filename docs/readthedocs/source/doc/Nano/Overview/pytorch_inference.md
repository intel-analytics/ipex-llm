# PyTorch Inference

BigDL-Nano provides several APIs which can help users easily apply optimizations on inference pipelines to improve latency and throughput. Currently, performance accelerations are achieved by integrating extra runtimes as inference backend engines or using quantization methods on full-precision trained models to reduce computation during inference. InferenceOptimizer (`bigdl.nano.pytorch.InferenceOptimizer`) provides the APIs for all optimizations that you need for inference.

For runtime acceleration, BigDL-Nano has enabled three kinds of graph mode format and corresponding runtime in `InferenceOptimizer.trace()`: ONNXRuntime, OpenVINO and TorchScript.

```eval_rst
.. warning::
    ``bigdl.nano.pytorch.Trainer.trace`` will be deprecated in future release.

    Please use ``bigdl.nano.pytorch.InferenceOptimizer.trace`` instead.
```

For quantization, BigDL-Nano provides only post-training quantization in `InferenceOptimizer.quantize()` for users to infer with models of 8-bit precision or 16-bit precision. Quantization-aware training is not available for now.

```eval_rst
.. warning::
    ``bigdl.nano.pytorch.Trainer.quantize`` will be deprecated in future release.

    Please use ``bigdl.nano.pytorch.InferenceOptimizer.quantize`` instead.
```

Before you go ahead with these APIs, you have to make sure BigDL-Nano is correctly installed for PyTorch. If not, please follow [this](../Overview/nano.md) to set up your environment.

```eval_rst
.. note::
    You can install all required dependencies by

    ::

        pip install --pre --upgrade bigdl-nano[pytorch,inference]

    This will install all dependencies required by BigDL-Nano PyTorch inference.

    Or if you just want to use one of supported optimizations:

    - `INC (Intel Neural Compressor) <https://github.com/intel/neural-compressor>`_: ``pip install neural-compressor``

    - `OpenVINO <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html>`_: ``pip install openvino-dev``

    - `ONNXRuntime <https://onnxruntime.ai/>`_: ``pip install onnx onnxruntime onnxruntime-extensions onnxsim neural-compressor``

    We recommand installing all dependencies by ``pip install --pre --upgrade bigdl-nano[pytorch,inference]``, because you may run into version issues if you install dependencies manually.
```

## Graph Mode Acceleration
All available runtime accelerations are integrated in `InferenceOptimizer.trace(accelerator='onnxruntime'/'openvino'/'jit')` with different accelerator values. Let's take mobilenetv3 as an example model and here is a short script that you might have before applying any BigDL-Nano's optimizations:
```python
from torchvision.models.mobilenetv3 import mobilenet_v3_small
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from bigdl.nano.pytorch import InferenceOptimizer, Trainer

# step 1: create your model
model = mobilenet_v3_small(num_classes=10)

# step 2: prepare your data and dataloader
x = torch.rand((10, 3, 256, 256))
y = torch.ones((10, ), dtype=torch.long)
ds = TensorDataset(x, y)
dataloader = DataLoader(ds, batch_size=2)

# (Optional) step 3: Something else, like training ...
```
### ONNXRuntime Acceleration
You can simply append the following part to enable your [ONNXRuntime](https://onnxruntime.ai/) acceleration.
```python
# step 4: trace your model as an ONNXRuntime model
# if you have run `trainer.fit` before trace, then argument `input_sample` is not required.
ort_model = InferenceOptimizer.trace(model, accelerator='onnxruntime', input_sample=x)

# step 5: use returned model for transparent acceleration
# The usage is almost the same with any PyTorch module
y_hat = ort_model(x)
```
### OpenVINO Acceleration
The [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) usage is quite similar to ONNXRuntime, the following usage is for OpenVINO:
```python
# step 4: trace your model as a openvino model
# if you have run `trainer.fit` before trace, then argument `input_sample` is not required.
ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=x)

# step 5: use returned model for transparent acceleration
# The usage is almost the same with any PyTorch module
y_hat = ov_model(x)
```

### TorchScript Acceleration
The [TorchScript](https://pytorch.org/docs/stable/jit.html) usage is a little different from above two cases. In addition to specifying `accelerator=jit`, you can also set `use_ipex=True` to enable the additional acceleration provided by [IPEX (Intel® Extension for PyTorch*)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/extension-for-pytorch.html), we generally recommend the combination of `jit` and `ipex`.The following usage is for TorchScript:
```python
# step 4: trace your model as a JIT model
jit_model = InferenceOptimizer.trace(model, accelerator='jit', input_sample=x)

# or you can combine jit with ipex
jit_model = InferenceOptimizer.trace(model, accelerator='jit',
                                     use_ipex=True, input_sample=x)

# step 5: use returned model for transparent acceleration
# The usage is almost the same with any PyTorch module
y_hat = jit_model(x)
```

## Quantization
Quantization is widely used to compress models to a lower precision, which not only reduces the model size but also accelerates inference. For quantization precision, BigDL-Nano supports two common choices: `int8` and `bfloat16`. The usage of the two kinds of precision is quite different.

### Int8 Quantization
BigDL-Nano provides `InferenceOptimizer.quantize()` API for users to quickly obtain a int8 quantized model with accuracy control by specifying a few arguments. Intel Neural Compressor (INC) and Post-training Optimization Tools (POT) from OpenVINO toolkit are enabled as options.

To use INC as your quantization engine, you can choose accelerator as `None` or `'onnxruntime'`. Otherwise, `accelerator='openvino'` means using OpenVINO POT to do quantization.

By default, `InferenceOptimizer.quantize()` doesn't search the tuning space and returns the fully-quantized model without considering the accuracy drop. If you need to search quantization tuning space for a model with accuracy control, you'll have to specify a few arguments to define the tuning space. More instructions in [Quantization with Accuracy Control](#quantization-with-accuracy-control)

#### Quantization using Intel Neural Compressor
**Quantization without extra accelerator**

Without extra accelerator, `InferenceOptimizer.quantize()` returns a PyTorch module with desired precision and accuracy. Following the example in [Runtime Acceleration](#runtime-acceleration), you can add quantization as below:
```python
q_model = InferenceOptimizer.quantize(model, calib_data=dataloader)
# run simple prediction with transparent acceleration
y_hat = q_model(x)
```
This is a most basic usage to quantize a model with defaults, INT8 precision, and without search tuning space to control accuracy drop.

**Quantization with ONNXRuntime accelerator**

With the ONNXRuntime accelerator, `InferenceOptimizer.quantize()` will return a model with compressed precision and running inference in the ONNXRuntime engine.
Still taking the example in [Runtime Acceleration](pytorch_inference.md#runtime-acceleration), you can add quantization as below:
```python
ort_q_model = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_data=dataloader)
# run simple prediction with transparent acceleration
y_hat = ort_q_model(x)
```

#### Quantization using Post-training Optimization Tools
The POT (Post-training Optimization Tools) is provided by OpenVINO toolkit.
Take the example in [Runtime Acceleration](#runtime-acceleration), and add quantization:
```python
ov_q_model = InferenceOptimizer.quantize(model, accelerator='openvino', calib_data=dataloader)
# run simple prediction with transparent acceleration
y_hat = ov_q_model(x)
```

#### Quantization with Accuracy Control
A set of arguments that helps to tune the results for both INC and POT quantization:

- `calib_data`: A calibration dataloader is required for static post-training quantization. And for POT, it's also used for evaluation
- `metric`: A metric of `torchmetric` to run evaluation and compare with baseline

- `accuracy_criterion`: A dictionary to specify the acceptable accuracy drop, e.g. `{'relative': 0.01, 'higher_is_better': True}`

    - `relative` / `absolute`: Drop type, the accuracy drop should be relative or absolute to baseline
    - `higher_is_better`: Indicate if a larger value of metric means better accuracy
- `max_trials`: Maximum trails on the search, if the algorithm can't find a satisfying model, it will exit and raise the error.

**Accuracy Control with INC**
There are a few arguments required only by INC, and you should not specify or modify any of them if you use `accelerator='openvino'`.
- `tuning_strategy` (optional): it specifies the algorithm to search the tuning space. In most cases, you don't need to change it.
- `timeout`: Timeout of your tuning. Defaults `0` means endless time for tuning.

Here is an example to use INC with accuracy control as below. It will search for a model within 1% accuracy drop with 10 trials.
```python
from torchmetrics.classification import MulticlassAccuracy
InferenceOptimizer.quantize(model,
                            precision='int8',
                            accelerator=None,
                            calib_data=dataloader,
                            metric=MulticlassAccuracy(num_classes=10)
                            accuracy_criterion={'relative': 0.01, 'higher_is_better': True},
                            approach='static',
                            method='fx',
                            tuning_strategy='bayesian',
                            timeout=0,
                            max_trials=10,
                            )
```
**Accuracy Control with POT**
Similar to INC, we can run quantization like:
```python
from torchmetrics.classification import Accuracy
InferenceOptimizer.quantize(model,
                            precision='int8',
                            accelerator='openvino',
                            calib_data=dataloader,
                            metric=MulticlassAccuracy(num_classes=10)
                            accuracy_criterion={'relative': 0.01, 'higher_is_better': True},
                            approach='static',
                            max_trials=10,
                            )
```

### BFloat16 Quantization

BigDL-Nano has support [mixed precision inference](https://pytorch.org/docs/stable/amp.html?highlight=mixed+precision) with BFloat16 and a series of additional performance tricks. BFloat16 Mixed Precison inference combines BFloat16 and FP32 during inference, which could lead to increased performance and reduced memory usage. Compared to FP16 mixed precison, BFloat16 mixed precision has better numerical stability.
It's quite easy for you use BFloat16 Quantization as below:
```python
bf16_model = InferenceOptimizer.quantize(model,
                                         precision='bf16')
# run simple prediction with transparent acceleration
with InferenceOptimizer.get_context(bf16_model):
    y_hat = bf16_model(x)
```

```eval_rst
.. note::
    For BFloat16 quantization, make sure your inference is under ``with InferenceOptimizer.get_context(bf16_model):``. Otherwise, the whole inference process is actually FP32 precision.

    For more details about the context manager provided by ``InferenceOptimizer.get_context()``, you could refer related `How-to guide <https://bigdl.readthedocs.io/en/latest/doc/Nano/Howto/Inference/PyTorch/pytorch_context_manager.html>`_.
```

#### Channels Last Memory Format
You could experience Bfloat16 Quantization with `channels_last=True` to use the channels last memory format, i.e. NHWC (batch size, height, width, channels), as an alternative way to store tensors in classic/contiguous NCHW order.
The usage for this is as below:
```python
bf16_model = InferenceOptimizer.quantize(model,
                                         precision='bf16',
                                         channels_last=True)
# run simple prediction with transparent acceleration
with InferenceOptimizer.get_context(bf16_model):
    y_hat = bf16_model(x)
```

#### Intel® Extension for PyTorch
[Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) (a.k.a. IPEX) extends PyTorch with optimizations for an extra performance boost on Intel hardware.

BigDL-Nano integrates IPEX through `InferenceOptimizer.quantize()`. Users can turn on IPEX by setting `use_ipex=True`:
```python
bf16_model = InferenceOptimizer.quantize(model,
                                         precision='bf16',
                                         use_ipex=True,
                                         channels_last=True)
# run simple prediction with transparent acceleration
with InferenceOptimizer.get_context(bf16_model):
    y_hat = bf16_model(x)
```

#### TorchScript Acceleration
The [TorchScript](https://pytorch.org/docs/stable/jit.html) can also be used for Bfloat16 quantization. We recommend you take advantage of IPEX with TorchScript for further optimizations. The following usage is for TorchScript:
```python
bf16_model = InferenceOptimizer.quantize(model,
                                         precision='bf16',
                                         accelerator='jit',
                                         input_sample=x,
                                         use_ipex=True,
                                         channels_last=True)
# run simple prediction with transparent acceleration
with InferenceOptimizer.get_context(bf16_model):
    y_hat = bf16_model(x)
```

## Automatically Choose the Best Optimization

If you have no idea about which one optimization to choose or you just want to compare them and choose the best one, you can use `InferenceOptimizer.optimize`.

Still taking the example in [Runtime Acceleration](#runtime-acceleration), you can use it as following:
```python
# try all supproted optimizations
opt = InferenceOptimizer()
opt.optimize(model, training_data=dataloader, thread_num=4)

# get the best optimization
best_model, option = opt.get_best_model()

# use the quantized model as before
with InferenceOptimizer.get_context(best_model):
    y_hat = best_model(x)
```

`InferenceOptimizer.optimize()` will try all supported optimizations and choose the best one by `get_best_model()`.
The output table of `optimize()` looks like:
```bash
 -------------------------------- ---------------------- --------------
|             method             |        status        | latency(ms)  |
 -------------------------------- ---------------------- --------------
|            original            |      successful      |    9.337     |
|              bf16              |      successful      |    8.974     |
|          static_int8           |      successful      |    8.934     |
|         jit_fp32_ipex          |      successful      |    10.013    |
|  jit_fp32_ipex_channels_last   |      successful      |    4.955     |
|         jit_bf16_ipex          |      successful      |    2.563     |
|  jit_bf16_ipex_channels_last   |      successful      |    3.135     |
|         openvino_fp32          |      successful      |    1.727     |
|         openvino_int8          |      successful      |    1.635     |
|        onnxruntime_fp32        |      successful      |    3.801     |
|    onnxruntime_int8_qlinear    |      successful      |    4.727     |
 -------------------------------- ---------------------- --------------
Optimization cost 58.3s in total.
```

For more details, you can refer [How-to guide](https://bigdl.readthedocs.io/en/latest/doc/Nano/Howto/Inference/PyTorch/inference_optimizer_optimize.html) and [API Doc](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Nano/pytorch.html#bigdl-nano-pytorch-inferenceoptimizer). 

## Multi-instance Acceleration

BigDL-Nano also provides multi-instance inference. To use it, you should call `multi_model = InferenceOptimizer.to_multi_instance(model, num_processes=n)` first, where `num_processes` specifies the number of processes you want to use.

After calling it, `multi_model` will receive a `DataLoader` or a list of batches instead of a batch, and produce a list of inference result instead of a single result. You can use it as following:

```python
multi_model = InferenceOptimizer.to_multi_instance(model, num_processes=4)

# predict a DataLoader
y_hat_list = multi_model(dataloader)

# or predict a list of batches instead of entire DataLoader
it = iter(dataloader)
batch_list = []
for i in range(10):
    batch = next(it)
    batch_list.append(batch)
y_hat_list = multi_model(batch_list)

# y_hat_list is a list of inference result, you can use it like this
for y_hat in y_hat_list:
    do_something(y_hat)
```

`InferenceOptimizer.to_multi_instance` also has a parameter named `cores_per_process` to specify the number of CPU cores used by each process, and a parameter named `cpu_for_each_process` to specify the CPU cores used by each process. Normally you don't need to set them manually, BigDL-Nano will find the best configuration automatically. But if you want, you can use them as following:
```python
# Use 4 processes to run inference,
# each process will use 2 CPU cores
multi_model = InferenceOptimizer.to_multi_instance(model, num_processes=4, cores_per_process=2)

# Use 4 processes to run inference,
# the first process will use core 0, the second process will use core 1,
# the third process will use core 2 and 3, the fourth process will use core 4 and 5
multi_model = InferenceOptimizer.to_multi_instance(model, cpu_for_each_process=[[0], [1], [2,3], [4,5]])
```

## Automatic Context Management
BigDL-Nano provides ``InferenceOptimizer.get_context(model=...)`` API to enable automatic context management for PyTorch inference. With only one line of code change, BigDL-Nano will automatically provide suitable context management for each accelerated model, it usually contains part of or all of following three types of context managers:

1. ``torch.no_grad()`` to disable gradients, which will be used for all model
   
2. ``torch.cpu.amp.autocast(dtype=torch.bfloat16)`` to run in mixed precision, which will be provided for bf16 related model
   
3. ``torch.set_num_threads()`` to control thread number, which will be used only if you specify thread_num when applying ``InferenceOptimizer.trace``/``quantize``/``optimize``

For model accelerated by ``InferenceOptimizer.trace``, usage now looks like below codes, here we just take ``ipex`` for example:
```python
from bigdl.nano.pytorch import InferenceOptimizer
ipex_model = InferenceOptimizer.trace(model,
                                      use_ipex=True,
                                      thread_num=4)

with InferenceOptimizer.get_context(ipex_model):
    output = ipex_model(x)
    assert torch.get_num_threads() == 4  # this line just to let you know Nano has provided thread control automatically : )
```

For ``InferenceOptimizer.quantize`` and ``InferenceOptimizer.optimize``, usage is the same.

``InferenceOptimizer.get_context(model=...)`` can be used for muitiple models. If you have a model pipeline, you can also get a common context manager by passing multiple models to `get_context`.
```python
from torch import nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1000, 1)
    
    def forward(self, x):
        return self.linear(x)

classifer = Classifier()

with InferenceOptimizer.get_context(ipex_model, classifer):
    # a pipeline consists of backbone and classifier
    x = ipex_model(input_sample)
    output = classifer(x) 
    assert torch.get_num_threads() == 4  # this line just to let you know Nano has provided thread control automatically : )
```

## One-click Accleration Without Code Change
```eval_rst
.. note::
    Neural Compressor >= 2.0 is needed for this function. You may call ``pip install --upgrade neural-compressor`` before using this functionality.
```

We also provides a no-code method for users to accelerate their pytorch inferencing workflow through Neural Coder. Neural Coder is a novel component under Intel® Neural Compressor to further simplify the deployment of deep learning models via one-click. BigDL-Nano is now a backend in Neural Coder. Users could call

```bash
python -m neural_coder -o <acceleration_name> example.py
```

For `example.py`, it could be a common pytorch inference script without any code changes needed. For `<acceleration_name>`, please check following table.

| Optimization Set | `<acceleration_name>` | 
| ------------- | ------------- | 
| BF16 + Channels Last | `nano_bf16_channels_last` | 
| BF16 + IPEX + Channels Last | `nano_bf16_ipex_channels_last` | 
| BF16 + IPEX | `nano_bf16_ipex` | 
| BF16 | `nano_bf16` | 
| Channels Last | `nano_fp32_channels_last` | 
| IPEX + Channels Last | `nano_fp32_ipex_channels_last` | 
| IPEX | `nano_fp32_ipex` | 
| INT8 | `nano_int8` | 
| JIT + BF16 + Channels Last | `nano_jit_bf16_channels_last` | 
| JIT + BF16 + IPEX + Channels Last | `nano_jit_bf16_ipex_channels_last` | 
| JIT + BF16 + IPEX | `nano_jit_bf16_ipex` | 
| JIT + BF16 | `nano_jit_bf16` | 
| JIT + Channels Last | `nano_jit_fp32_channels_last` | 
| JIT + IPEX + Channels Last | `nano_jit_fp32_ipex_channels_last` | 
| JIT + IPEX | `nano_jit_fp32_ipex` | 
| JIT | `nano_jit_fp32` | 
| ONNX Runtime | `nano_onnxruntime_fp32` | 
| ONNX Runtime + INT8 | `nano_onnxruntime_int8_qlinear` | 
| OpenVINO | `nano_openvino_fp32` | 
| OpenVINO + INT8 | `nano_openvino_int8` |