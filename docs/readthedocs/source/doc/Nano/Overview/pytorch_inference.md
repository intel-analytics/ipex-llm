# PyTorch Inference

BigDL-Nano provides several APIs which can help users easily apply optimizations on inference pipelines to improve latency and throughput. Currently, performance accelerations are achieved by integrating extra runtimes as inference backend engines or using quantization methods on full-precision trained models to reduce computation during inference. InferenceOptimizer (`bigdl.nano.pytorch.InferenceOptimizer`) provides the APIs for all optimizations that you need for inference.

For runtime acceleration, BigDL-Nano has enabled three kinds of runtime for users in `InferenceOptimizer.trace()`, ONNXRuntime, OpenVINO and jit.

```eval_rst
.. warning::
    ``bigdl.nano.pytorch.Trainer.trace`` will be deprecated in future release.

    Please use ``bigdl.nano.pytorch.InferenceOptimizer.trace`` instead.
```

For quantization, BigDL-Nano provides only post-training quantization in `InferenceOptimizer.quantize()` for users to infer with models of 8-bit precision or 16-bit precision. Quantization-aware training is not available for now. Model conversion to 16-bit like BF16 is supported now.

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

    - INC (Intel Neural Compressor): ``pip install neural-compressor``

    - OpenVINO: ``pip install openvino-dev``

    - ONNXRuntime: ``pip install onnx onnxruntime onnxruntime-extensions onnxsim neural-compressor``

    We recommand installing all dependencies by ``pip install bigdl-nano[pytorch,inference]``, because you may run into version issues if you install dependencies manually.
```

##  Runtime Acceleration
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
trainer = Trainer()
trainer.fit(model, dataloader)
...
...

# Inference/Prediction
trainer.validate(ort_model, dataloader)
trainer.test(ort_model, dataloader)
trainer.predict(ort_model, dataloader)
```
### ONNXRuntime Acceleration
You can simply append the following part to enable your ONNXRuntime acceleration.
```python
# step 4: trace your model as an ONNXRuntime model
# if you have run `trainer.fit` before trace, then argument `input_sample` is not required.
ort_model = InferenceOptimizer.trace(model, accelerator='onnruntime', input_sample=x)

# step 5: use returned model for transparent acceleration
# The usage is almost the same with any PyTorch module
y_hat = ort_model(x)

# validate, predict, test in Trainer also support acceleration
trainer.validate(ort_model, dataloader)
trainer.test(ort_model, dataloader)
trainer.predict(ort_model, dataloader)
# note that `ort_model` is not trainable any more, so you can't use like
# trainer.fit(ort_model, dataloader) # this is illegal
```
### OpenVINO Acceleration
The OpenVINO usage is quite similar to ONNXRuntime, the following usage is for OpenVINO:
```python
# step 4: trace your model as a openvino model
# if you have run `trainer.fit` before trace, then argument `input_sample` is not required.
ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=x)

# step 5: use returned model for transparent acceleration
# The usage is almost the same with any PyTorch module
y_hat = ov_model(x)

# validate, predict, test in Trainer also support acceleration
trainer = Trainer()
trainer.validate(ort_model, dataloader)
trainer.test(ort_model, dataloader)
trainer.predict(ort_model, dataloader)
# note that `ort_model` is not trainable any more, so you can't use like
# trainer.fit(ort_model, dataloader) # this is illegal
```

### Jit Acceleration


## Quantization
Quantization is widely used to compress models to a lower precision, which not only reduces the model size but also accelerates inference. BigDL-Nano provides `InferenceOptimizer.quantize()` API for users to quickly obtain a quantized model with accuracy control by specifying a few arguments. Intel Neural Compressor (INC) and Post-training Optimization Tools (POT) from OpenVINO toolkit are enabled as options. In the meantime, runtime acceleration is also included directly in the quantization pipeline when using `accelerator='onnxruntime'/'openvino'` so you don't have to run `InferenceOptimizer.trace` before quantization.

To use INC as your quantization engine, you can choose accelerator as `None` or `'onnxruntime'`. Otherwise, `accelerator='openvino'` means using OpenVINO POT to do quantization.

By default, `InferenceOptimizer.quantize()` doesn't search the tuning space and returns the fully-quantized model without considering the accuracy drop. If you need to search quantization tuning space for a model with accuracy control, you'll have to specify a few arguments to define the tuning space. More instructions in [Quantization with Accuracy Control](#quantization-with-accuracy-control)

### Quantization using Intel Neural Compressor
By default, Intel Neural Compressor is not installed with BigDL-Nano. So if you determine to use it as your quantization backend, you'll need to install it first:
```shell
pip install neural-compressor==1.11.0
```
**Quantization without extra accelerator**

Without extra accelerator, `InferenceOptimizer.quantize()` returns a PyTorch module with desired precision and accuracy. Following the example in [Runtime Acceleration](#runtime-acceleration), you can add quantization as below:
```python
q_model = InferenceOptimizer.quantize(model, calib_dataloader=dataloader)
# run simple prediction with transparent acceleration
y_hat = q_model(x)

# validate, predict, test in Trainer also support acceleration
trainer.validate(q_model, dataloader)
trainer.test(q_model, dataloader)
trainer.predict(q_model, dataloader)
```
This is a most basic usage to quantize a model with defaults, INT8 precision, and without search tuning space to control accuracy drop.

**Quantization with ONNXRuntime accelerator**

With the ONNXRuntime accelerator, `InferenceOptimizer.quantize()` will return a model with compressed precision and running inference in the ONNXRuntime engine. It's also required to install onnxruntime-extensions as a dependency of INC when using ONNXRuntime as backend as well as the dependencies required in [ONNXRuntime Acceleration](#onnxruntime-acceleration):
```shell
pip install onnx onnxruntime onnxruntime-extensions
```
Still taking the example in [Runtime Acceleration](pytorch_inference.md#runtime-acceleration), you can add quantization as below:
```python
ort_q_model = InferenceOptimizer.quantize(model, accelerator='onnxruntime', calib_dataloader=dataloader)
# run simple prediction with transparent acceleration
y_hat = ort_q_model(x)

# validate, predict, test in Trainer also support acceleration
trainer.validate(ort_q_model, dataloader)
trainer.test(ort_q_model, dataloader)
trainer.predict(ort_q_model, dataloader)
```
Using `accelerator='onnxruntime'` actually equals to converting the model from PyTorch to ONNX firstly and then do quantization on the converted ONNX model:
```python
ort_model = InferenceOptimizer.trace(model, accelerator='onnruntime', input_sample=x):
ort_q_model = InferenceOptimizer.quantize(ort_model, accelerator='onnxruntime', calib_dataloader=dataloader)

# run inference with transparent acceleration
y_hat = ort_q_model(x)
trainer.validate(ort_q_model, dataloader)
trainer.test(ort_q_model, dataloader)
trainer.predict(ort_q_model, dataloader)
```

### Quantization using Post-training Optimization Tools
The POT (Post-training Optimization Tools) is provided by OpenVINO toolkit. To use POT, you need to install OpenVINO as the same in [OpenVINO acceleration](#openvino-acceleration):
```shell
pip install openvino-dev
```
Take the example in [Runtime Acceleration](#runtime-acceleration), and add quantization:
```python
ov_q_model = InferenceOptimizer.quantize(model, accelerator='openvino', calib_dataloader=dataloader)
# run simple prediction with transparent acceleration
y_hat = ov_q_model(x)

# validate, predict, test in Trainer also support acceleration
trainer.validate(ov_q_model, dataloader)
trainer.test(ov_q_model, dataloader)
trainer.predict(ov_q_model, dataloader)
```
Same as using ONNXRuntime accelerator, it equals to converting the model from PyTorch to OpenVINO firstly and then doing quantization on the converted OpenVINO model:
```python
ov_model = InferenceOptimizer.trace(model, accelerator='openvino', input_sample=x):
ov_q_model = InferenceOptimizer.quantize(ov_model, accelerator='onnxruntime', calib_dataloader=dataloader)

# run inference with transparent acceleration
y_hat = ov_q_model(x)
trainer.validate(ov_q_model, dataloader)
trainer.test(ov_q_model, dataloader)
trainer.predict(ov_q_model, dataloader)
```

### Quantization with Accuracy Control
A set of arguments that helps to tune the results for both INC and POT quantization:

- `calib_dataloader`: A calibration dataloader is required for static post-training quantization. And for POT, it's also used for evaluation
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
from torchmetrics.classification import Accuracy
InferenceOptimizer.quantize(model,
                            precision='int8',
                            accelerator=None,
                            calib_dataloader= dataloader,
                            metric=Accuracy()
                            accuracy_criterion={'relative': 0.01, 'higher_is_better': True},
                            approach='static',
                            method='fx',
                            tuning_strategy='bayesian',
                            timeout=0,
                            max_trials=10,
                            ):
```
**Accuracy Control with POT**
Similar to INC, we can run quantization like:
```python
from torchmetrics.classification import Accuracy
InferenceOptimizer.quantize(model,
                            precision='int8',
                            accelerator='openvino',
                            calib_dataloader= dataloader,
                            metric=Accuracy()
                            accuracy_criterion={'relative': 0.01, 'higher_is_better': True},
                            approach='static',
                            max_trials=10,
                            ):
```

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
