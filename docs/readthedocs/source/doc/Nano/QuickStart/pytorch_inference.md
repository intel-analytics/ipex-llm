# BigDL-Nano PyTorch Inference Overview

BigDL-Nano provides several APIs which can help users easily apply optimizations on inference pipelines to improve latency and throughput. Currently, performance accelerations are achieved by integrating extra runtimes as inference backend engines or using quantization methods on full-precision trained models to reduce computation during inference. Trainer (`bigdl.nano.pytorch.Trainer`) provides the APIs for all optimizations that you need for inference.

For runtime acceleration, BigDL-Nano has enabled two kinds of runtime for users in `Trainer.trace()`, ONNXRuntime and OpenVINO.

For quantization, BigDL-Nano provides only post-training quantization in `trainer.quantize()` for users to infer with models of 8-bit precision. Quantization-aware training is not available for now. Model conversion to 16-bit like BF16, and FP16 will be coming soon.

Before you go ahead with these APIs, you have to make sure BigDL-Nano is correctly installed for PyTorch. If not, please follow [this](../Overview/nano.md) to set up your environment.

##  Runtime Acceleration
All available runtime accelerations are integrated in `Trainer.trace(accelerator='onnxruntime'/'openvino')` with different accelerator values. Let's take mobilenetv3 as an example model and here is a short script that you might have before applying any BigDL-Nano's optimizations:
```python
from torchvision.models.mobilenetv3 import mobilenet_v3_small
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from bigdl.nano.pytorch import Trainer
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
Before you start with ONNXRuntime accelerator, you are required to install some ONNX packages as follows to set up your environment with ONNXRuntime acceleration.
```shell
pip install onnx onnxruntime
```
When you're ready, you can simply append the following part to enable your ONNXRuntime acceleration.
```python
# step 4: trace your model as an ONNXRuntime model
# if you have run `trainer.fit` before trace, then argument `input_sample` is not required.
ort_model = Trainer.trace(model, accelerator='onnruntime', input_sample=x)

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
To use OpenVINO acceleration, you have to install the OpenVINO toolkit:
```shell
pip install openvino-dev
```
The OpenVINO usage is quite similar to ONNXRuntime, the following usage is for OpenVINO:
```python
# step 4: trace your model as a openvino model
# if you have run `trainer.fit` before trace, then argument `input_sample` is not required.
ov_model = Trainer.trace(model, accelerator='openvino', input_sample=x)

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

## Quantization
Quantization is widely used to compress models to a lower precision, which not only reduces the model size but also accelerates inference. BigDL-Nano provides `Trainer.quantize()` API for users to quickly obtain a quantized model with accuracy control by specifying a few arguments. Intel Neural Compressor (INC) and Post-training Optimization Tools (POT) from OpenVINO toolkit are enabled as options. In the meantime, runtime acceleration is also included directly in the quantization pipeline when using `accelerator='onnxruntime'/'openvino'` so you don't have to run `Trainer.trace` before quantization.

To use INC as your quantization engine, you can choose accelerator as `None` or `'onnxruntime'`. Otherwise, `accelerator='openvino'` means using OpenVINO POT to do quantization.

By default, `Trainer.quantize()` doesn't search the tuning space and returns the fully-quantized model without considering the accuracy drop. If you need to search quantization tuning space for a model with accuracy control, you'll have to specify a few arguments to define the tuning space. More instructions in [Quantization with Accuracy Control](#quantization-with-accuracy-control)

### Quantization using Intel Neural Compressor
By default, Intel Neural Compressor is not installed with BigDL-Nano. So if you determine to use it as your quantization backend, you'll need to install it first:
```shell
pip install neural-compressor==1.11.0
```
**Quantization without extra accelerator**

Without extra accelerator, `Trainer.quantize()` returns a PyTorch module with desired precision and accuracy. Following the example in [Runtime Acceleration](#runtime-acceleration), you can add quantization as below:
```python
q_model = trainer.quantize(model, calib_dataloader=dataloader)
# run simple prediction with transparent acceleration
y_hat = q_model(x)

# validate, predict, test in Trainer also support acceleration
trainer.validate(q_model, dataloader)
trainer.test(q_model, dataloader)
trainer.predict(q_model, dataloader)
```
This is a most basic usage to quantize a model with defaults, INT8 precision, and without search tuning space to control accuracy drop.  

**Quantization with ONNXRuntime accelerator**

With the ONNXRuntime accelerator, `Trainer.quantize()` will return a model with compressed precision and running inference in the ONNXRuntime engine. It's also required to install onnxruntime-extensions as a dependency of INC when using ONNXRuntime as backend as well as the dependencies required in [ONNXRuntime Acceleration](#onnxruntime-acceleration):
```shell
pip install onnx onnxruntime onnxruntime-extensions
```
Still taking the example in [Runtime Acceleration](pytorch_inference.md#runtime-acceleration), you can add quantization as below:
```python
ort_q_model = trainer.quantize(model, accelerator='onnxruntime', calib_dataloader=dataloader)
# run simple prediction with transparent acceleration
y_hat = ort_q_model(x)

# validate, predict, test in Trainer also support acceleration
trainer.validate(ort_q_model, dataloader)
trainer.test(ort_q_model, dataloader)
trainer.predict(ort_q_model, dataloader)
```
Using `accelerator='onnxruntime'` actually equals to converting the model from PyTorch to ONNX firstly and then do quantization on the converted ONNX model:
```python
ort_model = Trainer.trace(model, accelerator='onnruntime', input_sample=x):
ort_q_model = trainer.quantize(ort_model, accelerator='onnxruntime', calib_dataloader=dataloader)

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
ov_q_model = trainer.quantize(model, accelerator='openvino', calib_dataloader=dataloader)
# run simple prediction with transparent acceleration
y_hat = ov_q_model(x)

# validate, predict, test in Trainer also support acceleration
trainer.validate(ov_q_model, dataloader)
trainer.test(ov_q_model, dataloader)
trainer.predict(ov_q_model, dataloader)
```
Same as using ONNXRuntime accelerator, it equals to converting the model from PyTorch to OpenVINO firstly and then doing quantization on the converted OpenVINO model:
```python
ov_model = Trainer.trace(model, accelerator='openvino', input_sample=x):
ov_q_model = trainer.quantize(ov_model, accelerator='onnxruntime', calib_dataloader=dataloader)

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
trainer.quantize(model,
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
trainer.quantize(model,
                 precision='int8',
                 accelerator=`openvino`,
                 calib_dataloader= dataloader,
                 metric=Accuracy()
                 accuracy_criterion={'relative': 0.01, 'higher_is_better': True},
                 approach='static',
                 max_trials=10,
                ):
```