# Accelerate PyTorch Inference using ONNXRuntime

You can use ``Trainer.trace(accelerator='onnxruntime')`` API to enable the ONNXRuntime acceleration for PyTorch inference. It only takes a few lines.

Let's take an [ResNet-18 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) pretrained on ImageNet dataset as an example. First, we load the model:

```python
from torchvision.models import resnet18

model_ft = resnet18(pretrained=True)
```

Then we set it in evaluation mode:

```python
model_ft.eval()
```

To enable ONNXRuntime acceleration for your PyTorch inference pipeline, **the only change you need to made is to import BigDL-Nano Trainer, and trace your PyTorch model to convert it into an ONNXRuntime accelerated module for inference**:
```python
import torch
from bigdl.nano.pytorch import Trainer

ort_model = Trainer.trace(model_ft,
                          accelerator="onnxruntime",
                          input_sample=torch.rand(1, 3, 224, 224))
```
```eval_rst
.. note::
    ``input_sample`` is the parameter for ONNXRuntime accelerator to know the **shape** of the model input. So both the batch size and the specific values are not important to ``input_sample``. 
    
    If we want our test dataset to consist of images with :math:`224 \times 224` pixels, we could use ``torch.rand(1, 3, 224, 224)`` for ``input_sample`` here.
```

You could then do the normal inference steps with the model optimized by ONNXRuntime:

```python
x = torch.rand(2, 3, 224, 224)
# use the optimized model here
y_hat = ort_model(x)
predictions = y_hat.argmax(dim=1)
print(predictions)
```

A short runnable example to demonstrate this functionality can be found [here](https://github.com/intel-analytics/BigDL/blob/main/python/nano/tutorial/inference/pytorch/pytorch_inference_onnx.py).

```eval_rst
.. card:: Relative Readings

    * `How to install BigDL-Nano <../Overview/nano.html#install>`_
    * `How to install BigDL-Nano in Google Colab <install_in_colab.html>`_
    * How to load/save optimized models
```