# Accelerate PyTorch Inference using OpenVINO

To accelerate your PyTorch inference pipeline through OpenVINO as backend engine, BigDL-Nano provides you with `Trainer.trace(accelerator='openvino')` API. Based on that, the optimization could be implemented by applying very few lines of code changes.

Let us suppose we have a [ResNet-18 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) pretrained on ImageNet dataset, and would like to do the inference on a randomly generated test dataset. The model would be like: 

```python
from torchvision.models import resnet18

model_ft = resnet18(pretrained=True)
# set the model in evaluation (inference) mode
model_ft.eval()
```

To enable OpenVINO acceleration for your PyTorch inference pipeline, **the only change you need to made is to import BigDL-Nano Trainer, and trace your PyTorch model to convert it into an OpenVINO accelerated module for inference**:
```python
import torch
from bigdl.nano.pytorch import Trainer

ov_model = Trainer.trace(model_ft,
                         accelerator="openvino",
                         input_sample=torch.rand(1, 3, 224, 224))
```
```eval_rst
.. note::
    ``input_sample`` is the parameter for OpenVINO accelerator to know the **shape** of the model input. So both the batch size and the specific values are not important to ``input_sample``. 
    
    If we want our test dataset to consist of images with :math:`224 \times 224` pixels, we could use ``torch.rand(1, 3, 224, 224)`` for ``input_sample`` here.
```

You could then do the normal inference steps with the model optimized by OpenVINO:

```python
x = torch.rand(2, 3, 224, 224)
# use the optimized model here
y_hat = ov_model(x)
predictions = y_hat.argmax(dim=1)
print(predictions)
```

An executable version of the example above could be found [here](https://github.com/intel-analytics/BigDL/blob/main/python/nano/tutorial/inference/pytorch/pytorch_inference_openvino.py).

```eval_rst
.. card:: Relative Readings

    * `How to install BigDL-Nano <../Overview/nano.html#install>`_
    * `How to install BigDL-Nano in Google Colab <install_in_colab.html>`_
    * How to load/save optimized models
```