# Quantize PyTorch Model for Inference using Post-training Optimization Tools

As Post-training Optimization Tools (POT) is provided by OpenVINO toolkit, OpenVINO acceleration will be enabled in the meantime when using POT for quantization. You can apply `Trainer.quantize(accelerator='openvino')` API to use POT for your PyTorch `nn.Module`. It only takes a few lines.

Let's take an [ResNet-18 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) pretrained on ImageNet dataset and finetuned on [OxfordIIITPet dataset](https://github.com/intel-analytics/BigDL/blob/main/python/nano/tutorial/inference/pytorch/pytorch_quantization_openvino.py) as an example:

```python
from torchvision.models import resnet18

model = resnet18(pretrained=True)
_, train_dataset, val_dataset = finetune_pet_dataset(model)
```
> _The full definition of function_ `finetune_pet_dataset` _could be found_ [_here_](https://github.com/intel-analytics/BigDL/blob/dc44a2dc9d08c91b3af4e00e21bc627e63ea1c6c/python/nano/tutorial/inference/pytorch/pytorch_quantization_openvino.py#L33).

Then we set it in evaluation mode:

```python
model.eval()
```
To enable quantization using POT for inference, you could simply **import BigDL-Nano Trainer, and use Trainer to quantize your PyTorch model**:

```python
from bigdl.nano.pytorch import Trainer

q_model = Trainer.quantize(model,
                           accelerator='openvino',
                           calib_dataloader=DataLoader(train_dataset, batch_size=32))
```

```eval_rst
.. note::
    For POT, only **static** post-training quantization is supported. So ``calib_dataloader`` (for calibration data) is always required when ``accelerator='openvino'``. 
    
    For ``calib_dataloader``, batch size is not important as it intends to read 100 samples. And there could be no label in calibration data.

    Please refer to `API documentation <../../PythonAPI/Nano/pytorch.html#bigdl.nano.pytorch.Trainer.quantize>`_ for more information on ``Trainer.quantize``.
```

You could then do the normal inference steps with the quantized model:

```python
x = torch.stack([val_dataset[0][0], val_dataset[1][0]])
# use the quantized model here
y_hat = q_model(x)
predictions = y_hat.argmax(dim=1)
print(predictions)
```

A short runnable example to demonstrate this functionality can be found [here](https://github.com/intel-analytics/BigDL/blob/main/python/nano/tutorial/inference/pytorch/pytorch_quantization.py).


```eval_rst
.. card:: Related Readings

    * `How to install BigDL-Nano <../Overview/nano.html#install>`_
    * `How to install BigDL-Nano in Google Colab <install_in_colab.html>`_
    * How to load/save optimized models
```