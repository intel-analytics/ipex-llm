# Quantize PyTorch Model for Inference using Intel Neural Compressor

With Intel Neural Compressor (INC) as quantization engine, you can apply `Trainer.quantize` API to realize post-training quantization on your PyTorch `nn.Module`. It only takes a few lines.

Let's take an [ResNet-18 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) pretrained on ImageNet dataset and finetuned on [OxfordIIITPet dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html) as an example:

```python
from torchvision.models import resnet18

model = resnet18(pretrained=True)
_, train_dataset, val_dataset = finetune_pet_dataset(model)
```
> _The full definition of function_ `finetune_pet_dataset` _could be found_ [_here_](https://github.com/intel-analytics/BigDL/blob/1122517e0f919d7880e8f9848a476ce889ed7183/python/nano/tutorial/inference/pytorch/pytorch_quantization.py#L33).

Then we set it in evaluation mode:

```python
model.eval()
```
To enable quantization using INC for inference, you could simply **import BigDL-Nano Trainer, and use Trainer to quantize your PyTorch model**. `Trainer.quantize` also support ONNXRuntime acceleration at the meantime through specifying `accelerator='onnxruntime'`:

```eval_rst
.. tabs::

    .. tab:: Default

        .. code-block:: python

            from bigdl.nano.pytorch import Trainer

            q_model = Trainer.quantize(model, 
                                       calib_dataloader=DataLoader(train_dataset, batch_size=32))

    .. tab:: ONNXRuntime

        .. code-block:: python

                from bigdl.nano.pytorch import Trainer

                q_model = Trainer.quantize(model,
                                           accelerator='onnxruntime',
                                           calib_dataloader=DataLoader(train_dataset, batch_size=32))
```

```eval_rst
.. note::
    ``Trainer`` will by default quantize your PyTorch ``nn.Module`` through **static** post-training quantization. For this case, ``calib_dataloader`` (for calibration data) is required. Batch size is not important to ``calib_dataloader``, as it intends to read 100 data. And there could be no label in calibration data.

    If you would like to implement dynamic post-training quantization, you could set parameter ``approach='dynamic'``. In this case, ``calib_dataloader`` should be ``None``. Compared to dynamic quantization, static quantization could lead to faster inference as it eliminates the data conversion costs between layers.
```

You could then do the normal inference steps with the quantized model:

```python
x = torch.stack([val_dataset[0][0], val_dataset[1][0]])
# use the quantized model here
y_hat = q_model(x)
predictions = y_hat.argmax(dim=1)
print(predictions)
```

A short runnable example to demonstrate this functionality can be found [here](https://github.com/intel-analytics/BigDL/blob/main/python/nano/tutorial/inference/pytorch/pytorch_quantization.py) with no runtime acceleration, and [here](https://github.com/intel-analytics/BigDL/blob/main/python/nano/tutorial/inference/pytorch/pytorch_quantization_onnx.py) with extra ONNXRuntime acceleration.

```eval_rst
.. card:: Relative Readings

    * `How to install BigDL-Nano <../Overview/nano.html#install>`_
    * `How to install BigDL-Nano in Google Colab <install_in_colab.html>`_
    * How to load/save optimized models
```