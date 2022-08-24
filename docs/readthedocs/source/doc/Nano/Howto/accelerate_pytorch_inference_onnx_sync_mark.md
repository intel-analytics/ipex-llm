# Accelerate PyTorch Inference using ONNXRuntime

You can use ``Trainer.trace(accelerator='onnxruntime')`` API to enable the ONNXRuntime acceleration for PyTorch inference. It only takes a few lines.

Let's take an [ResNet-18 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) pretrained on ImageNet dataset as an example. First, we load the model:

```eval_rst
.. literalinclude:: ../../../../../../python/nano/tutorial/inference/pytorch/pytorch_inference_onnx.py
    :language: python
    :start-after: # >>> import a pretrained ResNet-18 model >>>
    :end-before: # <<< import a pretrained ResNet-18 model <<<
    :dedent: 4
```

Then we set it in evaluation mode:

```eval_rst
.. literalinclude:: ../../../../../../python/nano/tutorial/inference/pytorch/pytorch_inference_onnx.py
    :language: python
    :start-after: # >>> set the model in evaluation mode >>>
    :end-before: # <<< set the model in evaluation mode <<<
    :dedent: 4
```

To enable ONNXRuntime acceleration for your PyTorch inference pipeline, **the only change you need to made is to import BigDL-Nano Trainer, and trace your PyTorch model to convert it into an ONNXRuntime accelerated module for inference**:

```eval_rst
.. literalinclude:: ../../../../../../python/nano/tutorial/inference/pytorch/pytorch_inference_onnx.py
    :language: python
    :start-after: # >>> import BigDL-Nano Trainer, and trace the model for acceleration >>>
    :end-before: # <<< import BigDL-Nano Trainer, and trace the model for acceleration <<<
    :dedent: 4
```
```eval_rst
.. note::
    ``input_sample`` is the parameter for ONNXRuntime accelerator to know the **shape** of the model input. So both the batch size and the specific values are not important to ``input_sample``. If we want our test dataset to consist of images with :math:`224 \times 224` pixels, we could use ``torch.rand(1, 3, 224, 224)`` for ``input_sample`` here.

    ``input_sample`` is not required if you have used an instance of ``Trainer`` to fit your model before, or the model is a ``LightningModule`` with any dataloader attached.

    Please refer to `API documentation <../../PythonAPI/Nano/pytorch.html#bigdl.nano.pytorch.Trainer.trace>`_ for more information on ``Trainer.trace``.
```

You could then do the normal inference steps with the model optimized by ONNXRuntime:

```eval_rst
.. literalinclude:: ../../../../../../python/nano/tutorial/inference/pytorch/pytorch_inference_onnx.py
    :language: python
    :start-after: # >>> do normal inference steps with the optimized model >>>
    :end-before: # <<< do normal inference steps with the optimized model <<<
    :dedent: 4
```

A short runnable example to demonstrate this functionality can be found [here](https://github.com/intel-analytics/BigDL/blob/main/python/nano/tutorial/inference/pytorch/pytorch_inference_onnx.py).

```eval_rst
.. card:: Related Readings

    * `How to install BigDL-Nano <../Overview/nano.html#install>`_
    * `How to install BigDL-Nano in Google Colab <install_in_colab.html>`_
    * How to load/save optimized models
```