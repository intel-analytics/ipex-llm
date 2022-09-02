# Install BigDL-Nano in Google Colab

```eval_rst
.. note::
    This page is still a work in progress.
```

In this guide, we will show you how to install BigDL-Nano in Google Colab, and the solutions to possible version conflicts caused by pre-installed packages in Colab hosted runtime.

Please select the corresponding section to follow for your specific usage. 

## PyTorch
For PyTorch users, you need to install BigDL-Nano for PyTorch first:

```eval_rst
.. tabs::

    .. tab:: Latest

        .. code-block:: python

            !pip install bigdl-nano[pytorch]

    .. tab:: Nightly-Built

        .. code-block:: python

            !pip install --pre --upgrade bigdl-nano[pytorch]
```

```eval_rst
.. warning::
    For Google Colab hosted runtime, ``source bigdl-nano-init`` is hardly to take effect as environment variables need to be set before jupyter kernel is started.
```

To avoid version conflicts caused by `torchtext`, you should uninstall it:

```python
!pip uninstall -y torchtext
```

### ONNXRuntime
To enable ONNXRuntime acceleration, you need to install corresponding onnx packages:

```python
!pip install onnx onnxruntime
```

### OpenVINO / Post-training Optimization Tools (POT)
To enable OpenVINO acceleration, or use POT for quantization, you need to install the OpenVINO toolkit:

```python
!pip install openvino-dev
# Please remember to restart runtime to use packages with newly-installed version
```

```eval_rst
.. note::
    If you meet ``ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`` when using ``Trainer.trace`` or ``Trainer.quantize`` function, you could try to solve it by upgrading ``numpy`` through:
    
    .. code-block:: python

            !pip install --upgrade numpy
            # Please remember to restart runtime to use numpy with newly-installed version
```

### Intel Neural Compressor (INC)
To use INC as your quantization backend, you need to install it:

```eval_rst
.. tabs::

    .. tab:: With no Extra Runtime Acceleration

        .. code-block:: python

            !pip install neural-compressor==1.11.0

    .. tab:: With Extra ONNXRuntime Acceleration

        .. code-block:: python

            !pip install neural-compressor==1.11.0 onnx onnxruntime onnxruntime_extensions
```