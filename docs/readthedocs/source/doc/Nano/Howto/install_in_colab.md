# Install BigDL-Nano in Google Colab

In this guide, we will show you how to install BigDL-Nano in Google Colab, and the solutions to possible version conflicts caused by pre-installed packages in Colab.

Please select the corresponding section to follow for your specific usage. 

## PyTorch
For PyTorch users, you could add the following code snippet to your notebook to install BigDL-Nano and set environment variables for acceleration based on your current hardware:

```eval_rst
.. tabs::

    .. tab:: Latest

        .. code-block:: python

            !pip install bigdl-nano[pytorch]
            !source bigdl-nano-init

    .. tab:: Nightly-Built

        .. code-block:: python

            !pip install --pre --upgrade bigdl-nano[pytorch]
            !source bigdl-nano-init
```

To avoid version conflicts caused by `trochtext`, you should uninstall it:

```python
!pip uninstall -y torchtext
```

### ONNXRuntime
To enable ONNXRuntime acceleration, you need to install corresponding onnx packages:

```python
!pip install onnx onnxruntime
```

### OpenVINO
To enable OpenVINO acceleration, you need to install the OpenVINO toolkit:

```python
!pip install openvino-dev
# Please remember to restart runtime to use packages with newly-installed version
```


```eval_rst
.. note::
    If you meet ``ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`` when using ``Trainer.trace`` function, you could try to solve it by upgrading ``numpy`` through:
    
    .. code-block:: python

            !pip install --upgrade numpy
            # Please remember to restart runtime to use numpy with newly-installed version
```