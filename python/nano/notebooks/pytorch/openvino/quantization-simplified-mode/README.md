# Simplified Post-Training Quantization of Image Classification Models with OpenVINO™ 
This tutorial was adapted from https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/114-quantization-simplified-mode. Here, we use OpenVINO APIs provided by BigDL Nano instead to simplify the original tutorial.

This tutorial demostrates how to perform INT8 quantization with an image classification model using the [Post-Training Optimization
Tool Simplified Mode](https://docs.openvino.ai/latest/pot_docs_simplified_mode.html) (part of [OpenVINO](https://docs.openvino.ai/)). We use [ResNet20](https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py) model and [Cifar10](http://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) dataset.

The code in this tutorial is designed to extend to custom models and datasets. It consists of the following steps:
- Download and prepare the ResNet20 model and calibration dataset
- Prepare the model for quantization
- Compress the model using the simplified mode
- Compare performance of the original and quantized models
- Demonstrate the results of the optimized model
