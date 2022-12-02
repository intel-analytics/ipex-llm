Nano How-to Guides
=========================
.. note::
    This page is still a work in progress. We are adding more guides.

In Nano How-to Guides, you could expect to find multiple task-oriented, bite-sized, and executable examples. These examples will show you various tasks that BigDL-Nano could help you accomplish smoothly.

Training Optimization
-------------------------

PyTorch Lightning
~~~~~~~~~~~~~~~~~~~~~~~~~
* `How to accelerate a PyTorch Lightning application on training workloads through Intel® Extension for PyTorch* <Training/PyTorchLightning/accelerate_pytorch_lightning_training_ipex.html>`_
* `How to accelerate a PyTorch Lightning application on training workloads through multiple instances <Training/PyTorchLightning/accelerate_pytorch_lightning_training_multi_instance.html>`_
* `How to use the channels last memory format in your PyTorch Lightning application for training <Training/PyTorchLightning/pytorch_lightning_training_channels_last.html>`_
* `How to conduct BFloat16 Mixed Precision training in your PyTorch Lightning application <Training/PyTorchLightning/pytorch_lightning_training_bf16.html>`_
* `How to accelerate a computer vision data processing pipeline <Training/PyTorchLightning/pytorch_lightning_cv_data_pipeline.html>`_

TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~
* `How to accelerate a TensorFlow Keras application on training workloads through multiple instances <Training/TensorFlow/accelerate_tensorflow_training_multi_instance.html>`_
* |tensorflow_training_embedding_sparseadam_link|_

.. |tensorflow_training_embedding_sparseadam_link| replace:: How to optimize your model with a sparse ``Embedding`` layer and ``SparseAdam`` optimizer
.. _tensorflow_training_embedding_sparseadam_link: Training/TensorFlow/tensorflow_training_embedding_sparseadam.html

General
~~~~~~~~~~~~~~~~~~~~~~~~~
* `How to choose the number of processes for multi-instance training <Training/General/choose_num_processes_training.html>`_

Inference Optimization
-------------------------

OpenVINO
~~~~~~~~~~~~~~~~~~~~~~~~~

* `How to run inference on OpenVINO model <Inference/OpenVINO/openvino_inference.html>`_
* `How to run asynchronous inference on OpenVINO model <Inference/OpenVINO/openvino_inference_async.html>`_

.. toctree::
    :maxdepth: 1
    :hidden:

    Inference/OpenVINO/openvino_inference
    Inference/OpenVINO/openvino_inference_async

PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~

* `How to accelerate a PyTorch inference pipeline through ONNXRuntime <Inference/PyTorch/accelerate_pytorch_inference_onnx.html>`_
* `How to accelerate a PyTorch inference pipeline through OpenVINO <Inference/PyTorch/accelerate_pytorch_inference_openvino.html>`_
* `How to accelerate a PyTorch inference pipeline through JIT/IPEX <Inference/PyTorch/accelerate_pytorch_inference_jit_ipex.html>`_
* `How to quantize your PyTorch model for inference using Intel Neural Compressor <Inference/PyTorch/quantize_pytorch_inference_inc.html>`_
* `How to quantize your PyTorch model for inference using OpenVINO Post-training Optimization Tools <Inference/PyTorch/quantize_pytorch_inference_pot.html>`_
* `How to save and load optimized IPEX model <Inference/PyTorch/pytorch_save_and_load_ipex.html>`_
* `How to save and load optimized JIT model <Inference/PyTorch/pytorch_save_and_load_jit.html>`_
* `How to save and load optimized ONNXRuntime model <Inference/PyTorch/pytorch_save_and_load_onnx.html>`_
* `How to save and load optimized OpenVINO model <Inference/PyTorch/pytorch_save_and_load_openvino.html>`_
* `How to find accelerated method with minimal latency using InferenceOptimizer <Inference/PyTorch/inference_optimizer_optimize.html>`_

Install
-------------------------
* `How to install BigDL-Nano in Google Colab <install_in_colab.html>`_
* `How to install BigDL-Nano on Windows <windows_guide.html>`_