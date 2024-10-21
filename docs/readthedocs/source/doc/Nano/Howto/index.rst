Nano How-to Guides
=========================
.. note::
    This page is still a work in progress. We are adding more guides.

In Nano How-to Guides, you could expect to find multiple task-oriented, bite-sized, and executable examples. These examples will show you various tasks that BigDL-Nano could help you accomplish smoothly.

Preprocessing Optimization
---------------------------

PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~
* `How to accelerate a computer vision data processing pipeline <Preprocessing/PyTorch/accelerate_pytorch_cv_data_pipeline.html>`_


Training Optimization
-------------------------

PyTorch Lightning
~~~~~~~~~~~~~~~~~~~~~~~~~
* `How to accelerate a PyTorch Lightning application on training workloads through Intel® Extension for PyTorch* <Training/PyTorchLightning/accelerate_pytorch_lightning_training_ipex.html>`_
* `How to accelerate a PyTorch Lightning application on training workloads through multiple instances <Training/PyTorchLightning/accelerate_pytorch_lightning_training_multi_instance.html>`_
* `How to use the channels last memory format in your PyTorch Lightning application for training <Training/PyTorchLightning/pytorch_lightning_training_channels_last.html>`_
* `How to conduct BFloat16 Mixed Precision training in your PyTorch Lightning application <Training/PyTorchLightning/pytorch_lightning_training_bf16.html>`_

PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~
* |convert_pytorch_training_torchnano|_
* |use_nano_decorator_pytorch_training|_
* `How to accelerate a PyTorch application on training workloads through Intel® Extension for PyTorch* <Training/PyTorch/accelerate_pytorch_training_ipex.html>`_
* `How to accelerate a PyTorch application on training workloads through multiple instances <Training/PyTorch/accelerate_pytorch_training_multi_instance.html>`_
* `How to use the channels last memory format in your PyTorch application for training <Training/PyTorch/pytorch_training_channels_last.html>`_
* `How to conduct BFloat16 Mixed Precision training in your PyTorch application <Training/PyTorch/accelerate_pytorch_training_bf16.html>`_

.. |use_nano_decorator_pytorch_training| replace:: How to accelerate your PyTorch training loop with ``@nano`` decorator
.. _use_nano_decorator_pytorch_training: Training/PyTorch/use_nano_decorator_pytorch_training.html
.. |convert_pytorch_training_torchnano| replace:: How to convert your PyTorch training loop to use ``TorchNano`` for acceleration
.. _convert_pytorch_training_torchnano: Training/PyTorch/convert_pytorch_training_torchnano.html

TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~
* `How to accelerate a TensorFlow Keras application on training workloads through multiple instances <Training/TensorFlow/accelerate_tensorflow_training_multi_instance.html>`_
* |tensorflow_training_embedding_sparseadam_link|_
* `How to conduct BFloat16 Mixed Precision training in your TensorFlow Keras application <Training/TensorFlow/tensorflow_training_bf16.html>`_
* `How to accelerate TensorFlow Keras customized training loop through multiple instances <Training/TensorFlow/tensorflow_custom_training_multi_instance.html>`_

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
* `How to accelerate a PyTorch / TensorFlow inference pipeline on Intel GPUs through OpenVINO <Inference/OpenVINO/accelerate_inference_openvino_gpu.html>`_

PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~

* `How to find accelerated method with minimal latency using InferenceOptimizer <Inference/PyTorch/inference_optimizer_optimize.html>`_
* `How to accelerate a PyTorch inference pipeline through ONNXRuntime <Inference/PyTorch/accelerate_pytorch_inference_onnx.html>`_
* `How to accelerate a PyTorch inference pipeline through OpenVINO <Inference/PyTorch/accelerate_pytorch_inference_openvino.html>`_
* `How to accelerate a PyTorch inference pipeline through JIT/IPEX <Inference/PyTorch/accelerate_pytorch_inference_jit_ipex.html>`_
* `How to quantize your PyTorch model in INT8 for inference using Intel Neural Compressor <Inference/PyTorch/quantize_pytorch_inference_inc.html>`_
* `How to quantize your PyTorch model in INT8 for inference using OpenVINO Post-training Optimization Tools <Inference/PyTorch/quantize_pytorch_inference_pot.html>`_
* `How to enable automatic context management for PyTorch inference on Nano optimized models <Inference/PyTorch/pytorch_context_manager.html>`_
* `How to save and load optimized ONNXRuntime model <Inference/PyTorch/pytorch_save_and_load_onnx.html>`_
* `How to save and load optimized OpenVINO model <Inference/PyTorch/pytorch_save_and_load_openvino.html>`_
* `How to save and load optimized JIT model <Inference/PyTorch/pytorch_save_and_load_jit.html>`_
* `How to save and load optimized IPEX model <Inference/PyTorch/pytorch_save_and_load_ipex.html>`_
* `How to accelerate a PyTorch inference pipeline through multiple instances <Inference/PyTorch/multi_instance_pytorch_inference.html>`_
* `How to accelerate a PyTorch inference pipeline using Intel ARC series dGPU <Inference/PyTorch/accelerate_pytorch_inference_gpu.html>`_
* `How to accelerate PyTorch inference using async multi-stage pipeline <Inference/PyTorch/accelerate_pytorch_inference_async_pipeline.html>`_

TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~
* `How to accelerate a TensorFlow inference pipeline through ONNXRuntime <Inference/TensorFlow/accelerate_tensorflow_inference_onnx.html>`_
* `How to accelerate a TensorFlow inference pipeline through OpenVINO <Inference/TensorFlow/accelerate_tensorflow_inference_openvino.html>`_
* `How to conduct BFloat16 Mixed Precision inference in a TensorFlow Keras application <Inference/TensorFlow/tensorflow_inference_bf16.html>`_
* `How to save and load optimized ONNXRuntime model in TensorFlow <Inference/TensorFlow/tensorflow_save_and_load_onnx.html>`_
* `How to save and load optimized OpenVINO model in TensorFlow <Inference/TensorFlow/tensorflow_save_and_load_openvino.html>`_

AutoML
-------------------------
* `How to search the best hyper-parameters for TensorFlow models via Nano HPO <hpo/use_hpo_tune_hyperparameters_tensorflow.html>`_

Install
-------------------------
* `How to install BigDL-Nano in Google Colab <Install/install_in_colab.html>`_
* `How to install BigDL-Nano on Windows <Install/windows_guide.html>`_