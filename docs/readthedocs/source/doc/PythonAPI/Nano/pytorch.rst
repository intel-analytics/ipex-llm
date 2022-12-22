Nano PyTorch API
==================

bigdl.nano.pytorch.Trainer
---------------------------

.. autoclass:: bigdl.nano.pytorch.Trainer
    :members:
    :undoc-members:
    :exclude-members: accelerator_connector, checkpoint_connector, reload_dataloaders_every_n_epochs, limit_val_batches, logger, logger_connector, state

bigdl.nano.pytorch.InferenceOptimizer
---------------------------

.. autoclass:: bigdl.nano.pytorch.InferenceOptimizer
    :members:
    :undoc-members:
    :exclude-members: ALL_INFERENCE_ACCELERATION_METHOD, DEFAULT_INFERENCE_ACCELERATION_METHOD
    :inherited-members:

TorchNano API
---------------------------

.. automodule:: bigdl.nano.pytorch.torch_nano
    :members:
    :undoc-members:
    :exclude-members: run

Patch API
---------------------------

.. automodule:: bigdl.nano.pytorch.dispatcher
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: bigdl.nano.pytorch.patching.gpu_cpu.gpu_cpu
    :members: patch_cuda, unpatch_cuda
    :show-inheritance:

.. automodule:: bigdl.nano.pytorch.patching.dtype_patching.dtype_patching
    :members: patch_dtype
    :show-inheritance:
