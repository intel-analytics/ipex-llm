Nano PyTorch API
==================

bigdl.nano.pytorch.Trainer
---------------------------

.. autoclass:: bigdl.nano.pytorch.Trainer
    :members:
    :undoc-members:
    :exclude-members: accelerator_connector, checkpoint_connector, reload_dataloaders_every_n_epochs, limit_val_batches, logger, logger_connector, state

bigdl.nano.pytorch.InferenceOptimizer
---------------------------------------

.. autoclass:: bigdl.nano.pytorch.InferenceOptimizer
    :members:
    :undoc-members:
    :exclude-members: ALL_INFERENCE_ACCELERATION_METHOD, DEFAULT_INFERENCE_ACCELERATION_METHOD, method
    :inherited-members:

TorchNano API
---------------------------

.. autoclass:: bigdl.nano.pytorch.TorchNano
    :members:
    :undoc-members:
    :exclude-members: run

.. autofunction:: bigdl.nano.pytorch.nano

Patch API
---------------------------

.. autofunction:: bigdl.nano.pytorch.patch_torch

.. autofunction:: bigdl.nano.pytorch.unpatch_torch

.. autofunction:: bigdl.nano.pytorch.patching.patch_cuda

.. autofunction:: bigdl.nano.pytorch.patching.unpatch_cuda

.. autofunction:: bigdl.nano.pytorch.patching.patch_dtype
    
.. autofunction:: bigdl.nano.pytorch.patching.patch_encryption