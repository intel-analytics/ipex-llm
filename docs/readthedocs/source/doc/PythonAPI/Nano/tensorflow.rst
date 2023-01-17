Nano Tensorflow API
==================

bigdl.nano.tf.keras
---------------------------

.. autoclass:: bigdl.nano.tf.keras.Model
    :members: fit, quantize, trace
    :undoc-members:

.. autoclass:: bigdl.nano.tf.keras.Sequential
    :members:
    :undoc-members:
    :inherited-members: Sequential

.. autoclass:: bigdl.nano.tf.keras.layers.Embedding
    :members:
    :undoc-members:

bigdl.nano.tf.optimizers
---------------------------
.. autoclass:: bigdl.nano.tf.optimizers.SparseAdam
    :members: 
    :undoc-members:

bigdl.nano.tf.keras.InferenceOptimizer
---------------------------------------
.. autoclass:: bigdl.nano.tf.keras.InferenceOptimizer
    :members:
    :undoc-members:
    :exclude-members: ALL_INFERENCE_ACCELERATION_METHOD

Patch API
---------------------------

.. autofunction:: bigdl.nano.tf.patch_tensorflow

.. autofunction:: bigdl.nano.tf.unpatch_tensorflow

.. autofunction:: bigdl.nano.tf.keras.nano_bf16