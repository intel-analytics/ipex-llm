Nano HPO API
==================

Search Space
---------------------------

.. autoclass:: bigdl.nano.automl.hpo.space.Categorical

.. autoclass:: bigdl.nano.automl.hpo.space.Real

.. autoclass:: bigdl.nano.automl.hpo.space.Int

.. autoclass:: bigdl.nano.automl.hpo.space.Bool


HPO for Tensorflow
---------------------------

.. autoclass:: bigdl.nano.automl.tf.keras.Model.Model
    :members: fit, compile
    :inherited-members: search, search_summary


.. autoclass:: bigdl.nano.automl.tf.keras.Sequential.Sequential
    :members: fit, compile, search, search_summary


HPO for PyTorch
---------------------------

.. autoclass:: bigdl.nano.pytorch.Trainer
    :members: search, search_summary
    :undoc-members:

