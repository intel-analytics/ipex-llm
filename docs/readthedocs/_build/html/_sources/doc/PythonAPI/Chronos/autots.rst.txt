AutoTS (deprecated)
=====================

.. warning::
    The API in this page will be deprecated soon. Please refer to our new AutoTS API.

AutoTSTrainer
----------------------------------------

AutoTSTrainer trains a time series pipeline (including data processing, feature engineering, and model) with AutoML.

.. autoclass:: bigdl.chronos.autots.deprecated.forecast.AutoTSTrainer
    :members:
    :show-inheritance:


TSPipeline
----------------------------------------

A pipeline for time series forecasting.

.. autoclass:: bigdl.chronos.autots.deprecated.forecast.TSPipeline
    :members:
    :show-inheritance:


Recipe
----------------------------------------

Recipe is used for search configuration for AutoTSTrainer.

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.SmokeRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.MTNetSmokeRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.TCNSmokeRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.PastSeqParamHandler
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.GridRandomRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.LSTMSeq2SeqRandomRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.LSTMGridRandomRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.Seq2SeqRandomRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.MTNetGridRandomRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.TCNGridRandomRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.RandomRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.BayesRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.XgbRegressorGridRandomRecipe
    :members:
    :show-inheritance:

.. autoclass:: bigdl.chronos.autots.deprecated.config.recipe.XgbRegressorSkOptRecipe
    :members:
    :show-inheritance: