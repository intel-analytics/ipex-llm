TSDataset
===========

TSDataset
----------------------------------------

Time series data is a special data formulation with specific operations. TSDataset is an abstract of time series dataset, which provides various data processing operations (e.g. impute, deduplicate, resample, scale/unscale, roll) and feature engineering methods (e.g. datetime feature, aggregation feature). Cascade call is supported for most of the methods.
TSDataset can be initialized from a pandas dataframe and be converted to a pandas dataframe or numpy ndarray.

.. automodule:: bigdl.chronos.data.tsdataset
    :members:
    :undoc-members:
    :show-inheritance:

XShardsTSDataset
----------------------------------------

Time series data is a special data formulation with specific operations. XShardsTSDataset is an abstract of time series dataset, which provides various data processing operations (e.g. impute, deduplicate, resample, scale/unscale, roll) and feature engineering methods (e.g. datetime feature, aggregation feature). Cascade call is supported for most of the methods.
XShardsTSDataset can be initialized from xshards of pandas dataframe and be converted to xshards of numpy in an distributed and parallized fashion.

.. automodule:: bigdl.chronos.data.experimental.xshards_tsdataset
    :members:
    :undoc-members:
    :show-inheritance:


Built-in Dataset
--------------------------------------------

Built-in dataset can be downloaded and preprocessed by this function. Train, validation and test split is also supported.

.. automodule:: bigdl.chronos.data.repo_dataset
    :members:
