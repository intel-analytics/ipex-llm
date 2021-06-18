TSDataset
===========

chronos.data.tsdataset
----------------------------------------

Time series data is a special data formulation with specific operations. TSDataset is an abstract of time series dataset, which provides various data processing operations (e.g. impute, deduplicate, resample, scale/unscale, roll) and feature engineering methods (e.g. datetime feature, aggregation feature). Cascade call is supported for most of the methods.
TSDataset can be initialized from a pandas dataframe and be converted to a pandas dataframe or numpy ndarray.

.. automodule:: zoo.chronos.data.tsdataset
    :members:
    :undoc-members:
    :show-inheritance:
