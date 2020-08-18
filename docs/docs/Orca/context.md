We support users to make configurations when using the functionalities of [Project Orca](overview.md) via `OrcaContext`.

Import OrcaContext using `from from zoo.orca import OrcaContext` and then you can choose to modify the following options:

* log_output
```python
OrcaContext.log_output = False
OrcaContext.log_output = True
```
Whether to redirect Spark driver JVM's stdout and stderr to the current python process. 
This is useful when running Analytics Zoo in jupyter notebook.
Default to be False. Needs to be set before initializing SparkContext.

* pandas_read_backend
```python
OrcaContext.pandas_read_backend = "spark"
OrcaContext.pandas_read_backend = "pandas"
```
The backend for reading csv/json files. Either "spark" or "pandas". 
"spark" backend would call `spark.read` and "pandas" backend would call `pandas.read`. 
Default to be "spark".

* serialize_data_creation
```python
OrcaContext.serialize_data_creation = False
OrcaContext.serialize_data_creation = True
```
Whether add a file lock to the data loading process for PyTorch Horovod training. 
This would be useful when you run multiple workers on a single node to download data to the same destination. 
Default to be False.
