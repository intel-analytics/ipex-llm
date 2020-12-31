### **Initialization and Termination**
As a starting point of [Project Orca](overview.md), you need to call `init_orca_context` to create or get a SparkContext for your Spark cluster (and launch Ray services
across the cluster if necessary). When your application finishes, you need to call `stop_orca_context` to stop the SparkContext (and stop Ray services across the cluster if necessary).

```python
from zoo.orca import init_orca_context, stop_orca_context

# At the very beginning:
sc = init_orca_context(cluster_mode="local", cores=2, memory="2g", num_nodes=1,
                       init_ray_on_spark=False, **kwargs)

# Your application goes after init_orca_context.

# When your application finishes:
stop_orca_context()
```

**Arguments for** `init_orca_context`:

* `cluster_mode`: The mode for the Spark cluster. One of "local", "yarn-client", "k8s-client", "standalone" and "spark-submit". Default to be "local". 

For "spark-submit", you are supposed to use spark-submit to submit the application. In this case, please set the Spark configurations through command line options or
the properties file. You need to use "spark-submit" for yarn-cluster or k8s-cluster mode. To make things easier, you are recommended to use the 
launch [scripts](https://github.com/intel-analytics/analytics-zoo/tree/master/scripts) we provide.

For other cluster modes, you are recommended to install and run analytics-zoo through pip, which is more convenient.

* `cores`: The number of cores to be used on each node. Default to be 2.
* `memory`: The memory allocated for each node. Default to be '2g'.
* `num_nodes`: The number of nodes to be used in the cluster. Default to be 1. For Spark local, num_nodes should always be 1 and you don't need to change it.
* `init_ray_on_spark`: Whether to launch Ray services across the cluster. Default to be False and in this case the Ray cluster would be launched lazily when Ray is involved in Project Orca.
* `kwargs`: The extra keyword arguments used for creating SparkContext and launching Ray if any.


---
### **Extra Configurations**
Users can make extra configurations when using the functionalities of Project Orca via `OrcaContext`.

Import OrcaContext using `from from zoo.orca import OrcaContext` and then you can choose to modify the following options:

* log_output
```python
OrcaContext.log_output = False  # Default
OrcaContext.log_output = True
```
Whether to redirect Spark driver JVM's stdout and stderr to the current python process. 
This is useful when running Analytics Zoo in jupyter notebook.
Default to be False. Needs to be set before initializing SparkContext.

* pandas_read_backend
```python
OrcaContext.pandas_read_backend = "spark"  # Default
OrcaContext.pandas_read_backend = "pandas"
```
The backend for reading csv/json files. Either "spark" or "pandas". 
"spark" backend would call `spark.read` and "pandas" backend would call `pandas.read`. 
Default to be "spark".

* serialize_data_creation
```python
OrcaContext.serialize_data_creation = False  # Default
OrcaContext.serialize_data_creation = True
```
Whether add a file lock to the data loading process for PyTorch Horovod training. 
This would be useful when you run multiple workers on a single node to download data to the same destination. 
Default to be False.
