# Orca Context

---

`OrcaContext` is the main entry for provisioning the Orca program on the underlying cluster (such as K8s or Hadoop cluster), or just on a single laptop.

---
### **1. Initialization**

An Orca program usually starts with the initialization of `OrcaContext` as follows:

```python
from zoo.orca import init_orca_context

init_orca_context(...)
```

In `init_orca_context`, the user may specify necessary runtime configurations for the program, including:

- *Cluster mode*: Users can specify the computing environment for the program (a local machine, K8s cluster, Hadoop/YARN cluster, etc.).
- *Physical resources*: Users can specify the amount of physical resources to be allocated for the program on the underlying cluster, including the number of nodes in the cluster, the cores and memory allocated for each node, etc.

The Orca program simply runs `init_orca_context` on the local machine, which will automatically provision the runtime Python environment and distributed execution engine on the underlying computing environment (such as a single laptop, a large K8s or Hadoop cluster, etc.). <TODO: Add a architecture chart?>

View the related [Python API doc]() for more details.

---
### **2. Python Dependencies**

A key challenge for scaling out Python program across a distributed cluster is how to properly install the required Python environment (libraries and dependencies) on each node in the cluster (preferably in an automatic and dynamic fashion). 

For K8s cluster, the user may install required Python packages in the container and specify the `container_image` argument when `init_orca_context`. For Hadoop/YARN cluster, the user may use `conda` to create the Python virtual environment with required dependencies on the local machine, and `init_orca_context` will automatically detect the active `conda` environment and provision it on each node in the cluster.

View the user guide for [K8s]() and [Hadoop/YARN]() for more details.

---
### **3. Execution Engine**

Under the hood, `OrcaContext` will automatically provision Apache Spark and/or Ray as the underlying execution engine for the distributed data processing and model training/inference.

Users can easily retrieve `SparkContext` and `RayContext`, the main entry point for Spark and Ray respectively, via `OrcaContext`:

```python
from zoo.orca import OrcaContext

sc = OrcaContext.get_spark_context()
ray_ctx = OrcaContext.get_ray_context()
```

---
### **4. Extra Configurations**

Users can make extra configurations when using the functionalities of Project Orca via `OrcaContext`.

* `OrcaContext.log_output`: Default to be False. Setting it to True is recommended when running Jupyter notebook (this will display all the program output in the notebook). Make sure you set it before `init_orca_context`.
* `OrcaContext.serialize_data_creator`: Default to be False. Setting it to True would add a file lock when initializing data for distributed training (this may be useful if you run multiple workers on a single node and they download data to the same destination).
* `OrcaContext.pandas_read_backend`: Setting it to the backend to be used for reading data as Panda DataFrame. See [here](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/readthedocs/Orca/Overview/data-parallel-processing.md#31-data-parallel-pandas) for more details.

---
### **5. Termination**

After the Orca program finishes, the user can call `stop_orca_context` to release resources and shut down the underlying Spark and/or Ray execution engine.

```python
from zoo.orca import stop_orca_context

stop_orca_context()
```
