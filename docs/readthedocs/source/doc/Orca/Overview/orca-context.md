# Orca Context

---

`OrcaContext` is the main entry for provisioning the Orca program on the underlying cluster (such as K8s or Hadoop cluster), or just on a single laptop.

---
### **1. Initialization**

An Orca program usually starts with the initialization of `OrcaContext` as follows:

```python
from bigdl.orca import init_orca_context

init_orca_context(...)
```

In `init_orca_context`, the user may specify necessary runtime configurations for the Orca program, including:

- *Cluster mode*: Users can specify the computing environment for the program (a local machine, K8s cluster, Hadoop/YARN cluster, etc.).
- *Runtime*: Users can specify the backend for the program (spark and
ray, etc.) to create SparkContext and/or RayContext, the cluster mode
would work based on the specified runtime backend.
- *Physical resources*: Users can specify the amount of physical resources to be allocated for the program on the underlying cluster, including the number of nodes in the cluster, the cores and memory allocated for each node, etc.

The Orca program simply runs `init_orca_context` on the local machine, which will automatically provision the runtime Python environment and distributed execution engine on the underlying computing environment (such as a single laptop, a large K8s or Hadoop cluster, etc.).

View the related [Python API doc]() for more details.

---
### **2. Python Dependencies**

A key challenge for scaling out Python program across a distributed cluster is how to properly install the required Python environment (libraries and dependencies) on each node in the cluster (preferably in an automatic and dynamic fashion). 

For K8s cluster, the user may install required Python packages in the container and specify the `container_image` argument when `init_orca_context`. For Hadoop/YARN cluster, the user may use `conda` to create the Python virtual environment with required dependencies on the local machine, and `init_orca_context` will automatically detect the active `conda` environment and provision it on each node in the cluster.

You can also add .py, .zip or .egg files to distribute with your application by specifying `extra_python_lib` in `init_orca_context`. If you depend on multiple Python files we recommend packaging them into a .zip or .egg. Those files will be added to each node's python search path.

```python
init_orca_context(..., extra_python_lib="func1.py,func2.py,lib3.zip")
```

View the user guide for [K8s](../../UserGuide/k8s.md) and [Hadoop/YARN](../../UserGuide/hadoop.md) for more details.

---
### **3. Execution Engine**

Under the hood, `OrcaContext` will automatically provision Apache Spark and/or Ray as the underlying execution engine for the distributed data processing and model training/inference.

Users can easily retrieve `SparkContext` and `RayContext`, the main entry point for Spark and Ray respectively, via `OrcaContext`:

```python
from bigdl.orca import OrcaContext

sc = OrcaContext.get_spark_context()
ray_ctx = OrcaContext.get_ray_context()
```

---
### **4. Extra Configurations**

Users can make extra configurations when using the functionalities of Project Orca via `OrcaContext`.

* `OrcaContext.log_output`: Default to be False. `OrcaContext.log_output = True` is recommended when running Jupyter notebook (this will display all the program output in the notebook). Make sure you set it before `init_orca_context`.

* `OrcaContext.serialize_data_creator`: Default to be False. `OrcaContext.serialize_data_creator = True` would add a file lock when initializing data for distributed training (this may be useful if you run multiple workers on a single node and they download data to the same destination).

* `OrcaContext.pandas_read_backend`: The backend to be used for reading data as Panda DataFrame. Default to be "spark". See [here](./data-parallel-processing.html#data-parallel-pandas) for more details.

* `OrcaContext.train_data_store`: Default to be "DRAM". `OrcaContext.train_data_store = "DISK_n"` (e.g., "DISK_2") if the training data cannot fit in memory (this will store the data on disk, and cache only 1/n of the data in memory; after going through the 1/n, it will release the current cache, and load another 1/n into memory). Currently it works for TensorFlow and Keras Estimators only.

* `OrcaContext.barrier_mode`: Whether to use Spark barrier execution mode to launch Ray. Default to be True. You can set it to be False if you are using Spark below 2.4 or you need to have dynamic allocation enabled.

---

### **5. Termination**

After the Orca program finishes, the user can call `stop_orca_context` to release resources and shut down the underlying Spark and/or Ray execution engine.

```python
from bigdl.orca import stop_orca_context

stop_orca_context()
```
