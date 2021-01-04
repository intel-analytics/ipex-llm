# Orca Context

---

**`OrcaContext` is the main entry for provisioning the Orca program on the underlying cluster (such as K8s or Hadoop cluster), or even on a single laptop.**

### **1. Initialization**
An Orca program usually starts with the initialization of `OrcaContext` as follows:

```
from zoo.orca import init_orca_context
init_orca_context(...)
```
In `init_orca_context`, the user may specify necessary runtime configurations for the program, including:
- *Cluster mode*: <TODO: explain computing environment supported>
- *Physical resource*: <TODO: explain common resources, e.g., node, core, memory, driver memory, etc.>

The Orca program simply runs `init_orca_context` on the local machine, which will automatically provision the runtime Python environment and distributed execution engine on the underlying computing environment (such as a single laptop, a large K8s or Hadoop cluster, etc.). <TODO: Add a architecture chart?>

View the related [Python API doc]() for more details.

### **2. Python Dependency**
A key challenge for scaling out Python program across a distributed cluster is how to properly install the required Python environment (libraries and dependencies) on each node in the cluster (preferably in an automatic and dynamic fashion). 

For K8s cluster, the user may install required Python packages in the container and specify the `container_image` argument when initializing `OrcaContext`. For Hadoop/YARN cluster, the user may use `conda` to create the Python virtual environment with required dependencies on the local machine, and `OrcaContext` will automatically detect the active `conda` environment and provision it on each node in the cluster.

View the user guide for [K8s]() and [Hadoop/YARN]() for more details.

### **3. Execution Engine**

Under the hood, `OrcaContext` will automatically provision Apache Spark and/or Ray as the underlying execution engine for the distributed data processing and model training/inference. <TODO: explain how the user can retrieve `SparkContext` and `RayContext`; we may also add these two as member variables of `OrcaContext`, and user can call `OrcaContext.get_spark_context()` and `OrcaContext.get_ray_context()`?>

### **4. Extra Configurations**
 <TODO: explain the following configurations>

 - log_output: shall we always set it to true?
 - pandas_read_backend: we may describe it in xShards?
 - serialize_data_creation: shall we always set it to true?

### **5. Stopping**
After the Orca program finishes, the user can call `stop_orca_context` to release resources and shut down the underlying Spark and/Ray execution engine.
