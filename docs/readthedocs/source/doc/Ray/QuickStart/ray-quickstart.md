# RayOnSpark Quickstart

---

![](../../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/ray_parameter_server.ipynb) &nbsp;![](../../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/orca/colab-notebook/quickstart/ray_parameter_server.ipynb)

---

**In this guide, we will describe how to use RayOnSpark to directly run Ray programs on Big Data clusters in 2 simple steps.**

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../../UserGuide/python.md) for more details.

```bash
conda create -n bigdl python=3.7 # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl
pip install bigdl-orca[ray]
```

### **Step 1: Initialize**

We recommend using `init_orca_context` to initiate and run Analytics Zoo on the underlying cluster. The Ray cluster would be launched automatically by specifying `init_ray_on_spark=True`.

```python
from bigdl.orca import init_orca_context

if cluster_mode == "local":  # For local machine
    sc = init_orca_context(cluster_mode="local", cores=4, memory="10g", init_ray_on_spark=True)
elif cluster_mode == "k8s":  # For K8s cluster
    sc = init_orca_context(cluster_mode="k8s", num_nodes=2, cores=2, memory="10g", driver_memory="10g", driver_cores=1, init_ray_on_spark=True)
elif cluster_mode == "yarn":  # For Hadoop/YARN cluster
    sc = init_orca_context(cluster_mode="yarn", num_nodes=2, cores=2, memory="10g", driver_memory="10g", driver_cores=1, init_ray_on_spark=True)
```

This is the only place where you need to specify local or distributed mode.

By default, the Ray cluster would be launched using Spark barrier execution mode, you can turn it off via the configurations of `OrcaContext`:

```python
from bigdl.orca import OrcaContext

OrcaContext.barrier_mode = False
```

View [Orca Context](./../../Orca/Overview/orca-context.md) for more details.

**Note:** You should `export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir` when running on Hadoop YARN cluster. View [Hadoop User Guide](./../../UserGuide/hadoop.md) for more details.

You can retrieve the information of the Ray cluster via `OrcaContext`:

```python
from bigdl.orca import OrcaContext

ray_ctx = OrcaContext.get_ray_context()
address_info = ray_ctx.address_info  # The dictionary information of the ray cluster, including node_ip_address, object_store_address, webui_url, etc.
redis_address = ray_ctx.redis_address  # The redis address of the ray cluster.
```

### **Step 2: Run Ray Applications**

After the initialization, you can directly write Ray code inline with your Spark code, and run Ray programs on the underlying existing Big Data clusters. Ray [tasks](https://docs.ray.io/en/master/walkthrough.html#remote-functions-tasks) and [actors](https://docs.ray.io/en/master/actors.html) would be launched across the cluster.

The following example uses actor handles to implement a parameter server example for distributed asynchronous stochastic gradient descent. This is a simple Ray example for demonstration purpose. Similarly, you can write other Ray applications as you wish.

A parameter server is simply an object that stores the parameters (or "weights") of a machine learning model (this could be a neural network, a linear model, or something else). It exposes two methods: one for getting the parameters and one for updating the parameters.

By adding the `@ray.remote` decorator, the `ParameterServer` class becomes a Ray actor.

```python
import ray
import numpy as np

dim = 10
@ray.remote
class ParameterServer(object):
    def __init__(self, dim):
        self.parameters = np.zeros(dim)
    
    def get_parameters(self):
        return self.parameters
    
    def update_parameters(self, update):
        self.parameters += update

ps = ParameterServer.remote(dim)
```

In a typical machine learning training application, worker processes will run in an infinite loop that does the following:

1. Get the latest parameters from the parameter server.
2. Compute an update to the parameters (using the current parameters and some data).
3. Send the update to the parameter server.

By adding the `@ray.remote` decorator, the `worker` function becomes a Ray remote function.

```python
import time

@ray.remote
def worker(ps, dim, num_iters):
    for _ in range(num_iters):
        # Get the latest parameters.
        parameters = ray.get(ps.get_parameters.remote())
        # Compute an update.
        update = 1e-3 * parameters + np.ones(dim)
        # Update the parameters.
        ps.update_parameters.remote(update)
        # Sleep a little to simulate a real workload.
        time.sleep(0.5)

# Test that worker is implemented correctly. You do not need to change this line.
ray.get(worker.remote(ps, dim, 1))

# Start two workers.
worker_results = [worker.remote(ps, dim, 100) for _ in range(2)]
```

As the worker tasks are executing, you can query the parameter server from the driver and see the parameters changing in the background.

```
print(ray.get(ps.get_parameters.remote()))
```

**Note:** You should call `stop_orca_context()` when your program finishes:

```python
from bigdl.orca import stop_orca_context

stop_orca_context()
```
