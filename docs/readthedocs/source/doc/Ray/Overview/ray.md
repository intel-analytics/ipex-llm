# RayOnSpark

---

[Ray](https://github.com/ray-project/ray) is an open source distributed framework for emerging AI applications. With the _**RayOnSpark**_ support in Analytics Zoo, Users can seamlessly integrate Ray applications into the big data processing pipeline on the underlying Big Data cluster (such as [Hadoop/YARN](../../UserGuide/hadoop.md) or [K8s](../../UserGuide/k8s.md)).

_**Note:** Analytics Zoo has been tested on Ray 1.2.0 and you are highly recommended to use this tested version._


### **1. Install**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment. 
When installing analytics-zoo with pip, you can specify the extras key `[ray]` to additionally install the additional dependencies essential for running Ray (i.e. `ray==1.2.0`, `psutil`, `aiohttp`, `setproctitle`):

```bash
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo

pip install analytics-zoo[ray]
```

View [here](./python.html#install) for more installation instructions.

---
### **2. Initialize**

We recommend using `init_orca_context` to initiate and run Analytics Zoo on the underlying cluster. The Ray cluster would be launched as well by specifying `init_ray_on_spark=True`. For example, to launch Spark and Ray on standard Hadoop/YARN clusters in [YARN client mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn):

```python
from zoo.orca import init_orca_context

sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, init_ray_on_spark=True)
```

By default, the Ray cluster would be launched using Spark barrier execution mode, you can turn it off via the configurations of `OrcaContext`:

```python
from zoo.orca import OrcaContext

OrcaContext.barrier_mode = False
```

View [Orca Context](../../Orca/Overview/orca-context.md) for more details.

---
### **3. Run**

- After the initialization, you can directly run Ray applications on the underlying cluster. [Ray tasks](https://docs.ray.io/en/master/walkthrough.html#remote-functions-tasks) or [actors](https://docs.ray.io/en/master/actors.html) would be launched across the cluster. The following code shows a simple example:

  ```python
  import ray

  @ray.remote
  class Counter(object):
        def __init__(self):
            self.n = 0
  
        def increment(self):
            self.n += 1
            return self.n


  counters = [Counter.remote() for i in range(5)]
  print(ray.get([c.increment.remote() for c in counters]))
  ```

- You can retrieve the information of the Ray cluster via [`OrcaContext`](../Orca/Overview/orca-context.md):

  ```python
  from zoo.orca import OrcaContext
  
  ray_ctx = OrcaContext.get_ray_context()
  address_info = ray_ctx.address_info  # The dictionary information of the ray cluster, including node_ip_address, object_store_address, webui_url, etc.
  redis_address = ray_ctx.redis_address  # The redis address of the ray cluster.
  ```

- You should call `stop_orca_context()` when your program finishes:

  ```python
  from zoo.orca import stop_orca_context
  
  stop_orca_context()
  ```

---
### **4. Known Issue**
If you encounter the following error when launching Ray on the underlying cluster, especially when you are using a [Spark standalone](https://spark.apache.org/docs/latest/spark-standalone.html) cluster:

```
This system supports the C.UTF-8 locale which is recommended. You might be able to resolve your issue by exporting the following environment variables:

    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
```

Add the environment variables when calling `init_orca_context` would resolve the issue:

```python
sc = init_orca_context(cluster_mode, init_ray_on_spark=True, env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
```
