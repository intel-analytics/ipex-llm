# RayOnSpark

---

[Ray](https://github.com/ray-project/ray) is an open source distributed framework for emerging AI applications.
With the _**RayOnSpark**_ support packaged in [BigDL Orca](../Overview/orca.md),
Users can seamlessly integrate Ray applications into the big data processing pipeline on the underlying Big Data cluster
(such as [Hadoop/YARN](../../UserGuide/hadoop.md) or [K8s](../../UserGuide/k8s.md)).

_**Note:** BigDL has been tested on Ray 1.9.2 and you are highly recommended to use this tested version._


### 1. Install

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment.
When installing bigdl-orca with pip, you can specify the extras key `[ray]` to install the additional dependencies
for running Ray (i.e. `ray[default]==1.9.2`, `aiohttp==3.9.2`, `async-timeout==4.0.1`, `aioredis==1.3.1`, `hiredis==2.0.0`, `prometheus-client==0.11.0`, `psutil`,  `setproctitle`):

```bash
conda create -n py37 python=3.7  # "py37" is conda environment name, you can use any name you like.
conda activate py37

pip install bigdl-orca[ray]
```

View [Python User Guide](../../UserGuide/python.html#install) and [Orca User Guide](../Overview/orca.md) for more installation instructions.

---
### 2. Initialize

We recommend using `init_orca_context` to initiate and run RayOnSpark on the underlying cluster. The Ray cluster would be launched by specifying `init_ray_on_spark=True`. For example, to launch Spark and Ray on standard Hadoop/YARN clusters in [YARN client mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn):

```python
from bigdl.orca import init_orca_context

sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, init_ray_on_spark=True)
```

You can input the following RayOnSpark related arguments when you `init_orca_context` for Ray configurations:
- `redis_port`: The redis port for the ray head node. The value would be randomly picked if not specified.
- `redis_password`: The password for redis. The value would be ray's default password if not specified.
- `object_store_memory`: The memory size for ray object_store in string. This can be specified in bytes(b), kilobytes(k), megabytes(m) or gigabytes(g). For example, "50b", "100k", "250m", "30g".
- `verbose`: True for more logs when starting ray. Default is False.
- `env`: The environment variable dict for running ray processes. Default is None.
- `extra_params`: The key value dict for extra options to launch ray. For example, `extra_params={"dashboard-port": "11281", "temp-dir": "/tmp/ray/"}`.
- `include_webui`: Default is True for including web ui when starting ray.
- `system_config`: The key value dict for overriding RayConfig defaults. Mainly for testing purposes. An example for system_config could be: `{"object_spilling_config":"{\"type\":\"filesystem\", \"params\":{\"directory_path\":\"/tmp/spill\"}}"}`.
- `num_ray_nodes`: The number of ray processes to start across the cluster. For Spark local mode, you don't need to specify this value.
For Spark cluster mode, it is default to be the number of Spark executors. If spark.executor.instances can't be detected in your SparkContext, you need to explicitly specify this. It is recommended that num_ray_nodes is not larger than the number of Spark executors to make sure there are enough resources in your cluster.
- `ray_node_cpu_cores`: The number of available cores for each ray process. For Spark local mode, it is default to be the number of Spark local cores.
For Spark cluster mode, it is default to be the number of cores for each Spark executor. If spark.executor.cores or spark.cores.max can't be detected in your SparkContext, you need to explicitly specify this. It is recommended that ray_node_cpu_cores is not larger than the number of cores for each Spark executor to make sure there are enough resources in your cluster.

By default, the Ray cluster would be launched using Spark barrier execution mode, you can turn it off via the configurations of `OrcaContext`:

```python
from bigdl.orca import OrcaContext

OrcaContext.barrier_mode = False
```

View [Orca Context](../Overview/orca-context.md) for more details.

---
### 3. Run

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

- You can retrieve the information of the Ray cluster via [`OrcaContext`](../Overview/orca-context.md):

  ```python
  from bigdl.orca import OrcaContext

  ray_ctx = OrcaContext.get_ray_context()
  address_info = ray_ctx.address_info  # The dictionary information of the ray cluster, including node_ip_address, object_store_address, webui_url, etc.
  redis_address = ray_ctx.redis_address  # The redis address of the ray cluster.
  ```

- You should call `stop_orca_context()` when your program finishes:

  ```python
  from bigdl.orca import stop_orca_context

  stop_orca_context()
  ```

---
### 4. Known Issue
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

---
### 5. FAQ
- **ValueError: Ray component worker_ports is trying to use a port number ... that is used by other components.**

  This error is because that some port in worker port list is occupied by other processes. To handle this issue, you can set range of the worker port list by using the parameters `min-worker-port` and `max-worker-port` in `init_orca_context` as follows:

  ```python
  init_orca_context(extra_params={"min-worker-port": "30000", "max-worker-port": "30033"})
  ```

- **ValueError: Failed to bind to 0.0.0.0:8265 because it's already occupied. You can use `ray start --dashboard-port ...` or `ray.init(dashboard_port=...)` to select a different port.**

  This error is because that ray dashboard port is occupied by other processes. To handle this issue, you can end the process that occupies the port or you can manually set the ray dashboard port by using the parameter `dashboard-port` in `init_orca_context` as follows:

  ```python
  init_orca_context(extra_params={"dashboard-port": "50005"})
  ```

  Note that, the similar error can happen to ray redis port as well, you can also set the ray redis port by using the parameter `redis_port` in `init_orca_context` as follows:

  ```python
  init_orca_context(redis_port=50006)
  ```
