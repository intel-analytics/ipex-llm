# Orca Known Issues

## Estimator Issues

### UnkownError: Could not start gRPC server

This error occurs while running Orca TF2 Estimator with spark backend, which may because the previous pyspark tensorflow job was not cleaned completely. You can retry later or you can set spark config `spark.python.worker.reuse=false` in your application.

If you are using `init_orca_context(cluster_mode="yarn-client")`:
   ```
   conf = {"spark.python.worker.reuse": "false"}
   init_orca_context(cluster_mode="yarn-client", conf=conf)
   ```
   If you are using `init_orca_context(cluster_mode="spark-submit")`:
   ```
   spark-submit --conf spark.python.worker.reuse=false
   ```

### RuntimeError: Inter op parallelism cannot be modified after initialization

This error occurs if you build your TensorFlow model on the driver rather than on workers. You should build the complete model in `model_creator` which runs on each worker node. You can refer to the following examples:

**Wrong Example**
   ```
   model = ...

   def model_creator(config):
       model.compile(...)
       return model

   estimator = Estimator.from_keras(model_creator=model_creator,...)
   ...
   ```

**Correct Example**
   ```
   def model_creator(config):
       model = ...
       model.compile(...)
       return model

   estimator = Estimator.from_keras(model_creator=model_creator,...)
   ...
   ```

## OrcaContext Issues

### Exception: Failed to read dashbord log: [Errno 2] No such file or directory: '/tmp/ray/.../dashboard.log'

This error occurs when initialize an orca context with `init_ray_on_spark=True`. We have not locate the root cause of this problem, but it might be caused by an atypical python environment.

You could follow below steps to workaround:

1. If you only need to use functions in ray (e.g. `bigdl.orca.learn` with `backend="ray"`, `bigdl.orca.automl` for pytorch/tensorflow model, `bigdl.chronos.autots` for time series model's auto-tunning), we may use ray as the first-class.

   1. Start a ray cluster by `ray start --head`. if you already have a ray cluster started, please direcetly jump to step 2.
   2. Initialize an orca context with `runtime="ray"` and `init_ray_on_spark=False`, please refer to detailed information [here](./orca-context.html).
   3. If you are using `bigdl.orca.automl` or `bigdl.chronos.autots` on a single node, please set:
      ```python
      ray_ctx = OrcaContext.get_ray_context()
      ray_ctx.is_local=True
      ```

2. If you really need to use ray on spark, please install bigdl-orca under a conda environment. Detailed information please refer to [here](./orca.html).

## Ray Issues

### ValueError: Ray component worker_ports is trying to use a port number ... that is used by other components.

This error is because that some port in worker port list is occupied by other processes. To handle this issue, you can set range of the worker port list by using the parameters `min-worker-port` and `max-worker-port` in `init_orca_context` as follows:

```python
init_orca_context(extra_params={"min-worker-port": "30000", "max-worker-port": "30033"})
```

### ValueError: Failed to bind to 0.0.0.0:8265 because it's already occupied. You can use `ray start --dashboard-port ...` or `ray.init(dashboard_port=...)` to select a different port.

This error is because that ray dashboard port is occupied by other processes. To handle this issue, you can end the process that occupies the port or you can manually set the ray dashboard port by using the parameter `dashboard-port` in `init_orca_context` as follows:

```python
init_orca_context(extra_params={"dashboard-port": "50005"})
```

Note that, the similar error can happen to ray redis port as well, you can also set the ray redis port by using the parameter `redis_port` in `init_orca_context` as follows:

```python
init_orca_context(redis_port=50006)
```

## Other Issues

### OSError: Unable to load libhdfs: ./libhdfs.so: cannot open shared object file: No such file or directory

This error is because PyArrow fails to locate `libhdfs.so` in default path of `$HADOOP_HOME/lib/native` when you run with YARN on Cloudera.
To solve this issue, you need to set the path of `libhdfs.so` in Cloudera to the environment variable of `ARROW_LIBHDFS_DIR` on Spark driver and executors with the following steps:

1. Run `locate libhdfs.so` on the client node to find `libhdfs.so`
2. `export ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64` (replace with the result of `locate libhdfs.so` in your environment)
3. If you are using `init_orca_context(cluster_mode="yarn-client")`:
   ```
   conf = {"spark.executorEnv.ARROW_LIBHDFS_DIR": "/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64"}
   init_orca_context(cluster_mode="yarn-client", conf=conf)
   ```
   If you are using `init_orca_context(cluster_mode="spark-submit")`:
   ```
   # For yarn-client mode
   spark-submit --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64

   # For yarn-cluster mode
   spark-submit --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64 \
                --conf spark.yarn.appMasterEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64

### Spark Dynamic Allocation

By design, BigDL does not support Spark Dynamic Allocation mode, and needs to allocate fixed resources for deep learning model training. Thus if your environment has already configured Spark Dynamic Allocation, or stipulated that Spark Dynamic Allocation must be used, you may encounter the following error:

> **requirement failed: Engine.init: spark.dynamicAllocation.maxExecutors and spark.dynamicAllocation.minExecutors must be identical in dynamic allocation for BigDL**
>

Here we provide a workaround for running BigDL under Spark Dynamic Allocation mode.

For `spark-submit` cluster mode, the first solution is to disable the Spark Dynamic Allocation mode in `SparkConf` when you submit your application as follows:

```bash
spark-submit --conf spark.dynamicAllocation.enabled=false
```

Otherwise, if you can not set this configuration due to your cluster settings, you can set `spark.dynamicAllocation.minExecutors` to be equal to `spark.dynamicAllocation.maxExecutors` as follows:

```bash
spark-submit --conf spark.dynamicAllocation.enabled=true \
             --conf spark.dynamicAllocation.minExecutors 2 \
             --conf spark.dynamicAllocation.maxExecutors 2
```

For other cluster modes, such as `yarn` and `k8s`, our program will initiate `SparkContext` for you, and the Spark Dynamic Allocation mode is disabled by default. Thus, generally you wouldn't encounter such problem.

If you are using Spark Dynamic Allocation, you have to disable barrier execution mode at the very beginning of your application as follows:

```python
from bigdl.orca import OrcaContext

OrcaContext.barrier_mode = False
```

For Spark Dynamic Allocation mode, you are also recommended to manually set `num_ray_nodes` and `ray_node_cpu_cores` equal to `spark.dynamicAllocation.minExecutors` and `spark.executor.cores` respectively. You can specify `num_ray_nodes` and `ray_node_cpu_cores` in `init_orca_context` as follows:

```python
init_orca_context(..., num_ray_nodes=2, ray_node_cpu_cores=4)
```
