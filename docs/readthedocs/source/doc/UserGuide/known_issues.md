# BigDL Known Issues

## Spark Dynamic Allocation

By design, BigDL does not support Spark Dynamic Allocation mode, and needs to allocate fixed resources for deep learning model training. But if your environment has already configured Spark Dynamic Allocation, or stipulated that Spark Dynamic Allocation must be used, you may encounter the following error:

> **requirement failed: Engine.init: spark.dynamicAllocation.maxExecutors and spark.dynamicAllocation.minExecutors must be identical in dynamic allocation for BigDL**
> 

If so, you can set up the `SparkConf` as a workaround.

For `spark-submit` cluster mode,  the first solution is to disable the Spark Dynamic Allocation mode in `SparkConf`. You can set the `spark.dynamicAllocation.enabled` as follows:

```bash
spark-submit --conf spark.dynamicAllocation.enabled=false
```

Otherwise, if you can not set this configuration, you can set the `conf` as follows:

The value of `spark.dynamicAllocation.minExecutors` should equal to the value of `spark.dynamicAllocation.maxExecutors`, and the certain value can be set by users as demands. Here for example, we set them to 2.

```bash
spark-submit --conf spark.dynamicAllocation.enabled=true \
             --conf spark.dynamicAllocation.minExecutors 2 \
             --conf spark.dynamicAllocation.maxExecutors 2
```

For other cluster mode, such as `yarn` and `k8s`, our program will initiate `SparkContext`, and the Spark Dynamic Allocation mode is disabled by default. Thus, such problem will be avoid by design. 

After setting `SparkConf`, you have to disable barrier execution mode.

```python
from bigdl.orca import OrcaContext
OrcaContext.barrier_mode = False
```

You are also recommended to set `num_ray_nodes` and `ray_node_cpu_cores` equal to `spark.dynamicAllocation.minExecutors` and `spark.executor.cores` respectively. You can specify `num_ray_nodes` and `ray_node_cpu_cores` as follows:

```python
init_orca_context(num_ray_nodes=2, ray_node_cpu_cores=4)
```
