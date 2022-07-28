# BigDL Known Issues

## Spark Dynamic Allocation

By design, BigDL does not support Spark Dynamic Allocation mode, and needs to allocate fixed resources for deep learning model training. Thus if your environment has already configured Spark Dynamic Allocation, or stipulated that Spark Dynamic Allocation must be used, you may encounter the following error:

> **requirement failed: Engine.init: spark.dynamicAllocation.maxExecutors and spark.dynamicAllocation.minExecutors must be identical in dynamic allocation for BigDL**
> 

Here we provide a workaround for running BigDL under Spark Dynamic Allocation mode.

For `spark-submit` cluster mode, the first solution is to disable the Spark Dynamic Allocation mode in `SparkConf` as you submit your application as follows:

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
