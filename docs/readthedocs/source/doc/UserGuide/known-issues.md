# BigDL Known Issues

## Spark Dynamic Allocation

By design, BigDL does not support Spark Dynamic Allocation mode, and needs to allocate fixed resources. But if your environment has already configured Spark Dynamic Allocation, or stipulated that Spark Dynamic Allocation must be used, you can set up the `SparkConf` as a workaround.

### Spark-submit

If you use spark-submit to submit a task, you can set the `spark.dynamicAllocation.enabled` in command line parameters as follows:

```bash
spark-submit --conf spark.dynamicAllocation.enabled=false 
```

Otherwise, if you can not set this configuration, you can set the `conf` as follows:

```bash
# the number of minExecutors should equals to the number of maxExecutors, and the certain number can be set by users as demands. Here for example, we set them to 2.
spark-submit --conf spark.dynamicAllocation.minExecutors 2 \
             --conf spark.dynamicAllocation.maxExecutors 2
```

### Python API

If you use Python API to submit a task, the `SparkContext` will be initiated by your own program, so you can directly set the `spark.dynamicAllocation.enabled=false` in `conf` as follows:

```bash
init_orca_context(conf={"spark.dynamicAllocation.enabled": "false"})
```

Besides, there may be the following errors and the corresponding solutions are as follows:

### requirement failed: Engine.init: spark.dynamicAllocation.maxExecutors and spark.dynamicAllocation.minExecutors must be identical in dynamic allocation for BigDL

This error is because of the inequality of `spark.dynamicAllocation.maxExecutors` and `spark.dynamicAllocation.maxExecutors` . You can set them to the same value as follows:

```bash
# Spark-submit
spark-submit --conf spark.dynamicAllocation.minExecutors 2 \
             --conf spark.dynamicAllocation.maxExecutors 2
# Python API
init_orca_context(conf={"spark.dynamicAllocation.minExecutors": "2", "spark.dynamicAllocation.maxExecutors": "2"})
```

### Exception: spark.executor.cores not detected in the SparkContext, you need to manually specify num_ray_nodes and ray_node_cpu_cores for RayOnSparkContext to start ray services

This error is because that `num_ray_nodes` and `ray_node_cpu_cores` is not specified and also `spark.executor.cores` can not be detected when using dynamic allocation mode. You can specify `num_ray_nodes` and `ray_node_cpu_cores` as follows:

```bash
init_orca_context(num_ray_nodes=2, ray_node_cpu_cores=4)
```

### AttributeError: 'dict' object has no attribute 'getAll’

This error maybe because you specify `conf` in `init_orca_context` when using spark-submit command line to submit the task. `conf` will not be convert to `SparkConf` when using spark-submit, so there will not be `getAll` for `dict` object. You can set `SparkConf` in command line parameters as follows:

```bash
spark-submit --conf spark.dynamicAllocation.enabled=true \
             --conf spark.dynamicAllocation.minExecutors 2 \
             --conf spark.dynamicAllocation.maxExecutors 2
```
