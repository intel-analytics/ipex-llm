# BigDL Known Issues

## Spark Dynamic Allocation

By design, BigDL does not support Spark Dynamic Allocation mode, and needs to allocate fixed resources. But if your environment has already configured Spark Dynamic Allocation, or stipulated that Spark Dynamic Allocation must be used, you can set up the `SparkConf` as a workaround.

You can set the `spark.dynamicAllocation.enabled` as follows:

- Spark-submit
    
    ```bash
    spark-submit --conf spark.dynamicAllocation.enabled=false
    ```
    
- Yarn or K8s
    
    ```python
    init_orca_context(conf={"spark.dynamicAllocation.enabled": "false"})
    ```
    
Otherwise, if you can not set this configuration, you can set the `conf` as follows:

The value of `spark.dynamicAllocation.minExecutors` should equal to the value of `spark.dynamicAllocation.maxExecutors`, and the certain value can be set by users as demands. Here for example, we set them to 2.

- Spark-submit
    
    ```bash
    spark-submit --conf spark.dynamicAllocation.enabled=true \
                 --conf spark.dynamicAllocation.minExecutors 2 \
                 --conf spark.dynamicAllocation.maxExecutors 2
    ```
    
- Yarn or K8s
    
    ```python
    init_orca_context(conf={"spark.dynamicAllocation.enabled": "true", 
                            "spark.dynamicAllocation.minExecutors": "2", 
                            "spark.dynamicAllocation.maxExecutors": "2"})
    ```
    
After setting `conf`, you have to disable barrier execution mode.

```python
from bigdl.orca import OrcaContext
OrcaContext.barrier_mode = False
```

Besides, there may be the following errors and the corresponding solutions are as follows:

- **requirement failed: Engine.init: spark.dynamicAllocation.maxExecutors and spark.dynamicAllocation.minExecutors must be identical in dynamic allocation for BigDL**
    
    This error is because of the inequality of `spark.dynamicAllocation.maxExecutors` and `spark.dynamicAllocation.maxExecutors` . You can set them to the same value as follows:
    
    ```bash
    # Spark-submit
    spark-submit --conf spark.dynamicAllocation.minExecutors 2 \
    						 --conf spark.dynamicAllocation.maxExecutors 2
    # Yarn or K8s
    init_orca_context(conf={"spark.dynamicAllocation.minExecutors": "2", "spark.dynamicAllocation.maxExecutors": "2"})
    ```
    
- **Exception: spark.executor.cores not detected in the SparkContext, you need to manually specify num_ray_nodes and ray_node_cpu_cores for RayOnSparkContext to start ray services**
    
    This error is because that `num_ray_nodes` and `ray_node_cpu_cores` is not specified and also `spark.executor.cores` can not be detected when using dynamic allocation mode. You can specify `num_ray_nodes` and `ray_node_cpu_cores` as follows:
    
    ```bash
    init_orca_context(num_ray_nodes=2, ray_node_cpu_cores=4)
    ```
