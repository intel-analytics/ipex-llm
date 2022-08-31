# Orca Known Issues

## **Estimator Issues**

### **UnkownError: Could not start gRPC server**

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

### **RuntimeError: Inter op parallelism cannot be modified after initialization**

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

## **OrcaContext Issues**

### **Exception: Failed to read dashbord log: [Errno 2] No such file or directory: '/tmp/ray/.../dashboard.log'**

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

## **Other Issues**

### **OSError: Unable to load libhdfs: ./libhdfs.so: cannot open shared object file: No such file or directory**

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
