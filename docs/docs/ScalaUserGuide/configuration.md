BigDL uses Java properties to control its behavior. Here's the list of
these properties.

## **How to set the properties**

### **Spark**
If you run BigDL on Apache Spark, you can set the properties by passing
spark-submit options. Here's an example:
```bash
# Say you want to set property FOO to value BAR
spark-submit ...
    --conf 'spark.executor.extraJavaOptions=-DFOO=BAR' # Set that property for executor process
    --conf 'spark.driver.extraJavaOptions=-DFOO=BAR'   # Set that property for driver process
    ...
```

### **Local Java/Scala program**
If you run BigDL as a local Java/Scala program, you can set the properties
by passing JVM parameters. Here's an example:
```bash
# Say you want to set property FOO to value BAR
java -cp xxx.jar -DFOO=BAR your.main.class.name
```

---
## **Available Properties**

**Logging**

- `bigdl.utils.LoggerFilter.disable`: To disable redirecting logs of Spark and BigDL to a file. Default is false.
- `bigdl.utils.LoggerFilter.logFile`: To set the path to redirect log. By default, it will be directed to `bigdl.log` in the current working directory.
- `bigdl.utils.LoggerFilter.enableSparkLog`: To enable redirecting Spark logs to logFile. Set it to false when you don't want to see Spark logs in the redirected log file. Default is true.

**Mode**

- `bigdl.localMode`: Whether BigDL is running as a local Java/Scala program. Default is false.
- `bigdl.engineType`: Default is **mklblas**. When you run model contains mkl dnn layers, you should set it to **mkldnn** to get better performance.

**Multi-threading**

- `bigdl.coreNumber`: To set how many cores BigDL will use on your machine. It will only be used when bigdl.localMode is set to true. If hyper thread is enabled on your machine, __DO NOT__ set it larger than half of the virtual core number. Default is half of the virtual core number.
- `bigdl.Parameter.syncPoolSize`: To set the thread pool size for syncing parameter between executors. Default is 4.

**Distributed Training**

- `bigdl.network.nio`: Whether use NIO as BlockManager backend in Spark 1.5. Default is true. If it is set to be false, user can specify spark.shuffle.blockTransferService to change the BlockManager backend. __ONLY__ use this when running on Spark 1.5.
- `bigdl.failure.retryTimes`: To set how many times to retry when there's failure in distributed training. Default is 5.
- `bigdl.failure.retryTimeInterval`: To set how long to recount the retry times. Time unit here is second. Default is 120.
- `bigdl.check.singleton`: To check whether multiple partitions run on the same executor, which is bad for performance. Default is false.
- `bigdl.ModelBroadcastFactory`: Specify a ModelBroadcastFactory which creates a ModelBroadcast to control how to broadcast the model in the distributed training.

**Tensor**

- `bigdl.tensor.fold`: To set how many elements in a tensor to determine it is a large tensor, and thus print only part of it. Default is 1000.