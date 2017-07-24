# BigDL Configuration
BigDL uses Java properties to control its behavior. Here's the list of
these properties.

## How to set the properties

### Spark
If you run BigDL on Apache Spark, you can set the properties by passing
spark-submit options. Here's an example:
```bash
# Say you want to set property FOO to value BAR

spark-submit ...
    --conf 'spark.executor.extraJavaOptions=-DFOO=BAR' # Set that property for executor process
    --conf 'spark.driver.extraJavaOptions=-DFOO=BAR'   # Set that property for driver process
    ...
```

### Local Java/Scala program
If you run BigDL in a local Java/Scala program, you can set the properties
by passing JVM parameters. Here's an example:
```bash
# Say you want to set property FOO to value BAR
java -cp xxx.jar -DFOO=BAR your.main.class.name
```

## Available Properties

Category|Property|Default value|Description
-----|-----|------|-----
**Logging**|bigdl.utils.LoggerFilter.disable|*false*|Disable redirecting logs of Spark and BigDL to  a file.
|bigdl.utils.LoggerFilter.logFile|*Current_Working_Directory/bigdl.log*|Where is the redirecting log.
|bigdl.utils.LoggerFilter.enableSparkLog|*true*|Enable redirecting Spark logs to logFile. Set it to false when you don't want to see Spark logs in the redirecting log file.
**Mode**|bigdl.localMode|*false*|Whether BigDL is running as a local Java/Scala program.
**Multi-threading**|bigdl.coreNumber|*half of the virtual core number*|How many cores BigDL use on your machine. It is only used when bigdl.localMode is set to true. If hyper thread is enabled on your machine, DO NOT set it larger than half of the virtual core number.
|bigdl.Parameter.syncPoolSize|*4*|Thread pool size for syncing parameter between executors.
**Distributed Training**|bigdl.network.nio|*true*|Whether use NIO as BlockManager backend in Spark 1.5. If it is set to false, user can specify spark.shuffle.blockTransferService to change the BlockManager backend. **ONLY** used when running on Spark 1.5.
|bigdl.failure.retryTimes|*5*|How many times to retry when there's failure in distributed Training.
|bigdl.failure.retryTimeInterval|*120*|Unit is second. How long recount the retry times.
|bigdl.check.singleton|*false*|Whether to check if multiple partition run on a same executor, which is bad for performance.

