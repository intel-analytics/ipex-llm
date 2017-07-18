# BigDL Configuration
BigDL uses Java properties to control its behavior. Here's the list of
these properties.

## Logging
* bigdl.utils.LoggerFilter.disable

Default value is **false**. Disable redirecting logs of Spark and BigDL to a file.

* bigdl.utils.LoggerFilter.logFile

Default value is **Current_Working_Directory/bigdl.log**. Where is the redirecting log.

* bigdl.utils.LoggerFilter.enableSparkLog

Default value is **true**. Enable redirecting Spark logs to logFile

## Mode
* bigdl.localMode

Default value is **false**. Whether BigDL is running on a local JVM.

* bigdl.engineType

Default value is **mklblas**. Specify which kind of native library to speed up the computing.

## Multi-threading

* bigdl.coreNumber

Default value is **half of the virtual core number**. How many core BigDL use on your machine. If
hyper thread is enabled on your machine, DO NOT set it larger than half of the virtual core number.

* bigdl.Parameter.syncPoolSize

Default value is **4**. Thread pool size for syncing parameter between executors.


## Distributed Training
* bigdl.network.nio

Default value is **true**. Whether use NIO as BlockManager backend in Spark 1.5. If it is set to
false, user can specify spark.shuffle.blockTransferService to change the BlockManager backend.

Note this only take affect on Spark 1.5.

* bigdl.failure.retryTimes

Default value is **5**. How many times to retry when there's failure in distributed Training.

* bigdl.failure.retryTimeInterval

Default value is **120**. Unit is second. How long recount the retry times.

* bigdl.check.singleton

Default value is **false**. Whether to check if multiple partition run on a same executor, which is
bad for performance.