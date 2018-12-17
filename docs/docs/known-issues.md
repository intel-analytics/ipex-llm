* Currently, BigDL uses synchronous mini-batch SGD in model training. The mini-batch size is
expected to be a multiple of **total cores** used in the job.

* You may observe very poor performance when running BigDL for Spark 2.0 with Java 7; it is highly
recommended to use Java 8 when building and running BigDL for Spark 2.0.

* On Spark 2.0, please use default Java serializer instead of Kryo because of
[Kryo Issue 341](https://github.com/EsotericSoftware/kryo/issues/341). The issue has been fixed in
Kryo 4.0. However, Spark 2.0 uses Kryo 3.0.3. Spark 1.5 and 1.6 do not have this problem.

* On CentOS 6 and 7, please increase the max user processes to a larger value (e.g., 514585);
otherwise, you may see errors like "unable to create new native thread".

* Currently, BigDL will load all the training and validation data into memory during training.
You may encounter errors if it runs out of memory.

* If you meet the program stuck after **Save model...** on Mesos, check the `spark.driver.memory`
and increase the value. Eg, VGG on Cifar10 may need 20G+.

* If you meet `can't find executor core number` on Mesos, you should pass the executor cores
through `--conf spark.executor.cores=xxx`

* On Windows, if you meet "Could not locate executable null\bin\winutils.exe" error, you need to
install winutils.exe. Please refer this
[post](https://stackoverflow.com/questions/35652665/java-io-ioexception-could-not-locate-executable-null-bin-winutils-exe-in-the-ha).

* If the training data are cached before all the executor resources are allocated(this sometimes
happens when the data set is too small), we find Spark may not distribute the training data
partitions evenly on each executor. So the training tasks will be unbalanced allocated among
executors. To solve this problem, you can increase the
`spark.scheduler.maxRegisteredResourcesWaitingTime` property(default is 30s).