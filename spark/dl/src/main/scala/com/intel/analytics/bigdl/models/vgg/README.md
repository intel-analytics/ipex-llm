# VGG Model on CIFAR-10
This example demonstrates how to use BigDL with Drizzle to train a [VGG-like](http://torch.ch/blog/2015/07/30/cifar.html) network on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data.

## Prepare CIFAR-10 Dataset
You can download CIFAR-10 dataset from [this webpage](https://www.cs.toronto.edu/~kriz/cifar.html) (remember to choose the binary version).
.

## Train Model on Spark
Example command for running in Spark cluster mode
```
spark-submit --master spark://xxx.xxx.xxx.xxx:xxxx \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--class com.intel.analytics.bigdl.models.vgg.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-folder \
-b batch_size \
--checkpoint ~/model \
--partitionNumber partition_number \
--nodeNumber node_number \
--corePerTask core_per_spark_task \
--drizzleGroupSize drizzle_groupSize \
--useDrizzle use_drizzle
```

Yarn cluster mode, example command
```
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores cores_per_executor \
--num-executors executors_number \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.vgg.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-folder \
-b batch_size \
--checkpoint ./model \
--partitionNumber partition_number \
--nodeNumber node_number \
--corePerTask core_per_spark_task \
--drizzleGroupSize drizzle_groupSize \
--useDrizzle use_drizzle
```
In the above commands
* -f: where you put your Cifar10 data
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as state.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number.
* --partitionNumber: partitions you want to have
* --nodeNumber: node numbers you want to use 
* --corePerTask: core numbers you want to use for each Spark task
* --drizzleGroupSize: drizzle group size
* --useDrizzle: whether use drizzle
