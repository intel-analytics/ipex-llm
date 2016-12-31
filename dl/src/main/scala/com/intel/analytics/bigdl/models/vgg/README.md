# VGG Model on CIFAR-10
This example demonstrates how to use BigDL to train and test a [VGG-like](http://torch.ch/blog/2015/07/30/cifar.html) network on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data.

## Prepare CIFAR-10 Dataset
You can download CIFAR-10 dataset from [this webpage](https://www.cs.toronto.edu/~kriz/cifar.html) (remember to choose the binary version).
.

## Get the JAR
You can build one by refer to the
[Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code. We
will release a pre-build package soon.

## Train Model
Example command for running as a local Java program
```
./dist/bin/bigdl.sh -- \
java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.vgg.Train \
--core physical_core_number \
--node 1 \
--env local \
-f Cifar-folder \
-b batch_size \
--checkpoint ~/model
```

## Train Model on Spark
Example command for running in Spark cluster mode
```
./dist/bin/bigdl.sh -- \
spark-submit --master local[core_number] \
--class com.intel.analytics.bigdl.models.vgg.Train \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--core physical_core_number \
--node 1 \
--env spark \
-f Cifar-folder \
-b batch_size \
--checkpoint ~/model
```

Example command for running in Spark cluster mode
```
./dist/bin/bigdl.sh -- \
spark-submit --class com.intel.analytics.bigdl.models.vgg.Train \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--core physical_core_number \
--node node_number \
--env spark \
-f Cifar-folder \
-b batchsize \
--checkpoint ~/model
```
In the above commands
* -f: where you put your Cifar10 data
* --core: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
* --node: Node number.
* --env: It can be local/spark.
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as state.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
## Test Model
Example command for running as a local Java program
```
./dist/bin/bigdl.sh -- \
java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.vgg.Test \
-f cifar-folder \
--model model_file \
--nodeNumber 1
--core physical_core_number \
--env local
```

Example command for running in Spark local mode
```
./dist/bin/bigdl.sh -- \
spark-submit --master local[physical_core_number] \
--class com.intel.analytics.bigdl.models.vgg.Test \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-all-in-one.jar \
-f cifar \
--model model_file \
--nodeNumber 1 \
--core physical_core \
--env spark \
-b batch_size
```

Example command for running in Spark cluster mode
```
./dist/bin/bigdl.sh -- \
spark-submit --class com.intel.analytics.bigdl.models.vgg.Test \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-all-in-one.jar \
-f cifar-folder \
--model model_file \
--nodeNumber node_number \
--core physical_core_number \
--env spark \
-b batch_size
```
In the above commands
* -f: where you put your MNIST data
* --model: the model snapshot file
* --core: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
* --nodeNumber: Node number.
* --env: It can be local/spark.
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
