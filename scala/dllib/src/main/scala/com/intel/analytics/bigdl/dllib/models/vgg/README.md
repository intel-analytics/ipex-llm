# VGG Model on CIFAR-10
This example demonstrates how to use BigDL to train and test a [VGG-like](http://torch.ch/blog/2015/07/30/cifar.html) network on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data.

## Prepare CIFAR-10 Dataset
You can download CIFAR-10 dataset from [this webpage](https://www.cs.toronto.edu/~kriz/cifar.html) (remember to choose the binary version).
.

## Get the JAR
You can build one by refer to the
[Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/) from the source code.

## Train Model on Spark
Example command for running in Spark cluster mode
```
spark-submit --master local[physical_core_number] \
--class com.intel.analytics.bigdl.models.vgg.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-folder \
-b batch_size \
--summary ./log \
--checkpoint ./model
```

Standalone cluster mode, example command
```
spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--class com.intel.analytics.bigdl.models.vgg.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-folder \
-b batchsize \
--summary ./log \
--checkpoint ./model
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
--summary ./log \
--checkpoint ./model
```
In the above commands
* -f: where you put your Cifar10 data
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as state.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number.
* --summary: Where you store the training metainfo, which can be visualized in tensorboard
* --optimizerVersion: option to set DistriOptimizer version, the value can be "optimizerV1" or "optimizerV2".
## Test Model
Example command for running in Spark local mode
```
spark-submit --master local[physical_core_number] \
--class com.intel.analytics.bigdl.models.vgg.Test \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f cifar \
--model model_file \
-b batch_size
```

Standalone cluster mode, example command
```
spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--class com.intel.analytics.bigdl.models.vgg.Test \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f cifar-folder \
--model model_file \
-b batch_size
```
Yarn cluster mode, example command
```
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores cores_per_executor \
--num-executors executors_number \
--class com.intel.analytics.bigdl.models.vgg.Test \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f cifar-folder \
--model ./model/model.iteration_number \
-b batch_size
```
In the above commands
* -f: where you put your MNIST data
* --model: the model snapshot file
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
