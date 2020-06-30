# ResNet
This example demonstrates how to use BigDL to train and evaluate the [ResNet](https://arxiv.org/abs/1512.03385) architecture on CIFAR-10 data and ImageNet data

## Data Processing
We use pipeline to process the input data.
Input data are transformed by several pipeline classes, such as HFlip, BGRImgNormalizer, RandomCropper, etc.

## Model
ShortcutType is a unique feature defined in ResNet. ShortcutType-A is used for Cifar-10, ShortcutType-B is used for ImageNet.
Model is implemented in <code>ResNet</code>

## Get the JAR
You can build one by refer to the
[Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/) from the source code.


## Train ResNet on Cifar-10

### Prepare Cifar-10 DataSet

Users can download the Cifar-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html)
The dataset contains two sub-directories, namely, train and val. Users need to set this dataset directory behind the "-f" flag in command line.

### Training
* Spark local example command
```shell
spark-submit --master local[physical_core_number] \
--driver-memory 3G \
--class com.intel.analytics.bigdl.models.resnet.TrainCIFAR10 \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-10/ \
--batchSize 448 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 \
--learningRate 0.1
```
* Spark standalone example command
```shell
spark-submit --master spark://xxx.xxx.xxx.xxx:xxxx \
--driver-memory 5g --executor-memory 5g \
--total-executor-cores 32 --executor-cores 8 \
--class com.intel.analytics.bigdl.models.resnet.TrainCIFAR10 \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-10/ \
--batchSize 448 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 \
--learningRate 0.1
```
* Spark yarn client example command
```shell
spark-submit --master yarn \
--driver-memory 5g --executor-memory 5g \
--num-executors 4 --executor-cores 8 \
--class com.intel.analytics.bigdl.models.resnet.TrainCIFAR10 \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-10/ \
--batchSize 448 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 \
--learningRate 0.1
```

<code>Optimizer</code> class is used to train the model. Users can define validation method to evaluate the model. We use Top1Accuracy as the validation method.

We support Local and Spark versions of training. Users can define <code>env</code> as "Local" or "Spark" to set the training environment.

### Parameters
```
    --folder | -f   [the directory to reach the data]
    --optnet        [share variables in convolutional layers to save the memory usage, default false]
    --depth         [number of layers for resnet]
    --classes       [number of classes]
    --shortcutType  [three shortcutTypes for resnet defined from the original paper, default "A"]
    --batchSize     [default 128, should be n*nodeNumber*coreNumber]
    --nEpochs       [number of epochs to train]
    --learningRate  [default 0.1]
    --momentum      [default 0.9]
    --weightDecay   [default 1e-4]
    --dampening     [default 0.0]
    --nesterov      [default true]
    --optimizerVersion  [distriOptimizer version, default "optimizerV1"]
```
## Train ResNet on ImageNet
This example shows the best practise we've experimented in multi-node training
### Prepare ImageNet DataSet
The imagenet dataset preparation can be found from
[BigDL inception Prepare the data](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/inception#prepare-the-data).
### Training
* Spark standalone example command
```shell
spark-submit \
--verbose \
--master spark://xxx.xxx.xxx.xxx:xxxx \
--driver-memory 200g \
--conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
--conf "spark.network.timeout=1000000" \
--executor-memory 200g \
--executor-cores 32 \
--total-executor-cores 2048 \
--class com.intel.analytics.bigdl.models.resnet.TrainImageNet \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f hdfs://xxx.xxx.xxx.xxx:xxxx/imagenet \
--batchSize 8192 --nEpochs 90 --learningRate 0.1 --warmupEpoch 5 \
 --maxLr 3.2 --cache /cache  --depth 50 --classes 1000
```

* Spark yarn client example command
```shell
spark-submit \
--verbose \
--master yarn \
--driver-memory 200g \
--conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
--conf "spark.network.timeout=1000000" \
--executor-memory 200g \
--executor-cores 32 \
--total-executor-cores 2048 \
--class com.intel.analytics.bigdl.models.resnet.TrainImageNet \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f hdfs://xxx.xxx.xxx.xxx:xxxx/imagenet \
--batchSize 8192 --nEpochs 90 --learningRate 0.1 --warmupEpoch 5 \
 --maxLr 3.2 --cache /cache --depth 50 --classes 1000
```
### Parameters
```
    --folder | -f   [the directory to reach the data]
    --batchSize     [default 8192, should be n*nodeNumber*coreNumber]
    --nEpochs       [number of epochs to train]
    --learningRate  [default 0.1]
    --warmupEpoch [warm up epochs]
    --maxLr [max learning rate, default to 3.2]
    --cache [directory to store snapshot]
    --depth         [number of layers for resnet, default to 50]
    --classes       [number of classes, default to 1000]
```
### Training reference
#### Hyper Parameters

**Global batch** : 8192

**Single batch per core** : 4

**Epochs** : 90

**Initial learning rate**: 0.1

**Warmup epochs**: 5

**Max learning rate**: 3.2

#### Training result (90 epochs)

**Top1 accuracy**: 0.76114

**Top5 accuracy**: 0.92724


