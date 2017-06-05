# ResNet
This example demonstrates how to use BigDL to train and evaluate the [ResNet](https://arxiv.org/abs/1512.03385) architecture on CIFAR-10 data

## DataSet
Support Cifar-10 dataset

Users can download the Cifar-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html)
The dataset contains two sub-directories, namely, train and val. Users need to set this dataset directory behind the "-f" flag in command line.


## Data Processing
We use pipeline to process the input data.
Input data are transformed by several pipeline classes, such as HFlip, BGRImgNormalizer, etc.

## Model
ShortcutType is a unique feature defined in ResNet. ShortcutType-A is used for Cifar-10, ShortcutType-B is used for ImageNet.
Model is implemented in <code>ResNet</code>

## Get the JAR
You can build one by refer to the
[Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code.

## Training
* bigdl.sh would setup the essential environment for you and it would accept a spark-submit command as an input parameter.

* Spark local, example command
```shell
dist/bin/bigdl.sh -- \
spark-submit --master local[physical_core_number] \
--class com.intel.analytics.bigdl.models.resnet.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-10/ \
--batchSize 448 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 \
--learningRate 0.1
```
* Spark standalone, example command
```shell
dist/bin/bigdl.sh -- \
spark-submit --master spark://xxx.xxx.xxx.xxx:xxxx \
--driver-memory 5g --executor-memory 5g \
--total-executor-cores 32 --executor-cores 8 \
--class com.intel.analytics.bigdl.models.resnet.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-10/ \
--batchSize 448 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 \
--learningRate 0.1
```
* Spark yarn client, example command
```shell
dist/bin/bigdl.sh -- \
spark-submit --master yarn \
--driver-memory 5g --executor-memory 5g \
--num-executors 4 --executor-cores 8 \
--class com.intel.analytics.bigdl.models.resnet.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f Cifar-10/ \
--batchSize 448 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 \
--learningRate 0.1
```

<code>Optimizer</code> class is used to train the model. Users can define validation method to evaluate the model. We use Top1Accuracy as the validation method.

We support Local and Spark versions of training. Users can define <code>env</code> as "Local" or "Spark" to set the training environment.

## Parameters
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
```
