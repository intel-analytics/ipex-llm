#ResNet
This example demonstrates how to use BigDL to train and evaluate the [ResNet](https://arxiv.org/abs/1512.03385) architecture on CIFAR-10 data

##DataSet
Support Cifar-10 dataset

Users can download the Cifar-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html)
The dataset contains two sub-directories, namely, train and val. Users need to set this dataset directory behind the "-f" flag in command line.


##Data Processing
We use pipeline to process the input data.
Input data are transformed by several pipeline classes, such as HFlip, BGRImgNormalizer, etc.

##Model
ShortcutType is a unique feature defined in ResNet. ShortcutType-A is used for Cifar-10, ShortcutType-B is used for ImageNet.
Model is implemented in <code>ResNet</code>

##Training
* bigdl.sh would setup the essential environment for you and it would accept a spark-submit command as an input parameter.

* Local:
    * Execute:

        ```shell
        ./bigdl.sh -- java -cp dl/target/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
        com.intel.analytics.bigdl.models.resnet.Train --env local -f /cifar-10 --batchSize 128 --core 4 \
        --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 --learningRate 0.1 -n 1
        ```
* Spark cluster:
    * Execute:

        ```shell
        MASTER=spark://xxx.xxx.xxx.xxx:xxxx
        ./bigdl.sh -- spark-submit --master ${MASTER} --driver-memory 5g --executor-memory 5g \
        --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.bigdl.models.resnet.Train \
        bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar --env spark -f Cifar-10/ \
        --batchSize 448 --core 28 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 \
        --learningRate 0.1 -n 4
        ```

<code>Optimizer</code> class is used to train the model. Users can define validation method to evaluate the model. We use Top1Accuracy as the validation method.

We support Local and Spark versions of training. Users can define <code>env</code> as "Local" or "Spark" to set the training environment.

##Parameters
```
    --folder | -f   [the directory to reach the data]
    --coreNumber    [number of cores of the node, e.g 4 for a desktop]
    --nodeNumber    [number of nodes | servers to run on spark, 1 for local]
    --optnet        [share variables in convolutional layers to save the memory usage, default false]
    --depth         [number of layers for resnet]
    --classes       [number of classes]
    --shortcutType  [three shortcutTypes for resnet defined from the original paper, default "A"]
    --batchSize     [default 128, should be n*nodeNumber*coreNumber for spark]
    --nEpochs       [number of epochs to train]
    --learningRate  [default 0.1]
    --momentum      [default 0.9]
    --weightDecay   [default 1e-4]
    --dampening     [default 0.0]
    --nesterov      [default true]
    --env           [local | spark]
```
