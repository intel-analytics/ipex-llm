#ResNet

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
### Command for Local Train
```bash
./dist/bin/bigdl.sh -- java -cp dl/target/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.resnet.Train --evn local -f /cifar-10 --batchSize 128 --core 4 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 --learningRate 0.1 -n 4
```

### Command for Spark Train
```bash
./dist/bin/bigdl.sh -- java -cp dl/target/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar com.intel.analytics.bigdl.models.resnet.Train --evn spark -f /cifar-10 --batchSize 512 --core 28 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 156 --learningRate 0.1 -n 4
```

<code>Optimizer</code> class is used to train the model. Users can define validation method to evaluate the model. We use Top1Accurary as the validation method.

We support Local and Spark versions of training. Users can define <code>env</code> as "Local" or "Spark" to set the training envrionment.

##Parameters
depth, shortcutType, number of epochs, learning rate, momentum, etc are user defined parameters.
