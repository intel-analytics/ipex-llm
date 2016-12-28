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
<code>Optimizer</code> class is used to train the model. Users can define validation method to evaluate the model. We use Top1Accurary as the validation method.

We support Local and Spark versions of training. Users can define <code>env</code> as "Local" or "Spark" to set the training envrionment.

##Parameters
depth, shortcutType, number of epochs, learning rate, momentum, etc are user defined parameters.

##Sample Command Line
<code>java -cp bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar:spark-assembly-1.5.1-hadoop2.6.0.jar com.intel.analytics.bigdl.models.resnet.Train -f Cifar-10 --core 28 --optnet true --depth 20 --classes 10 --shortcutType A --batchSize 128 --nEpochs 50 --learningRate 0.1 --env local -n 4
</code>
