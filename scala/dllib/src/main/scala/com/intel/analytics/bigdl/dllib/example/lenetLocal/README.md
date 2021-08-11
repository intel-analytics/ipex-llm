# Running LeNet5 Model as a local Scala program

This example shows how to run training, prediction and testing with LeNet5  model on local JVM with BigDL. Lenet5 is a classical CNN model used in digital number classification. For detail information,
please refer to <http://yann.lecun.com/exdb/lenet/>.

To run the BigDL model as a local Scala program without Spark, user needs to set JVM property `bigdl.localMode` to `true`. If user wants to specify how many cores to be used for training/testing/prediction, he needs to set JVM property `bigdl.coreNumber` to the core number. User can either call `System.setProperty("bigdl.localMode", "true")` and `System.setProperty("bigdl.coreNumber", core_number)` in the Scala code, or pass -Dbigdl.localMode=true and -Dbigdl.coreNumber=core_number when runing the program. In this example, we use the former way to set these JVM properties. 

## Prepare MNIST Data
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the
files and put them in one folder(e.g. mnist).

There're four files. **train-images-idx3-ubyte** contains train images,
**train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images
 and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the
 download page.

## Get the JAR
You can build one by refer to the
[Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/) from the source code.

## Train the Model
Example command
```
scala -cp spark/dl/target/bigdl-VERSION-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.example.lenetLocal.Train \
-f path_to_mnist_folder \
-c core_number \
-b batch_size \
--checkpoint ./model
```

In the above commands
* -f: where you put your MNIST data
* -c: The core number on local machine used for this training. The default value is physical cores number. Get it through Runtime.getRuntime().availableProcessors() / 2
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of core_number
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as state.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.

## Test Model
The above commands will cache the model in specified path(--checkpoint). Run this command will
use the model to do a validation.

Example command
```
scala -cp spark/dl/target/bigdl-VERSION-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.example.lenetLocal.Test \
-f path_to_mnist_folder \
--model ./model/model.iteration \
-c core_number \
-b batch_size
```
In the above command
* -f: where you put your MNIST data
* --model: the model snapshot file
* -c: The core number on local machine used for this testing. The default value is physical cores number. Get it through Runtime.getRuntime().availableProcessors() / 2
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of core_number

## Predict with Model
The above commands will use the model in specified path(--checkpoint)to do a prediction with given data.

Example command
```
scala -cp spark/dl/target/bigdl-VERSION-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.example.lenetLocal.Predict \
-f path_to_mnist_folder \
-c core_number \
--model ./model/model.iteration
```
In the above command
* -f: where you put your MNIST data
* -c: The core number on local machine used for this prediction. The default value is physical cores number. Get it through Runtime.getRuntime().availableProcessors() / 2
* --model: the model snapshot file