# Running LeNet5 Model on local JVM

This example shows how to run training, prediction and testing with LeNet5  model on local JVM with BigDL. Lenet5 is a classical CNN model used in digital number classification. For detail information,
please refer to <http://yann.lecun.com/exdb/lenet/>.

## Prepare MNIST Data
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the
files and put them in one folder(e.g. mnist).

There're four files. **train-images-idx3-ubyte** contains train images,
**train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images
 and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the
 download page.

## Get the JAR
You can build one by refer to the
[Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code.

## Train the Model
Example command
```
java -cp dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
com.intel.analytics.bigdl.example.lenetLocal.Train \
-f path_to_mnist_folder \
-b batch_size \
--checkpoint ./model
```

In the above commands
* -f: where you put your MNIST data
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as state.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of core_number.

## Test Model
The above commands will cache the model in specified path(--checkpoint). Run this command will
use the model to do a validation.

Example command
```
java -cp dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
com.intel.analytics.bigdl.example.lenetLocal.Test \
-f path_to_mnist_folder \
--model ./model/model.iteration \
-b batch_size
```
In the above command
* -f: where you put your MNIST data
* --model: the model snapshot file
* -b: The mini-batch size.

## Predict with Model
The above commands will use the model in specified path(--checkpoint)to do a prediction with given data.

Example command
```
java -cp dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
com.intel.analytics.bigdl.example.lenetLocal.Predict \
-f path_to_mnist_folder \
--model ./model/model.iteration
```
In the above command
* -f: where you put your MNIST data
* --model: the model snapshot file