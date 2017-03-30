# LeNet5 Model on MNIST

LeNet5 is a classical CNN model used in digital number classification. For detail information,
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
### Use Apache Spark
Local mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--master local[physical_core_number] \
--driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.lenet.Train \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-f path_to_mnist_folder \
-b batch_size \
--checkpoint ./model
```
Standalone cluster mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.lenet.Train \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-f path_to_mnist_folder \
-b batch_size \
--checkpoint ./model
```
Yarn cluster mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores cores_per_executor \
--num-executors executors_number \
--driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.lenet.Train \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
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
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number.

## Test Model
The above commands will cache the model in specified path(--checkpoint). Run this command will
use the model to do a validation.

Spark local mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--master local[physical_core_number] \
--class com.intel.analytics.bigdl.models.lenet.Test \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-f path_to_mnist_folder \
--model ./model/model.iteration \
-b batch_size
```
Standalone cluster mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--class com.intel.analytics.bigdl.models.lenet.Test \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-f path_to_mnist_folder \
--model ./model/model.iteration_number \
-b batch_size
```
Yarn cluster mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores cores_per_executor \
--num-executors executors_number \
--class com.intel.analytics.bigdl.models.lenet.Test \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-f path_to_mnist_folder \
--model ./model/model.iteration_number \
-b batch_size
```
In the above command
* -f: where you put your MNIST data
* --model: the model snapshot file
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number.
