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
[Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page) from the source code. We
will release a pre-build package soon.

## Train the Model
Example command
```
dist/bin/bigdl.sh -- \
java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.lenet.Train \
-f ~/mnist/ \
--core pyshical_core_number \
--node 1 \
--checkpoint ~/model \
-b batch_size
```
### Use Apache Spark
Local mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--master local[physical_core_number] \
--driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.lenet.Train \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-f ~/mnist/ \
--core physical_core_number \
--node 1 \
--checkpoint ~/model
```
Cluster mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--driver-class-path dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.models.lenet.Train \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-f ~/mnist/ \
--core physical_core \
--node node_number \
-b batch_size
```
In the above commands
* -f: where you put your MNIST data
* --core: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
* --node: Node number.
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as state.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
## Test Model
The above commands will cache the model in specified path(--checkpoint). Run this command will
use the model to do a validation.

Example command
```
dist/bin/bigdl.sh -- \
java -cp dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies-and-spark.jar \
com.intel.analytics.bigdl.models.lenet.Test \
-f ~/mnist/ \
--core physical_core_number \
-n 1 \
--model ~/model/model.iteration
```
Spark local mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--master local[physical_core_number] \
--class com.intel.analytics.bigdl.models.lenet.Test \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-f mnist/ \
--model model.iteration
--nodeNumber 1 \
--core physical_core_number \
-b batch_size
```
Spark cluster mode, example command
```
./dist/bin/bigdl.sh -- \
spark-submit \
--class com.intel.analytics.bigdl.models.lenet.Test \
dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-f mnist/ \
--model model.iteration_number
--nodeNumber node_number \
--core physical_number_per_node \
-b batch_size
```
In the above command
* -f: where you put your MNIST data
* --model: the model snapshot file
* --core: How many cores of your machine will be used in the training. Note that the core number should be physical core number. If your machine turn on hyper threading, one physical core will map to two OS core.
* --nodeNumber: Node number.
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
