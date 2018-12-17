# LeNet Model on MNIST using Keras-Style API

LeNet5 is a classical CNN model used in digital number classification. For detailed information with regard to LeNet, please refer to <http://yann.lecun.com/exdb/lenet/>.

This example is the same as [../../models/lenet](../../models/lenet), except that here we use the new set of Keras-Style API in BigDL for model definition and training, which is more user-friendly.


## Prepare MNIST Data
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the
files and put them in one folder(e.g. mnist).

There're four files. **train-images-idx3-ubyte** contains train images,
**train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images
 and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the
 download page.

After you uncompress the gzip files, these files may be renamed by some uncompress tools, e.g. **train-images-idx3-ubyte** is renamed
to **train-images.idx3-ubyte**. Please change the name back before you run the example.

## Get the JAR
You can build one by refer to the
[Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/) from the source code.

## Train the Model
Local mode, example command
```
spark-submit \
--master local[physical_core_number] \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.example.keras.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f path_to_mnist_folder \
-b batch_size \
```
Standalone cluster mode, example command
```
spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.example.keras.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f path_to_mnist_folder \
-b batch_size \
```
Yarn cluster mode, example command
```
spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores cores_per_executor \
--num-executors executors_number \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.example.keras.Train \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f path_to_mnist_folder \
-b batch_size \
```
In the above commands
* -f: an option to set the path where you put your MNIST data.
* -b: an option to set the mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number.
* -e: an option to set the number of epochs to train the model, the default value is 15.