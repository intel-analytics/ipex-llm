## DLClassifierLogisticRegression

DLClassifierLogisticRegression example demonstrates how to use BigDL DLClassifier to train a
Logistic Regression Model. DLClassifier extends Spark Estimator and can act as a stage in a
ML Pipeline. The feature column can be Array or Spark Vectors, while the label column data should
be Double.

## DLEstimatorMultiLabelLR

DLEstimatorMultiLabelLR example demonstrates how to use BigDL DLEstimator to train a
multi-label Logistic Regression Model. DLEstimator extends Spark Estimator and can act as a
stage in a ML Pipeline. Both the feature and label column can be Array or Spark Vectors. The
feature column may also be Double.

## DLClassifierLeNet
DLClassifierLeNet example demonstrates how to use BigDL with Spark ML pipeline to train and predict LeNet5 model on MNIST dataset.

Learn more about Spark ML please refer to <http://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/Pipeline.html>
### Preparation

To start with this example, you need prepare your dataset.


1. Prepare  dataset

You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the
files and put them in one folder(e.g. mnist).

There're four files. 

**train-images-idx3-ubyte** contains train images.
**train-labels-idx1-ubyte** is train label file.
**t10k-images-idx3-ubyte** has validation images.
**t10k-labels-idx1-ubyte** contains validation labels.
 
For more detail, please refer to the download page.

### Run this example

Command to run the example in Spark local mode:
```
spark-submit \
--master local[physcial_core_number] \
--class com.intel.analytics.bigdl.example.MLPipeline.DLClassifierLeNet \
./dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f path_to_mnist_folder \
-b batch_size
```
Command to run the example in Spark standalone mode:
```
spark-submit \
--master spark://... \
--executor-cores cores_per_executor \
--total-executor-cores total_cores_for_the_job \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.example.MLPipeline.DLClassifierLeNet  \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f path_to_mnist_folder \
-b batch_size
```
Command to run the example in Spark yarn mode:
```
--master yarn \
--deploy-mode client \
--executor-cores cores_per_executor \
--num-executors executors_number \
--driver-class-path dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
--class com.intel.analytics.bigdl.example.MLPipeline.DLClassifierLeNet  \
dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
-f path_to_mnist_folder \
-b batch_size
```
where

* -f: where you put your MNIST data
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number
