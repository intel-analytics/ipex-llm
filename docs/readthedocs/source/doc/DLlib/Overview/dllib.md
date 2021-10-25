# DLlib User Guide

## 1. Overview

DLlib is a distributed deep learning library for Apache Spark; with DLlib, users can write their deep learning applications as standard Spark programs (using either Scala or Python APIs).

It includes the functionalities of the [original BigDL](https://github.com/intel-analytics/BigDL/tree/branch-0.14) project, and provides following high-level APIs for distributed deep learning on Spark:

* [Keras-like API](keras-api.md) 
* [Spark ML pipeline support](nnframes.md)

## 2. Scala user guide

### 2.1 Install

#### 2.1.1 **Download a pre-built library**
You can download the bigdl-dllib build from the [Release Page](../release.md).

#### 2.1.2 **Link with a release version**

Currently, dllib releases are hosted on maven central; here's an example to add the dllib dependency to your own project:
```xml
<dependency>
    <groupId>com.intel.analytics.bigdl</groupId>
    <artifactId>bigdl-dllib-[spark_2.4.6|spark_3.1.1]</artifactId>
    <version>${BIGD_DLLIB_VERSION}</version>
</dependency>
```
Please choose the suffix according to your Spark platform.

SBT developers can use
```sbt
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-[spark_2.4.6|spark_3.1.1]" % "${BIGDL_DLLIB_VERSION}"
```

### 2.2 Run
#### 2.2.1 **Set Environment Variables**
Set **BIGDL_HOME** and **SPARK_HOME**:

* If you download bigdl-dllib from the [Release Page](../release-download.md)
```bash
export SPARK_HOME=folder path where you extract the spark package
export BIGDL_HOME=folder path where you extract the bigdl package
```

---
#### 2.2.2 **Use Interactive Spark Shell**
You can try bigdl-dllib easily using the Spark interactive shell. Run below command to start spark shell with BigDL support:
```bash
${SPARK_HOME}/bin/spark-shell \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --jars ${BIGDL_HOME}/lib/bigdl-dllib-spark_2.4.6-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=${BIGDL_HOME}/lib/bigdl-dllib-spark_2.4.6-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${BIGDL_HOME}/lib/bigdl-dllib-spark_2.4.6-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  --master local[*]
```
You will see a welcome message looking like below:
```
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.3
      /_/

Using Scala version 2.11.12 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_181)
Spark context available as sc.
scala>
```

To use BigDL, you should first initialize the environment as below.
```scala
scala> import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.NNContext
scala> NNContext.initNNContext()
2021-10-25 10:12:36 WARN  SparkContext:66 - Using an existing SparkContext; some configuration may not take effect.
2021-10-25 10:12:36 WARN  SparkContext:66 - Using an existing SparkContext; some configuration may not take effect.
res0: org.apache.spark.SparkContext = org.apache.spark.SparkContext@525c0f74
```

Once the environment is successfully initiated, you'll be able to play with dllib API's.
For instance, to experiment with the ````dllib.nn```` APIs in dllib, you may try below code:
```scala
scala> import com.intel.analytics.bigdl.dllib.nn._
scala> import com.intel.analytics.bigdl.numeric.NumericFloat

scala> val model = Sequential()
       model.add(Reshape(Array(1, 28, 28)))
         .add(SpatialConvolution(1, 6, 5, 5))
         .add(Tanh())
         .add(SpatialMaxPooling(2, 2, 2, 2))
         .add(Tanh())
         .add(SpatialConvolution(6, 12, 5, 5))
         .add(SpatialMaxPooling(2, 2, 2, 2))
         .add(Reshape(Array(12 * 4 * 4)))
         .add(Linear(12 * 4 * 4, 100))
         .add(Tanh())
         .add(Linear(100, 10))
         .add(LogSoftMax())

res1: model.type =
Sequential[6bd4eba4]{
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
  (1): Reshape[33b96f86](1x28x28)
  (2): SpatialConvolution[8fa1895c](1 -> 6, 5 x 5, 1, 1, 0, 0)
  (3): Tanh[61c5d8aa]
  (4): SpatialMaxPooling[d1bdf524](2, 2, 2, 2, 0, 0)
  (5): Tanh[970561b3]
  (6): SpatialConvolution[53f5fbae](6 -> 12, 5 x 5, 1, 1, 0, 0)
  (7): SpatialMaxPooling[d562d2c5](2, 2, 2, 2, 0, 0)
  (8): Reshape[200b37de](192)
  (9): Linear[47c5b1f6](192 -> 100)
  (10): Tanh[fca82ebe]
  (11): Linear[96c684a2](100 -> 10)
  (12): LogSoftMax[7f1928fc]
}
```

---

#### 2.2.3 **Run as a Spark Program**
You can run a bigdl-dllib program, e.g., the [Image Inference](https://github.com/intel-analytics/BigDL/blob/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/imageInference), as a standard Spark program (running on either a local machine or a distributed cluster) as follows:

1. Download the pretrained caffe model and prepare the images

2. Run the following command:
```bash
# Spark local mode
${SPARK_HOME}/bin/spark-submit.sh \
  --master local[2] \
  --driver-class-path dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  --properties-file dist/conf/spark-bigdl.conf \
  --class com.intel.analytics.bigdl.dllib.examples.nnframes.imageInference.ImageInferenceExample \
  dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
  --caffeDefPath PROTOTXT_PATH \
  --caffeWeightsPath CAFFE_MODEL_PATH \
  --batchSize 32 \
  --imagePath IMAGE_PATH

# Spark standalone mode
## ${SPARK_HOME}/sbin/start-master.sh
## check master URL from http://localhost:8080
${SPARK_HOME}/bin/spark-submit.sh \
  --master spark://... \
  --driver-class-path dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  --properties-file dist/conf/spark-bigdl.conf \
  --conf spark.driver.extraClassPath=dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  --executor-cores cores_per_executor \
  --total-executor-cores total_cores_for_the_job \
  --class com.intel.analytics.bigdl.dllib.examples.nnframes.imageInference.ImageInferenceExample \
  dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
  --caffeDefPath PROTOTXT_PATH \
  --caffeWeightsPath CAFFE_MODEL_PATH \
  --batchSize 32 \
  --imagePath IMAGE_PATH

# Spark yarn client mode
${SPARK_HOME}/bin/spark-submit.sh \
 --master yarn \
 --deploy-mode client \
 --driver-class-path dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
 --properties-file dist/conf/spark-bigdl.conf \
 --conf spark.driver.extraClassPath=dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
 --conf spark.executor.extraClassPath=dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
 --executor-cores cores_per_executor \
 --num-executors executors_number \
 --class com.intel.analytics.bigdl.dllib.examples.nnframes.imageInference.ImageInferenceExample \
 dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
 --caffeDefPath PROTOTXT_PATH \
 --caffeWeightsPath CAFFE_MODEL_PATH \
 --batchSize 32 \
 --imagePath IMAGE_PATH

# Spark yarn cluster mode
${SPARK_HOME}/bin/spark-submit.sh \
 --master yarn \
 --deploy-mode cluster \
 --driver-class-path dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
 --properties-file dist/conf/spark-bigdl.conf \
 --conf spark.driver.extraClassPath=dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
 --conf spark.executor.extraClassPath=dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
 --executor-cores cores_per_executor \
 --num-executors executors_number \
 --class com.intel.analytics.bigdl.dllib.examples.nnframes.imageInference.ImageInferenceExample \
 dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
 --caffeDefPath PROTOTXT_PATH \
 --caffeWeightsPath CAFFE_MODEL_PATH \
 --batchSize 32 \
 --imagePath IMAGE_PATH
```

  The parameters used in the above command are:

  * --caffeDefPath: The path where your put the pretrained caffe prototxt.

  * --caffeWeightsPath: The path where your put the pretrained caffe model.

  * -b: The mini-batch size. The mini-batch size is expected to be a multiple of *total cores* used in the job. In this example, the mini-batch size is suggested to be set to *total cores * 4*

  * --imagePath: The folder where you put the image files.

If you are to run your own program, do remember to do the initialize before call other bigdl-dllib API's, as shown below.
```scala
 // Scala code example
 import com.intel.analytics.bigdl.dllib.NNContext
 NNContext.initNNContext()
```
---

### 2.3 Get started
---

This section show a single example of how to use dllib to build a deep learning application on Spark, using Keras and NNframe APIs

---
#### **Training build-in Inception**

A bigdl-dllib program starts with initialize `NNContext`;.
````scala
import com.intel.analytics.bigdl.dllib.NNContext
NNContext.initNNContext()
````

After the initialization, we need to:

1.load the image with nnframes api [```NNImageReader```](https://github.com/intel-analytics/BigDL/blob/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/nnframes/NNImageReader.scala)

````scala
val imageDF = NNImageReader.readImages(PASCAL_FILE_PATH, sc)
      .withColumn("label", lit(2.0f))
````
2.transform the image with [```imageProcessing```](https://github.com/intel-analytics/BigDL/tree/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/feature/image) (e.g., ````RowToImageFeature````, ````ImageResize```` and ````ImageCenterCrop````):

````scala
val transformer = RowToImageFeature() -> ImageResize(256, 256) -> ImageCenterCrop(224, 224) ->
      ImageChannelNormalize(123, 117, 104, 1, 1, 1) -> ImageMatToTensor() -> ImageFeatureToTensor()
````

3.After that, we _**create the [```NNClassifier```](https://github.com/intel-analytics/BigDL/blob/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/nnframes/NNClassifier.scala)**_ (either a distributed or local one depending on whether it runs on Spark or not) by specifying the ````DataSet````, the built-in model and the ````Criterion```` (which, given input and target, computes gradient per given loss function):
````scala
val estimator = NNClassifier(Inception_v1(1000), ZooClassNLLCriterion[Float](), transformer)
.setBatchSize(1)
.setEndWhen(Trigger.maxIteration(1))
.setFeaturesCol("image")
````

Finally, we _**train the model by calling ````estimator.fit````**_:
````scala
estimator.fit(imageDF)
````

---

## 3. Python user guide

### 3.1 Install

Run below command to install _bigdl-dllib_.

```bash
conda create -n my_env python=3.7
conda activate my_env
pip install bigdl-dllib
```

### 3.2 Run

#### **3.2.1 Interactive Shell**

You may test if the installation is successful using the interactive Python shell as follows:

* Type `python` in the command line to start a REPL.
* Try to run the example code below to verify the installation:

  ```python
  from bigdl.dllib.utils.nncontext import *

  sc = init_nncontext()  # Initiation of bigdl-dllib on the underlying cluster.
  ```

#### **3.2.2 Jupyter Notebook**

You can start the Jupyter notebook as you normally do using the following command and run Analytics Zoo programs directly in a Jupyter notebook:

```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

#### **3.2.3 Python Script**

You can directly write bigdl-dlllib programs in a Python file (e.g. script.py) and run in the command line as a normal Python program:

```bash
python script.py
```

---
### 3.3 Get started (example)
---

#### **Text Classification using BigDL Python API**

This tutorial describes the [textclassifier]( https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/models/textclassifier) example written using BigDL Python API, which builds a text classifier using a CNN (convolutional neural network) or LSTM or GRU model (as specified by the user). (It was first described by [this Keras tutorial](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html))

The example first creates the `SparkContext` using the SparkConf` return by the `create_spark_conf()` method, and then initialize the engine:
```python
  sc = SparkContext(appName="text_classifier",
                    conf=create_spark_conf())
  init_engine()
```

It then loads the [20 Newsgroup dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html) into RDD, and transforms the input data into an RDD of `Sample`. (Each `Sample` in essence contains a tuple of two NumPy ndarray representing the feature and label).

```python
  texts = news20.get_news20()
  data_rdd = sc.parallelize(texts, 2)
  ...
  sample_rdd = vector_rdd.map(
      lambda (vectors, label): to_sample(vectors, label, embedding_dim))
  train_rdd, val_rdd = sample_rdd.randomSplit(
      [training_split, 1-training_split])
```

After that, the example creates the neural network model as follows:
```python
def build_model(class_num):
    model = Sequential()

    if model_type.lower() == "cnn":
        model.add(Reshape([embedding_dim, 1, sequence_len]))
        model.add(SpatialConvolution(embedding_dim, 128, 5, 1))
        model.add(ReLU())
        model.add(SpatialMaxPooling(5, 1, 5, 1))
        model.add(SpatialConvolution(128, 128, 5, 1))
        model.add(ReLU())
        model.add(SpatialMaxPooling(5, 1, 5, 1))
        model.add(Reshape([128]))
    elif model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 128)))
        model.add(Select(2, -1))
    elif model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 128)))
        model.add(Select(2, -1))
    else:
        raise ValueError('model can only be cnn, lstm, or gru')

    model.add(Linear(128, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model
```
Finally the example creates the `Optimizer` (which accepts both the model and the training Sample RDD) and trains the model by calling `Optimizer.optimize()`:

```python
optimizer = Optimizer(
    model=build_model(news20.CLASS_NUM),
    training_rdd=train_rdd,
    criterion=ClassNLLCriterion(),
    end_trigger=MaxEpoch(max_epoch),
    batch_size=batch_size,
    optim_method=Adagrad())
...
train_model = optimizer.optimize()
```