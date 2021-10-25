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
scala> import com.intel.analytics.bigdl.dllib.keras.layers._
scala> import com.intel.analytics.bigdl.numeric.NumericFloat
scala> import com.intel.analytics.bigdl.dllib.utils.Shape

scala> val seq = Sequential()
       val layer = ConvLSTM2D(32, 4, returnSequences = true, borderMode = "same",
            inputShape = Shape(8, 40, 40, 32))
       seq.add(layer)
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
  --class com.intel.analytics.bigdl.dllib.example.languagemodel.PTBWordLM \
  dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
  -f DATA_PATH \
  -b 4 \
  --numLayers 2 --vocab 100 --hidden 6 \
  --numSteps 3 --learningRate 0.005 -e 1 \
  --learningRateDecay 0.001 --keepProb 0.5

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
  --class com.intel.analytics.bigdl.dllib.example.languagemodel.PTBWordLM \
  dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
  -f DATA_PATH \
  -b 4 \
  --numLayers 2 --vocab 100 --hidden 6 \
  --numSteps 3 --learningRate 0.005 -e 1 \
  --learningRateDecay 0.001 --keepProb 0.5

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
 --class com.intel.analytics.bigdl.dllib.example.languagemodel.PTBWordLM \
 dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
 -f DATA_PATH \
 -b 4 \
 --numLayers 2 --vocab 100 --hidden 6 \
 --numSteps 3 --learningRate 0.005 -e 1 \
 --learningRateDecay 0.001 --keepProb 0.5

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
 --class com.intel.analytics.bigdl.dllib.example.languagemodel.PTBWordLM \
 dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
 -f DATA_PATH \
 -b 4 \
 --numLayers 2 --vocab 100 --hidden 6 \
 --numSteps 3 --learningRate 0.005 -e 1 \
 --learningRateDecay 0.001 --keepProb 0.5
```

  The parameters used in the above command are:

  * -f: The path where you put your PTB data.
  * -b: The mini-batch size. The mini-batch size is expected to be a multiple of *total cores* used in the job. In this example, the mini-batch size is suggested to be set to *total cores * 4*
  * --learningRate: learning rate for adagrad
  * --learningRateDecay: learning rate decay for adagrad
  * --hidden: hiddensize for lstm
  * --vocabSize: vocabulary size, default 10000
  * --numLayers: numbers of lstm cell, default 2 lstm cells
  * --numSteps: number of words per record in LM
  * --keepProb: the probability to do dropout

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

#### 3.1.1 Official Release

Run below command to install _bigdl-dllib_.

```bash
conda create -n my_env python=3.7
conda activate my_env
pip install bigdl-dllib
```

#### 3.1.2 Nightly build

You can install the latest nightly build of bigdl-dllib as follows:
```bash
pip install --pre --upgrade bigdl-dllib
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

You can start the Jupyter notebook as you normally do using the following command and run bigdl-dllib programs directly in a Jupyter notebook:

```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

#### **3.2.3 Python Script**

You can directly write bigdl-dlllib programs in a Python file (e.g. script.py) and run in the command line as a normal Python program:

```bash
python script.py
```

#### **3.2.4 Run in cluster**

call ```init_spark_on_yarn``` to do the initialization to run on yarn cluster.
  ```python
      sc = init_spark_on_yarn(
          hadoop_conf=hadoop_conf_dir,
          conda_name=detect_conda_env_name(),  # auto detect current conda env name
          num_executors=num_executors,
          executor_cores=num_cores_per_executor,
          executor_memory=executor_memory,
          driver_memory=driver_memory,
          driver_cores=driver_cores,
          conf={"spark.rpc.message.maxSize": "1024",
                "spark.task.maxFailures": "1",
                "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})
  ```

---
### 3.3 Get started
---

#### **Autograd Examples using bigdl-dllb keras Python API**

This tutorial describes the [Autograd](https://github.com/intel-analytics/BigDL/tree/branch-2.0/python/dllib/examples/autograd).

The example first do the initializton using `init_nncontext()`:
```python
  sc = init_nncontext()
```

It then generate the input data X_, Y_

```python
    data_len = 1000
    X_ = np.random.uniform(0, 1, (1000, 2))
    Y_ = ((2 * X_).sum(1) + 0.4).reshape([data_len, 1])
```

It then define the custom loss

```python
def mean_absolute_error(y_true, y_pred):
    result = mean(abs(y_true - y_pred), axis=1)
    return result
```

After that, the example creates the model as follows and set the criterion as the custom loss:
```python
    a = Input(shape=(2,))
    b = Dense(1)(a)
    c = Lambda(function=add_one_func)(b)
    model = Model(input=a, output=c)

    model.compile(optimizer=SGD(learningrate=1e-2),
                  loss=mean_absolute_error)
```
Finally the example trains the model by calling `model.fit`:

```python
    model.fit(x=X_,
              y=Y_,
              batch_size=32,
              nb_epoch=int(options.nb_epoch),
              distributed=False)
```