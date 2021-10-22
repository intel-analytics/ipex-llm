# DLlib User Guide

## 1. Overview

DLlib is a distributed deep learning library for Apache Spark; with DLlib, users can write their deep learning applications as standard Spark programs (using either Scala or Python APIs).

It includes the functionalities of the [original BigDL](https://github.com/intel-analytics/BigDL/tree/branch-0.14) project, and provides following high-level APIs for distributed deep learning on Spark:

* [Keras-like API](keras-api.md) 
* [Spark ML pipeline support](nnframes.md)

## 2. Scala user guide

### 2.1 Install

You can download the bigdl-dllib build from the [Release Page](../release.md). After extracting the prebuild package, you need to set environment variables **SPARK_HOME** as follows:

```bash
export SPARK_HOME=folder path where you extract the Spark package
```

### 2.2 Run
## **Link with a release version**

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

### 2.3 Get started (example)
You can run bigdl-dllib example, e.g., the [Image Inference](https://github.com/intel-analytics/BigDL/blob/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/imageInference/ImageInferenceExample.scala), as a standard Spark program (running in either local mode or cluster mode) as follows:

1. Download the pretrained caffe model

2. Run the following command:
```bash
# Spark local mode
${SPARK_HOME}/bin/spark-submit.sh \
  --master local[2] \
  --driver-class-path dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  --properties-file dist/conf/spark-bigdl.conf \
  --class com.intel.analytics.bigdl.dllib.examples.nnframes.imageInference.ImageInferenceExample \
  dist/lib/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
  --caffeDefPath ./nnframes/deploy.prototxt \
  --caffeWeightsPath ./nnframes/bvlc_googlenet.caffemodel \
  --batchSize 32 \
  --imagePath ./samples
```
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
You can run bigdl-dllib example, e.g., the [Image Inference](https://github.com/intel-analytics/BigDL/tree/branch-2.0/python/dllib/src/bigdl/dllib/examples/nnframes/imageInference):

1. Download the pretrained caffe model

2. Download the [Image Inference](https://github.com/intel-analytics/BigDL/blob/branch-2.0/python/dllib/src/bigdl/dllib/examples/nnframes/imageInference/ImageInferenceExample.py) script,

2. Run the following command:
```bash
python ImageInferenceExample.py -m bigdl_inception-v1_imagenet_0.4.0.model -f ${image_file} --b 16
```