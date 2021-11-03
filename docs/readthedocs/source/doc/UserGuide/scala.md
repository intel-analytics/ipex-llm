# Scala User Guide

---

### **1. Try BigDL Examples**
This section will show you how to download BigDL prebuild packages and run the build-in examples.

#### **1.1 Download and config** 
You can download the BigDL official releases and nightly build from the [Release Page](../release.md). After extracting the prebuild package, you need to set environment variables **BIGDL_HOME** and **SPARK_HOME** as follows:

```bash
export SPARK_HOME=folder path where you extract the Spark package
export BIGDL_HOME=folder path where you extract the BigDL package
```

#### **1.2 Use Spark interactive shell**
You can  try BigDL using the Spark interactive shell as follows:

```bash
${BIGDL_HOME}/bin/spark-shell-with-dllib.sh
```

You will then see a welcome message like below:

```
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.6
      /_/
         
Using Scala version 2.11.12 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_112)
Type in expressions to have them evaluated.
Type :help for more information.
```

Before you try BigDL APIs, you should use `initNNcontext` to verify your environment:

```scala
scala> import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.NNContext

scala> val sc = NNContext.initNNContext("Run Example")
2021-01-26 10:19:52 WARN  SparkContext:66 - Using an existing SparkContext; some configuration may not take effect.
2021-01-26 10:19:53 WARN  SparkContext:66 - Using an existing SparkContext; some configuration may not take effect.
sc: org.apache.spark.SparkContext = org.apache.spark.SparkContext@487f025
```
Once the environment is successfully initiated, you'll be able to play with dllib API's.
For instance, to experiment with the ````dllib.keras```` APIs in dllib, you may try below code:
```scala
scala> import com.intel.analytics.bigdl.dllib.keras.layers._
scala> import com.intel.analytics.bigdl.numeric.NumericFloat
scala> import com.intel.analytics.bigdl.dllib.utils.Shape

scala> val seq = Sequential()
       val layer = ConvLSTM2D(32, 4, returnSequences = true, borderMode = "same",
            inputShape = Shape(8, 40, 40, 32))
       seq.add(layer)
```

#### **1.3 Run BigDL examples**

You can run a bigdl-dllib program, e.g., the [Image Inference](https://github.com/intel-analytics/BigDL/blob/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/nnframes/imageInference), as a standard Spark program (running on either a local machine or a distributed cluster) as follows:

1. Download the pretrained caffe model and prepare the images

2. Run the following command:
```bash
# Spark local mode
${BIGDL_HOME}/bin/spark-submit-with-dllib.sh \
  --master local[2] \
  --class com.intel.analytics.bigdl.dllib.example.languagemodel.PTBWordLM \
  ${BIGDL_HOME}/jars/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
  -f DATA_PATH \
  -b 4 \
  --numLayers 2 --vocab 100 --hidden 6 \
  --numSteps 3 --learningRate 0.005 -e 1 \
  --learningRateDecay 0.001 --keepProb 0.5

# Spark standalone mode
## ${SPARK_HOME}/sbin/start-master.sh
## check master URL from http://localhost:8080
${BIGDL_HOME}/bin/spark-submit-with-dllib.sh \
  --master spark://... \
  --executor-cores cores_per_executor \
  --total-executor-cores total_cores_for_the_job \
  --class com.intel.analytics.bigdl.dllib.example.languagemodel.PTBWordLM \
  ${BIGDL_HOME}/jars/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
  -f DATA_PATH \
  -b 4 \
  --numLayers 2 --vocab 100 --hidden 6 \
  --numSteps 3 --learningRate 0.005 -e 1 \
  --learningRateDecay 0.001 --keepProb 0.5

# Spark yarn client mode
${BIGDL_HOME}/bin/spark-submit-with-dllib.sh \
 --master yarn \
 --deploy-mode client \
 --executor-cores cores_per_executor \
 --num-executors executors_number \
 --class com.intel.analytics.bigdl.dllib.example.languagemodel.PTBWordLM \
 ${BIGDL_HOME}/jars/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
 -f DATA_PATH \
 -b 4 \
 --numLayers 2 --vocab 100 --hidden 6 \
 --numSteps 3 --learningRate 0.005 -e 1 \
 --learningRateDecay 0.001 --keepProb 0.5

# Spark yarn cluster mode
${BIGDL_HOME}/bin/spark-submit-with-dllib.sh \
 --master yarn \
 --deploy-mode cluster \
 --executor-cores cores_per_executor \
 --num-executors executors_number \
 --class com.intel.analytics.bigdl.dllib.example.languagemodel.PTBWordLM \
 ${BIGDL_HOME}/jars/bigdl-dllib-0.14.0-SNAPSHOT-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.14.0
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

### **2. Build BigDL Applications**

This section will show you how to build your own deep learning project with BigDL. 

#### **2.1 Add BigDL dependency**
##### **2.1.1 official Release** 
Currently, BigDL releases are hosted on maven central; below is an example to add the BigDL dllib dependency to your own project:

```xml
<dependency>
    <groupId>com.intel.analytics.bigdl</groupId>
    <artifactId>bigdl-dllib-spark_2.4.6</artifactId>
    <version>0.14.0</version>
</dependency>
```

You can find the other SPARK version [here](https://search.maven.org/search?q=bigdl-dllib), such as `spark_3.1.2`.   


SBT developers can use
```sbt
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-dllib-spark_2.4.6" % "0.14.0"
```

##### **2.1.2 Nightly Build**

Currently, BigDL nightly build is hosted on [SonaType](https://oss.sonatype.org/content/groups/public/com/intel/analytics/bigdl/).

To link your application with the latest BigDL nightly build, you should add some dependencies like [official releases](#11-official-release), but change `2.0.0` to the snapshot version (such as 0.14.0-snapshot), and add below repository to your pom.xml.


```xml
<repository>
    <id>sonatype</id>
    <name>sonatype repository</name>
    <url>https://oss.sonatype.org/content/groups/public/</url>
    <releases>
        <enabled>true</enabled>
    </releases>
    <snapshots>
        <enabled>true</enabled>
    </snapshots>
</repository>
```

SBT developers can use
```sbt
resolvers += "ossrh repository" at "https://oss.sonatype.org/content/repositories/snapshots/"
```


#### **2.2 Build a Scala project**
To enable BigDL in project, you should add BigDL to your project's dependencies using maven or sbt. Here is a [simple MLP example](https://github.com/intel-analytics/zoo-tutorials/tree/master/scala/SimpleMlp) to show you how to use BigDL to build your own deep learning project using maven or sbt, and how to run the simple example in IDEA and spark-submit.

