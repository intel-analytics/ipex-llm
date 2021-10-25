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
${BIGDL_HOME}/bin/spark-shell-with-zoo.sh --master local[2]
```

You will then see a welcome message like below:

```
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 2.4.3
      /_/
         
Using Scala version 2.11.12 (Java HotSpot(TM) 64-Bit Server VM, Java 1.8.0_112)
Type in expressions to have them evaluated.
Type :help for more information.
```

Before you try BigDL APIs, you should use `initNNcontext` to verify your environment:

```scala
scala> import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.common.NNContext

scala> val sc = NNContext.initNNContext("Run Example")
2021-01-26 10:19:52 WARN  SparkContext:66 - Using an existing SparkContext; some configuration may not take effect.
2021-01-26 10:19:53 WARN  SparkContext:66 - Using an existing SparkContext; some configuration may not take effect.
sc: org.apache.spark.SparkContext = org.apache.spark.SparkContext@487f025
```

#### **1.3 Run BigDL examples**

You can run an BigDL example, e.g., the [Wide & Deep Recommendation](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/recommendation), as a standard Spark program (running in either local mode or cluster mode) as follows:

1. Download Census Income Dataset to `./data/census` from [here](https://archive.ics.uci.edu/ml/datasets/Census+Income).

2. Run the following command:
```bash
# Spark local mode
${BIGDL_HOME}/bin/spark-submit-scala-with-zoo.sh \ 
  --master local[2] \
  --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample \
  dist/lib/analytics-zoo-bigdl_0.12.1-spark_2.4.3-0.9.0-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.9.0
  --inputDir ./data/census \
  --batchSize 320 \
  --maxEpoch 20 \
  --dataset census

# Spark standalone mode
${BIGDL_HOME}/bin/spark-submit-scala-with-zoo.sh \
  --master spark://... \         #add your spark master address
  --executor-cores cores_per_executor \
  --total-executor-cores total_cores_for_the_job \
  --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample \
  dist/lib/analytics-zoo-bigdl_0.12.1-spark_2.4.3-0.9.0-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.9.0
  --inputDir ./data/census \
  --batchSize 320 \
  --maxEpoch 20 \
  --dataset census

# Spark yarn client mode, please make sure the right HADOOP_CONF_DIR is set
${BIGDL_HOME}/bin/spark-submit-scala-with-zoo.sh \
  --master yarn \
  --deploy-mode client \
  --executor-cores cores_per_executor \
  --num-executors executors_number \
  --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample \
  dist/lib/analytics-zoo-bigdl_0.12.1-spark_2.4.3-0.9.0-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.9.0
  --inputDir ./data/census \
  --batchSize 320 \
  --maxEpoch 20 \
  --dataset census

# Spark yarn cluster mode, please make sure the right HADOOP_CONF_DIR is set
${BIGDL_HOME}/bin/spark-submit-scala-with-zoo.sh \
  --master yarn \
  --deploy-mode cluster \
  --executor-cores cores_per_executor \
  --num-executors executors_number \
  --class com.intel.analytics.zoo.examples.recommendation.WideAndDeepExample \
  dist/lib/analytics-zoo-bigdl_0.12.1-spark_2.4.3-0.9.0-jar-with-dependencies.jar \   #change to your jar file if your download is not spark_2.4.3-0.9.0
  --inputDir ./data/census \
  --batchSize 320 \
  --maxEpoch 20 \
  --dataset census
```

--- 

### **2. Build BigDL Applications**

This section will show you how to build your own deep learning project with BigDL. 

#### **2.1 Add BigDL dependency**
##### **2.1.1 official Release** 
Currently, BigDL releases are hosted on maven central; below is an example to add the BigDL dependency to your own project:


```xml
<dependency>
    <groupId>com.intel.analytics.zoo</groupId>
    <artifactId>analytics-zoo-bigdl_0.12.1-spark_2.4.3</artifactId>
    <version>0.9.0</version>
</dependency>
```

You can find the other SPARK version [here](https://search.maven.org/search?q=analytics-zoo-bigdl), such as `spark_2.1.1`, `spark_2.2.1`, `spark_2.3.1`, `spark_3.0.0`.   


SBT developers can use
```sbt
libraryDependencies += "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.12.1-spark_2.4.3" % "0.9.0"
```

##### **2.1.2 Nightly Build**

Currently, BigDL nightly build is hosted on [SonaType](https://oss.sonatype.org/content/groups/public/com/intel/analytics/zoo/).

To link your application with the latest BigDL nightly build, you should add some dependencies like [official releases](#11-official-release), but change `0.9.0` to the snapshot version (such as 0.10.0-snapshot), and add below repository to your pom.xml.


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

