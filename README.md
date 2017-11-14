# BigDL with Drizzle: Accelerating Large-Scale Distributed Deep Learning on Apache Spark

## What is BigDL?
BigDL is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.
* **Rich deep learning support.** Modeled after [Torch](http://torch.ch/), BigDL provides comprehensive support for deep learning, including numeric computing (via [Tensor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor)) and high level [neural networks](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/nn); in addition, users can load pre-trained [Caffe](http://caffe.berkeleyvision.org/) or [Torch](http://torch.ch/) models into Spark programs using BigDL.

* **Extremely high performance.** To achieve high performance, BigDL uses [Intel MKL](https://software.intel.com/en-us/intel-mkl) and multi-threaded programming in each Spark task. Consequently, it is orders of magnitude faster than out-of-box open source [Caffe](http://caffe.berkeleyvision.org/), [Torch](http://torch.ch/) or [TensorFlow](https://www.tensorflow.org/) on a single-node Xeon (i.e., comparable with mainstream GPU).

* **Efficiently scale-out.** BigDL can efficiently scale out to perform data analytics at "Big Data scale", by leveraging [Apache Spark](http://spark.apache.org/) (a lightning fast distributed data processing framework), as well as efficient implementations of synchronous SGD and all-reduce communications on Spark. 

## What is Drizzle?
Drizzle is a low latency execution engine for Apache Spark that is targeted at stream processing and iterative workloads. Currently, Spark uses a BSP computation model, and notifies the scheduler at the end of each task. Invoking the scheduler at the end of each task adds overheads and results in decreased throughput and increased latency.

In Drizzle, we introduce group scheduling, where multiple batches (or a group) of computation are scheduled at once. This helps decouple the granularity of task execution from scheduling and amortize the costs of task serialization and launch.

More information can be found at the Spark Drizzle project website:

https://github.com/amplab/drizzle-spark

## How to run BigDL with drizzle?
* Clone and build drizzle-spark. 
  1. git clone https://github.com/amplab/drizzle-spark 
  2. ./build/sbt -Phadoop-2.6 -Pyarn -Pmesos -Phive publish-local package 
  
   This will install the 2.1.1-drizzle Maven pom files.
* Clone the BigDL fork 
  git clone https://github.com/intel-analytics/BigDL -b new_parametermanager_drizzle

* Build BigDL using the spark_drizzle profile  
  bash make-dist.sh -P spark_drizzle

* Restart Spark using the Drizzle Spark jars and then run VGG/Inception benchmark. 