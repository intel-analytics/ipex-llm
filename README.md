# BigDL: Distributed Deep Learning on Apache Spark

## What is BigDL?
BigDL is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.
* **Rich deep learning support.** Modeled after [Torch](http://torch.ch/), BigDL provides comprehensive support for deep learning, including numeric computing (via [Tensor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor)) and high level [neural networks](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/nn); in addition, users can load pre-trained [Caffe](http://caffe.berkeleyvision.org/) or [Torch](http://torch.ch/) models into Spark programs using BigDL.

* **Extremely high performance.** To achieve high performance, BigDL uses [Intel MKL](https://software.intel.com/en-us/intel-mkl) and multi-threaded programming in each Spark task. Consequently, it is orders of magnitude faster than out-of-box open source [Caffe](http://caffe.berkeleyvision.org/), [Torch](http://torch.ch/) or [TensorFlow](https://www.tensorflow.org/) on a single-node Xeon (i.e., comparable with mainstream GPU).

* **Efficiently scale-out.** BigDL can efficiently scale out to perform data analytics at "Big Data scale", by leveraging [Apache Spark](http://spark.apache.org/) (a lightning fast distributed data processing framework), as well as efficient implementations of synchronous SGD and all-reduce communications on Spark. 

## Why BigDL?
You may want to write your deep learning programs using BigDL if:
* You want to analyze a large amount of data on the same Big Data (Hadoop/Spark) cluster where the data are stored (in, say, HDFS, HBase, Hive, etc.).

* You want to add deep learning functionalities (either training or prediction) to your Big Data (Spark) programs and/or workflow.

* You want to leverage existing Hadoop/Spark clusters to run your deep learning applications, which can be then dynamically shared with other workloads (e.g., ETL, data warehouse, feature engineering, classical machine learning, graph analytics, etc.)

## How to use BigDL?
* More information can be found at the BigDL project website:
  
  https://bigdl-project.github.io/
  
  In particular, you can check out the [Getting Started page](https://bigdl-project.github.io/master/#getting-started/) for a quick overview of how to use BigDL
  
* For step-by-step deep leaning tutorials on BigDL (using Python), you can check out the [BigDL Tutorials project](https://github.com/intel-analytics/BigDL-tutorials)

* You can join the [BigDL Google Group](https://groups.google.com/forum/#!forum/bigdl-user-group) (or subscribe to the [Mail List](mailto:bigdl-user-group+subscribe@googlegroups.com)) for more questions and discussions on BigDL

* You can post bug reports and feature requests at the [Issue Page](https://github.com/intel-analytics/BigDL/issues)
