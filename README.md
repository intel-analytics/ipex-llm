# BigDL: Distributed Deep learning on Apache Spark

## What is BigDL?
BigDL is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.
* **Rich deep learning support.** Modeled after [Torch](http://torch.ch/), BigDL provides comprehensive support for deep learning, including numeric computing (via [Tensor](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/tensor)) and high level [neural networks] (https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/nn); in addition, users can load pre-trained [Caffe](http://caffe.berkeleyvision.org/) or [Torch](http://torch.ch/) models into Spark programs using BigDL.

* **Extremely high performance.** To achieve high performance, BigDL uses [Intel MKL](https://software.intel.com/en-us/intel-mkl) and multi-threaded programming in each Spark task. Consequently, it is orders of magnitude faster than out-of-box open source [Caffe](http://caffe.berkeleyvision.org/), [Torch](http://torch.ch/) or [TensorFlow](https://www.tensorflow.org/) on a single-node Xeon (i.e., comparable with mainstream GPU).

* **Efficiently scale-out.** BigDL can efficiently scale out to perform data analytics at "Big Data scale", by leveraging [Apache Spark](http://spark.apache.org/) (a lightning fast distributed data processing framework), as well as efficient implementations of synchronous SGD and all-reduce communications on Spark. 

## Why BigDL?
You may want to write your deep learning programs using BigDL if:
* You want to analyze a large amount of data on the same Big Data (Hadoop/Spark) cluster where the data are stored (in, say, HDFS, HBase, Hive, etc.).

* You want to add deep learning functionalities (either training or prediction) to your Big Data (Spark) programs and/or workflow.

* You want to leverage existing Hadoop/Spark clusters to run your deep learning applications, which can be then dynamically shared with other workloads (e.g., ETL, data warehouse, feature engineering, classical machine learning, graph analytics, etc.)

## How to use BigDL?
* To learn how to install and build BigDL (on both Linux and macOS), you can check out the [Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page)
* To learn how to run BigDL programs (as either a local Java program or a Spark program), you can check out the [Getting Started Page](https://github.com/intel-analytics/BigDL/wiki/Getting-Started)
* To try BigDL out on EC2, you can check out the [Running on EC2 Page](https://github.com/intel-analytics/BigDL/wiki/Running-on-EC2)
* To learn how to create practical neural networks using BigDL in a couple of minutes, you can check out the [Tutorials Page](https://github.com/intel-analytics/BigDL/wiki/Tutorials)
* For more details, you can check out the [Documents Page](https://github.com/intel-analytics/BigDL/wiki/Documents) (including Tutorials, Examples, Programming Guide, etc.)
* [Experimental Python Binding](https://github.com/intel-analytics/BigDL/blob/master/dl/src/main/python/README.md)

## Support
* You can join the [BigDL Google Group](https://groups.google.com/forum/#!forum/bigdl-user-group) (or subscribe to the [Mail List](mailto:bigdl-user-group+subscribe@googlegroups.com)) for more questions and discussions on BigDL
* You can post bug reports and feature requests at the [Issue Page](https://github.com/intel-analytics/BigDL/issues)
