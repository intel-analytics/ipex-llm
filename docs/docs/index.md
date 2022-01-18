
**This is the _OLD_ document website for BigDL; please visit the new [document website](https://bigdl.readthedocs.io/) instead.**

---

## AI for Big Data

The **AI for Big Data** community includes the following projects:

* [BigDL](https://github.com/intel-analytics/BigDL): distributed deep learning library for Apache Spark
* [Analytics Zoo](https://github.com/intel-analytics/analytics-zoo): distributed ***Tensorflow***, ***PyTorch*** and ***Ray*** on Apache Spark (*as well as Spark ML pipeline for BigDL*)

---
## **What is BigDL**

[BigDL](https://arxiv.org/abs/1804.05839) is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.

* **Rich deep learning support.** Modeled after [Torch](http://torch.ch/), BigDL provides comprehensive support for deep learning, including numeric computing (via [Tensor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor)) and high level [neural networks](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/nn); in addition, users can load pre-trained [Caffe](http://caffe.berkeleyvision.org/) or [Torch](http://torch.ch/) models into Spark programs using BigDL.

* **Extremely high performance.** To achieve high performance, BigDL uses Intel [oneMKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html), [oneDNN](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onednn.html) and multi-threaded programming in each Spark task. Consequently, it is orders of magnitude faster than out-of-box open source [Caffe](http://caffe.berkeleyvision.org/) or [Torch](http://torch.ch/) on a single-node Xeon (i.e., comparable with mainstream GPU).

* **Efficiently scale-out.** BigDL can efficiently scale out to perform data analytics at "Big Data scale", by leveraging [Apache Spark](http://spark.apache.org/) (a lightning fast distributed data processing framework), as well as efficient implementations of synchronous SGD and all-reduce communications on Spark. 

---
## **Why BigDL?**

You may want to write your deep learning programs using BigDL if:

* You want to analyze a large amount of data on the same Big Data (Hadoop/Spark) cluster where the data are stored (in, say, HDFS, HBase, Hive, Parquet, etc.).

* You want to add deep learning functionalities (either training or prediction) to your Big Data (Spark) programs and/or workflow.

* You want to leverage existing Hadoop/Spark clusters to run your deep learning applications, which can be then dynamically shared with other workloads (e.g., ETL, data warehouse, feature engineering, classical machine learning, graph analytics, etc.)

---
## **How to use BigDL?**

It is highly recommended to use the high-level APIs provided by [Analytics Zoo](https://github.com/intel-analytics/), including:

* [Spark ML pipeline support](https://analytics-zoo.readthedocs.io/en/latest/doc/UseCase/nnframes.html) for BigDL
* [Keras-like API](https://analytics-zoo.readthedocs.io/en/latest/doc/UseCase/keras-api.html) for BigDL

For additional information, you may refer to:

* [BigDL paper](https://arxiv.org/abs/1804.05839)
* [Getting Started page](gettingstarted)
* [User Group](https://groups.google.com/forum/#!forum/bigdl-user-group)
* [Mail List](mailto:bigdl-user-group+subscribe@googlegroups.com)

---
## **Citing BigDL**
If you've found BigDL useful for your project, you can cite the [paper](https://arxiv.org/abs/1804.05839) as follows:

```
@inproceedings{SOCC2019_BIGDL,
  title={BigDL: A Distributed Deep Learning Framework for Big Data},
  author={Dai, Jason (Jinquan) and Wang, Yiheng and Qiu, Xin and Ding, Ding and Zhang, Yao and Wang, Yanzhang and Jia, Xianyan and Zhang, Li (Cherry) and Wan, Yan and Li, Zhichao and Wang, Jiao and Huang, Shengsheng and Wu, Zhongyuan and Wang, Yang and Yang, Yuhao and She, Bowen and Shi, Dongjie and Lu, Qi and Huang, Kai and Song, Guoqiong},
  booktitle={Proceedings of the ACM Symposium on Cloud Computing},
  publisher={Association for Computing Machinery},
  pages={50--60},
  year={2019},
  series={SoCC'19},
  doi={10.1145/3357223.3362707},
  url={https://arxiv.org/pdf/1804.05839.pdf}
}
```


