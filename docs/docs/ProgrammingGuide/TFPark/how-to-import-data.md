## TFDataset


### Introduction

**TFDatasets** is the main entrance point in TFPark for importing and manipulating data.
It represents a distributed collection of elements (backed by a RDD) to be fed into a
TensorFlow graph for training, evaluation or inference. It provides a rich set of tools
to import data from various data sources and work as a unified interface to interact with
other components of TFPark.

This guide will walk you through some common cases of importing data and you can find detailed description
of TFDataset's API in [Analytics-Zoo API Guide](../../APIGuide/TFPark/tf-dataset.md).


### Basics

`TFDataset`'s job is to take in dataset, distribute the data across the Spark cluster and transform each data
record into the format that is compatible with TFPark.

Here are a few common features that every TFDataset share:

1. `TFDataset` will automatically stack consecutive records into batches. The `batch_size` argument (for training)
or `batch_per_thread` argument (for inference or evaluation) should be set when creating TFDataset.
The `batch_size` here is used for training and it means the total batch size in distributed training.
In other words, it equals to the total number of records processed in one iteration in the
whole cluster. `batch_size` should be a multiple of the total number of cores that is allocated for this Spark application
so that we can distributed the workload evenly across the cluster. You may need to adjust your other training
hyper-parameters when `batch_size` is changed. `batch_per_thread` is used for inference or evaluation
and it means the number of records process in one iteration in one partition. `batch_per_thread` is argument for tuning
performance and it does not affect the correctness or accuracy of the prediction or evaluation. Too small `batch_per_thread`
might slow down the prediction/evaluation.

2. For training, `TFDataset` can optionally takes a validation data source for validation at the the end of each epoch.
The validation data source should has the same structure of the main data source used for training.

```python
import numpy as np
from zoo.tfpark import TFDataset
feature_data = np.random.randn(100, 28, 28, 1)
label_data = np.random.randint(0, 10, size=(100,))
val_feature_data = np.random.randn(100, 28, 28, 1)
val_label_data = np.random.randint(0, 10, size=(100,))
dataset = TFDataset.from_ndarrays((feature_data, label_data), batch_size=32, val_tensors=(val_feature_data, val_label_data))
```
 

### Working with in-memory ndarray

If your input data is quite small, the simplest way to create `TFDataset` to convert them to ndarrays and use
`TFDataset.from_ndarrays()`

E.g.

```python
import numpy as np
from zoo.tfpark import TFDataset
feature_data = np.random.randn(100, 28, 28, 1)
label_data = np.random.randint(0, 10, size=(100,))
dataset = TFDataset.from_ndarrays((feature_data, label_data), batch_size=32)
```

### Working with data files including (csv files, text files and TFRecord files)

TFDataset support reading the records in tf.data.Dataset, so you can use tf.data.Dataset to read and process your data
files and pass it to TFDataset. TFDataset will automatically ship the dataset to different Spark executors, shard the
data and batch the records for further consumption.

If you data files is already in HDFS, you should configure you dataset with the path with the following pattern
`"hdfs://namenode:port/path/to/file.txt"` and TFDataset will directly access that file in HDFS in each executor.
`HDFS_HDFS_HOME` environment may needs to be set to the location where hadoop is installed for both Spark driver
and Spark executor. More information on the environment variable can be found [here](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/hadoop.md).

If you data files are in local file system, you can either upload it to a HDFS cluster and use the above approach or
copy all the data files on each executor in exact the same location.

More information on `tf.data.Dataset` can be found [here](https://www.tensorflow.org/guide/data).

E.g.

```python
ds = tf.data.TextLineDataset("hdfs://path/to/data.csv")
ds = ds.map(lambda line: tf.parse_csv(line, COLUMNS))
ds = ds.map(lamdda data: extract_features_labels(data))
dataset = TFDataset.from_tf_data_dataset(dataset, batch_size=32)
```


### Working with Analytics Zoo Feature Engineering tools

Analytics Zoo provides a rich set of tools to build complex data engineering pipelines on top Spark, including
`ImageSet`, `TextSet` and `FeatureSet`. TFPark also support using those tools for manipulating data. Specifically,
you can use `TFDataset.from_image_set`, `TFDataset.from_text_set` and `TFDataset.from_feature_set` for importing
data pipeline written in those tools. Details for these api can be found in [Analytics-Zoo API Guide](../../APIGuide/TFPark/tf-dataset.md).
More information on Analytics Zoo's Feature Engineering tools can be found [here](../../APIGuide/FeatureEngineering/featureset.md).


### Working with RDD or Spark DataFrame data

If the about approach does not match your use cases, you can always transform your data into RDD or DataFrame using
Spark's data processing capability.

For rdd, we assume each record contains a tuple of numpy.ndarrays or a tuple of list/dict of numpy.ndarrays. The first
element of the tuple, will be interpreted as feature and the second (optional) will be interpreted as label. Each record
should has the same structure. Details for these api can be found in [Analytics-Zoo API Guide](../../APIGuide/TFPark/tf-dataset.md).

e.g.
```python
image_rdd = sc.parallelize(np.random.randn(100, 28, 28, 1))
labels_rdd = sc.parallelize(np.random.randint(0, 10, size=(100,)))
rdd = image_rdd.zip(labels_rdd)
dataset = TFDataset.from_rdd(rdd,
                             features=(tf.float32, [28, 28, 1]),
                             labels=(tf.int32, []),
                             batch_size=32)
```

For dataframe, you should which columns are features and which columns are labels (optional). And currently only numerical
types and vectors are supported. Details for these api can be found in [Analytics-Zoo API Guide](../../APIGuide/TFPark/tf-dataset.md).

e.g.
```python
rdd = self.sc.range(0, 1000)
df = rdd.map(lambda x: (DenseVector(np.random.rand(20).astype(np.float)),
                                x % 10)).toDF(["feature", "label"])
dataset = TFDataset.from_dataframe(train_df,
                                   feature_cols=["feature"],
                                   labels_cols=["label"],
                                   batch_size=32)
```
