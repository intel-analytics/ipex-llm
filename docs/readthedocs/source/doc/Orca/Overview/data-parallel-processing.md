# Distributed Data Processing 

---

**Orca provides efficient support of distributed data-parallel processing pipeline, a critical component for large-scale AI applications.**

### **1. TensorFlow Dataset and PyTorch DataLoader**

Orca will seamlessly parallelize the standard `tf.data.Dataset` or `torch.utils.data.DataLoader` pipelines across a large cluster in a data-parallel fashion, which can be directly used for distributed deep learning training, as shown below:

TensorFlow Dataset:
```python
import tensorflow as tf
import tensorflow_datasets as tfds
from zoo.orca.learn.tf.estimator import Estimator

def preprocess(data):
    data['image'] = tf.cast(data["image"], tf.float32) / 255.
    return data['image'], data['label']

dataset = tfds.load(name="mnist", split="train", data_dir=dataset_dir)
dataset = dataset.map(preprocess)
dataset = dataset.shuffle(1000)

est = Estimator.from_keras(keras_model=model)
est.fit(data=dataset)
```

Pytorch DataLoader:
```python
import torch
from torchvision import datasets, transforms
from zoo.orca.learn.pytorch import Estimator

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("/tmp/mnist", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

est = Estimator.from_torch(model=torch_model, optimizer=torch_optim, loss=torch_criterion)
zoo_estimator.fit(data=train_loader)
```

Under the hood, Orca will automatically replicate the _TensorFlow Dataset_ or _PyTorch DataLoader_ pipeline on each node in the cluster, shard the input data, and execute the data pipelines using Apache Spark and/or Ray distributedly. 

_**Note:** Known limitations include:_
1. _TensorFlow Dataset pipeline that contains transformations defined in native python function, such as `tf.py_func`, `tf.py_function`
and `tf.numpy_function` are currently not supported._
2. _TensorFlow Dataset pipeline created from generators, such as `Dataset.from_generators` are currently not supported._
3. _For TensorFlow Dataset and Pytorch DataLoader pipelines that read from files (including `tf.data.TFRecordDataset` and `tf.data.TextLineDataset`), one needs to ensure that the same file paths can be accessed on every node in the cluster._

#### **1.1. Data Creator Function**
Alternatively, the user may also pass a *Data Creator Function* as the input to the distributed training and inference. Inside the *Data Creator Function*, the user needs to create and return a `tf.data.Dataset` or `torch.utils.data.DataLoader` object, as shown below.

TensorFlow:
```python
import tensorflow as tf
import tensorflow_datasets as tfds
def preprocess(data):
    data['image'] = tf.cast(data["image"], tf.float32) / 255.
    return data['image'], data['label']

def train_data_creator(config, batch_size):
    dataset = tfds.load(name="mnist", split="train", data_dir=dataset_dir)
    dataset = dataset.map(preprocess)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

Pytorch:
```python
def train_data_creator(config, batch_size):
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(config["dir"], train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)
    return train_loader
```

### **2. Spark Dataframes**
Orca supports Spark Dataframes as the input to the distributed training, and as the input/output of the distributed inference. Consequently, the user can easily process large-scale dataset using Apache Spark, and directly apply AI models on the distributed (and possibly in-memory) Dataframes without data conversion or serialization. 

```python
df = spark.read.parquet("data.parquet")
est = Estimator.from_keras(keras_model=model) # the model accept two inputs and one label
est.fit(data=df,
        feature_cols=['user', 'item'], # specifies which column(s) to be used as inputs
        label_cols=['label']) # specifies which column(s) to be used as labels
```

### **3. XShards (Distributed Data-Parallel Python Processing)**

`XShards` in Orca allows the user to process large-scale dataset using *existing* Python codes in a distributed and data-parallel fashion, as shown below. 

```python
import numpy as np
from zoo.orca.data import XShards

train_images = np.random.random((20, 3, 224, 224))
train_label_images = np.zeros(20)
train_shards = XShards.partition([train_images, train_label_images])

def transform_to_dict(train_data):
    return {"x": train_data[0], "y": train_data[1]}
    
train_shards = train_shards.transform_shard(transform_to_dict)
```

In essence, an `XShards` contains an automatically sharded (or partitioned) Python object (e.g., Pandas Dataframe, Numpy NDArray,  Python Dictionary or List, etc.). Each partition of the `XShards` stores a subset of the Python object and is distributed across different nodes in the cluster; and the user may run arbitrary Python codes on each partition in a data-parallel fashion using `XShards.transform_shard`.

View the related [Python API doc](./data) for more details.
 
#### **3.1 Data-Parallel Pandas**
The user may use `XShards` to efficiently process large-size Pandas Dataframes in a distributed and data-parallel fashion.

First, the user can read CVS, JSON or Parquet files (stored on local disk, HDFS, AWS S3, etc.) to obtain an `XShards` of Pandas Dataframe, as shown below:
```python
from zoo.orca.data.pandas import read_csv
csv_path = "/path/to/csv_file_or_folder"
shard = read_csv(csv_path)
```

Each partition of the returned `XShards` stores a Pandas Dataframe object (containing a subset of the entire dataset), and then the user can apply Pandas operations as well as other (e.g., sklearn) operations on each partition, as shown below:   
```python
def negative(df, column_name):
    df[column_name] = df[column_name] * (-1)
    return df
    
train_shards = shard.transform_shard(negative, 'value')
```

In addition, some global operations  (such as `partition_by`, `unique`, etc.) are also supported on the `XShards` of Pandas Dataframe, as shown below:
```python
shard.partition_by(cols="location", num_partitions=4)
location_list = shard["location"].unique()
```
