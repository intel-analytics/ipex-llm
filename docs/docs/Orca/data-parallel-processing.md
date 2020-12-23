**Orca provides efficient support of distributed data-parallel processing pipeline, a critical component for large-scale AI applications.**

### **1. TensorFlow Dataset and PyTorch DataLoader**

Orca will seamlessly parallelize the standard `tf.data.Dataset` or `torch.utils.data.DataLoader` pipelines across a large cluster in a data-parallel fashion, which can be directly used for distributed deep learning training, as shown below:
<TODO: shown a simple example>

Under the hood, Orca will automatically replicate the _TensorFlow Dataset_ or _PyTorch DataLoader_ pipeline on each node in the cluster, shard the input data, and execute the data pipelines using Apache Spark and/or Ray distributedly. 

**Note:** Known limitations include: <TODO: describe known limitations>

#### **1.1. Data Creator Function**
Alternatively, the user may also pass a *Data Creator Function* as the input to the distributed training and inference. Inside the *Data Creator Function*, the user needs to create and return a `tf.data.Dataset` or `torch.utils.data.DataLoader` object, as shown below.
<TODO: shown a simple example>

### **2. Spark Dataframes**
Orca supports Spark Dataframes as the input to the distributed training, and as the input/output of the distributed inference. Consequently, the user can easily process large-scale dataset using Apache Spark, and directly apply AI models on the distributed (and possibly in-memory) Dataframes without data conversion or serialization. 

<TODO: shown a simple example, explain the input and output dataframe format>

### **3. XShards (Distributed Data-Parallel Python Processing)**

`XShards` in Orca allows the user to process large-scale dataset using *existing* Pyhton codes in a distributed and data-parallel fashion, as shown below. 
<TODO: show a simple example, maybe using `XShards.partition` and `XShards.transform_partition`?>

In essence, an `XShards` contains an automatically sharded (or partitioned) Python object (e.g., Pandas Dataframe, Numpy NDArray,  Python Dictionary or List, etc.). Each partition of the `XShards` stores a subset of the Python object and is distributed across different nodes in the cluster; and the user may run arbitrary Python codes on each partition in a data-parallel fashion using `XShards.transform_partition`.

View the related [Python API doc]() for more details.
 
#### **3.1 Data-Parallel Pandas**
The user may use `XShards` to efficiently process large-size Pandas Dataframes in a distributed and data-parallel fashion.

First, the user can read CVS, JSON or Parquet files (stored on local disk, HDFS, AWS S3, etc.) to obtain an `XShards` of Pandas Dataframe, as shown below:
<TODO: shown a simple pandas read example>

Each partition of the returned `XShards` stores a Pandas Dataframe object (containing a subset of the entire dataset), and then the user can apply Pandas operations as well as other (e.g., sklearn) operations on each partition, as shown below:   
<TODO: shown a simple `transform_partition` example>

In addition, some global operations  (such as `partition_by`, `unique`, etc.) are also supported on the `XShards` of Pandas Dataframe, as shown below:
<TODO: shown a simple `partition_by`, `unique` example>
