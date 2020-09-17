---
## **Introduction**

Analytics Zoo Orca data provides data-parallel pre-processing support for Python AI.

It supports data pre-processing from different data sources, like TensorFlow DataSet, PyTorch DataLoader, MXNet DataLoader, etc. and it supports various data formats, like Pandas DataFrame, Numpy, Images, Parquet, etc.

The distributed backend engine can be [Spark](https://spark.apache.org/) or [Ray](https://github.com/ray-project/ray). We now support Spark-based transformations to do the pre-processing, and provide functionality to seamlessly put data to Ray cluster for later training/inference on Ray. 

---
## **XShards**

XShards is a collection of data in Orca data API. We provide different backends(Spark and Ray) for XShards.

### **XShards General Operations**

#### **Pre-processing on XShards**

You can do pre-processing with your customized function on XShards using below API:
```
transform_shard(func, *args)
```
* `func` is your pre-processing function. In this function, you can do the pre-processing with the data using common Python libraries such as Pandas, Numpy, PIL, TensorFlow Dataset, PyTorch DataLoader, etc., then return the processed object. 
* `args` are the augurments for the pre-processing function.

This method would parallelly pre-process each element in the XShards with the customized function, and return a new XShards after transformation.

##### **SharedValue**
SharedValue can be used to give every node a copy of a large input dataset in an efficient manner.
This is an example of using SharedValue:
```
def func(df, item_set)
   item_set = item_set.value
   ....

item_set= ...
item_set= orca.data.SharedValue(item_set)
full_data.transform_shard(func, item_set)
```

#### **Get all the elements in XShards**

You can get all of elements in XShards with such API:
```
collect()
```
This method returns a list that contains all of the elements in this XShards. 


#### **Repartition XShards**

You can repartition XShards to different number of partitions.
```
repartition(num_partitions)
```
* `num_partitions` is the target number of partitions for the new XShards.

The method returns a new XShards that has exactly num_partitions partitions.


#### **Split XShards**

You can split one XShards into multiple XShards. Each element in the XShards needs be a list or tuple with same length.
```
split()
```
This method returns splits of XShards. If each element in the input SparkDataShard is not a list or tuple, return list of input SparkDataShards.

#### **Save/Load XShards**

You can save XShards on Spark as SequenceFiles of serialized objects.
The serializer used is pyspark.serializers.PickleSerializer.
```
save_pickle(path, batchSize=10)
```
* `path` is target save path.
* `batchSize` batch size for each chunk in sequence file.

And you can load pickle file to XShards if you use save_pickle() to save data.
```
zoo.orca.data.XShards.load_pickle(path, minPartitions=None)
```
* `path`: The pickle file path/directory.
* `minPartitions`: The minimum partitions for the XShards.

This method return an XShards object from pickle files.

#### **Move XShards on Spark to Ray backend**

You can put data of the XShards on Spark to Ray cluster object store for later processing on Ray.
```
to_ray()
```
This method save data of XShards on Spark to Ray object store, and return a new RayXShards which contains plasma ObjectID, the plasma object_store_address and the node IP on each partition.



### **XShards with Pandas DataFrame**

#### **Read data into XShards**

You can read csv/json files/directory into XShards with such APIs:
```
zoo.orca.data.pandas.read_csv(file_path, **kwargs)

zoo.orca.data.pandas.read_json(file_path, **kwargs)
```
* The `file_path` could be a csv/json file, list of multiple csv/json file paths, a directory containing csv/json files. Supported file systems are local file system,` hdfs`, and `s3`.
* `**kwargs` is read_csv/read_json options supported by pandas.
* You can use `OrcaContext.pandas_read_backend = "pandas"` to switch to pandas backend. Reference: [Orca Context](https://analytics-zoo.github.io/master/#Orca/context/)

After calling these APIs, you would get a XShards of Pandas DataFrame on Spark.

**For Cloudera YARN client mode users:**
If you use `pandas` as pandas_read_backend, you should configure `ARROW_LIBHDFS_DIR` before calling read_csv:
1. use `locate libhdfs.so` to find libhdfs.so
2. `export ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64` (replace with the result of `locate libhdfs.so`)
3. use `--conf "spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-5.15.2-1.cdh5.15.2.p0.3/lib64"` to export the environment variable to all executors.

#### **Partition by Pandas DataFrame columns**
You can re-partition XShards of Pandas DataFrame with specified columns.
```
partition_by(cols, num_partitions=None)
```
* `cols`: DataFrame columns to partition by.
* `num_partitions`: target number of partitions. If not specified, the new XShards would keep the current partition number.

This method return a new XShards partitioned using the specified columns.

#### **Get unique element list of XShards of Pandas Series**

You can get a unique list of elements of this XShards. This is useful when you want to count/get unique set of some column in the XShards of Pandas DataFrame. 
```
unique()
```
This method return a unique list of elements of the XShards of Pandas Series.

### **XShards with Numpy**

#### **Load local numpy data to XShards**

You can partition local in memory data and form an XShards on Spark.
```
zoo.orca.data.XShards.partition(data)
```
* `data`: The local data can be numpy.ndarray, a tuple, list, dict of numpy.ndarray, or a nested structure made of tuple, list, dict with ndarray as the leaf value.

This method returns a XShards which dispatch local data in parallel on Spark.



