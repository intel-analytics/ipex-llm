---
## **Introduction**

Analytics Zoo orca data provides data-parallel pre-processing support for AI.

It supports data pre-processing from different data sources, like TensorFlow DataSet, PyTorch DataLoader, MXNet DataLoader, etc. and it supports different data format, like Pandas DataFrame, Numpy, Images, Parquet.

The backend distributed preprocessing engine can be [Spark](https://spark.apache.org/) or [Ray](https://github.com/ray-project/ray).

In current version, orca data API only supports parallel pre-processing with Pandas DataFrame on Ray.

---
## **XShards**

XShards is a collection of data in orca data API. In current version, the element in XShards is a Pandas DataFrame.

### **XShards with Pandas DataFrame**

#### **Read data into XShards**

You can read csv/json files/directory into XShards with such APIs:
```
zoo.orca.data.pandas.read_csv(file_path, context)

zoo.orca.data.pandas.read_json(file_path, context)
```
* The `file_path` could be a csv/json file, multiple csv/json file paths seperated with comma, a directory containing csv/json files. Supported file systems are local file system,` hdfs`, and `s3`.
* `context` is a Ray context.

After calling these APIs, you would get a XShards of Pandas DataFrame.

#### **Pre-processing on XShards**

You can do pre-processing with your customized function on XShards using below API:
```
transform_shard(func, *args)
```
* `func` is your pre-processing function. In this function, you can do the pre-processing on a Pandas DataFrame, then return the processed object. 
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

To get the more examples on orca data API, you can refert to [Example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/orca/data)
