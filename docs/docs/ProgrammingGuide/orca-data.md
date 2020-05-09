---
## **Introduction**

Analytics Zoo orca data provides data-parallel pre-processing support for AI.

It supports data pre-processing from different data sources, like TensorFlow DataSet, PyTorch DataLoader, MXNet DataLoader, etc. and it supports different data format, like Pandas DataFrame, Numpy, Images, Parquet.

The backend distributed preprocessing engine can be [Spark](https://spark.apache.org/) or [Ray](https://github.com/ray-project/ray).

In current version, orca data API only supports parallel pre-processing with Pandas DataFrame on Ray.

---
## **DataShards**

DataShards is a collection of data in orca data API. In current version, the element in DataShards is a Pandas DataFrame.

### **DataShards with Pandas DataFrame**

#### **Read data into DataShards**

You can read csv/json files/directory into DataShards with such APIs:
```
zoo.orca.data.pandas.read_csv(file_path, context)

zoo.orca.data.pandas.read_json(file_path, context)
```
* The `file_path` could be a csv/json file, multiple csv/json file paths seperated with comma, a directory containing csv/json files. Supported file systems are local file system,` hdfs`, and `s3`.
* `context` is a Ray context.

After calling these APIs, you would get a DataShards of Pandas DataFrame.

#### **Pre-processing on DataShards**

You can do pre-processing with your customized function on DataShards using below API:
```
apply(func, *args)
```
* `func` is your pre-processing function. In this function, you can do the pre-processing on a Pandas DataFrame, then return the processed object. 
* `args` are the augurments for the pre-processing function.

This method would parallelly pre-process each element in the DataShards with the customized function.

#### **Get all the elements in DataShards**

You can get all of elements in DataShards with such API:
```
collect()
```
This method returns a list that contains all of the elements in this DataShards.

To get the more examples on orca data API, you can refert to [Example](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/orca/data)
