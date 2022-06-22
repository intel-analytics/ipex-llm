# Friesian Feature Engineering Overview

## **Overview**
Friesian Table provides a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems.

It provides high-level abstraction to simplify code and accelerates computation on Intel CPU. With Table recommender focused APIs, data scientists and machine learning engineers are able to quickly process datasets of all sizes.

Processed features are feed into Orca Estimator to train recommender models, and loaded into Redis online inference. 

## **Key Concepts**

### **Table**
A Table is a distributed collection of data. It is built on top of [spark dataframe](https://spark.apache.org/docs/latest/sql-programming-guide.html#datasets-and-dataframes), with richer abstraction and optimizations under the hood, specifically for recommender systems.

### **FeatureTable**
FeatureTable is built on top of Friesian Table, it provides rich data processing and feature engineering functions for recommender systems.

FeatureTable.df is then feed into [Orca Estimator](../../Orca/Overview/distributed-training-inference.md) for training purpose.

### **StringIndex**
A StringIndex is a Friesian Table with unique index values of categorical features.

A StringIndex is then used to transform categorical features of FeatureTable to integer values. 

### **TargetCode**
A TargetEncode is a Friesian Table, with representation of categorical data using target encoding. 
Target encoding allows us to retain actual useful information about the categories while keeping the dimensionality of our data the same as the unencoded data. To target encode data, for each feature, we simply replace each category with the mean target value for samples which have that category.

A TargetCode is then used to transform categorical features of FeatureTable to mean statics.

## **Create a Friesian Table**

Friesian FeatureTables can be created from spark dataframe, pandas dataframe, parquet file, jason files, csv files and text files.

### **Create a FeatureTable from dataframe**
```python
# Create a FeatureTable from spark dataframe
from bigdl.friesian.feature import FeatureTable
feature_tbl = FeatureTable(spark_df)

# create a FeatureTable from pandas dataframe
feature_tbl = FeatureTable.from_pandas(pd_df)
```

### **Create a FeatureTable from files**
```python
from bigdl.friesian.feature import FeatureTable   
# create a FeatureTable from parquet files
tbl1 = FeatureTable.read_parquet("/path/to/input_file")

# create a FeatureTable from jason files**
tbl2 = FeatureTable.read_json("/path/to/input_file", cols=None)

# create a FeatureTable from csv files**
tbl3 = FeatureTable.read_csv("/path/to/input_file", delimiter=",", header=False, names=None, dtype=None)

# create a FeatureTable fromtext files**
tbl4 = FeatureTable.read_text("/path/to/input_file", col_name="value")
```

### **Generate a StringIndex**
A StringIndex could be generated from FeatureTable generated from FeatureTable, could be loaded from parquet files `str_idx=StringIndex.read_parquet("/path/to/input_file") or dictionaries.

```python
from bigdl.friesian.feature import StringIndex
string_idx_list = feature_tbl.gen_string_idx(["user", "item"])
letter_index = StringIndex.from_dict(indices, col_name)
```

### **Create a TargetEncode**
A TargetCode is generated from FeatureTable.
```python
tbl, target_codes = tbl.target_encode(cat_cols="count", target_cols="label")
```

## **Quick Start**
FeatureTable provides a library of unified and easy-to-use APIs for data preprocessing and feature engineering. It inherits common spark dataframe functions, and provides complex feature generation.
A couple of end to end training pipelines are provided here to showcase how to use Friesian Table to generate features for popular recommender models.
See full demo code [here](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/democode/)

[Preprocess and Train Two Tower Model Using Movielens Data](Train_2tower.md)

[Preprocess and Train Wide and Deep Model Using Movielens Data](Train_wnd.md)