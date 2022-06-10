# Friesian Feature Engineering Overview

## **Overview**
Friesian Table provides a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems.

It provides high-level abstraction to simplify code and accelerates computation on Intel CPU. With Table recommender focused APIs, data scientists and machine learning engineers are able to quickly process datasets of all sizes.

## **Key Concepts**

### **Table**
A Table is a distributed collection of data. It is built on top of [spark dataframe](https://spark.apache.org/docs/latest/sql-programming-guide.html#datasets-and-dataframes), with richer abstraction and optimizations under the hood, specifically for recommender systems.

### **FeatureTable**
FeatureTable is built on top of Friesian Table, it provides rich data processing and feature engineering functions for recommender systems.

FeatureTable.df is then feed into [Orca Estimator] for training purpose.

### **StringIndex**
A StringIndex is a Friesian Table with unique index values of categorical features.

A StringIndex is then used to transform categorical features of FeatureTable to integer values. 

### **TargetCode**
A TargetEncode is a Friesian Table, with representation of categorical data using target encoding. 
Target encoding allows us to retain actual useful information about the categories while keeping the dimensionality of our data the same as the unencoded data. To target encode data, for each feature, we simply replace each category with the mean target value for samples which have that category.

A TargetCode is then used to transform categorical features of FeatureTable to mean statics.
## **feature engineering examples**

### **FeatureTable**
Create a FeatureTable from parquet file.
```python
from bigdl.friesian.feature import FeatureTable
feature_tbl = FeatureTable.read_parquet(data_dir)
```
more to come
### **StringIndex**
```python
from bigdl.friesian.feature import FeatureTable
feature_tbl = FeatureTable.read_parquet(file_path)
string_idx_list = feature_tbl.gen_string_idx(["col1", "col2"], freq_limit=None)
```
### **TargetEncode**
```python
from bigdl.friesian.feature import FeatureTable
feature_tbl, target_codes = feature_tbl.target_encode(cat_cols=cat_cols, target_cols=["label"])
feature_tbl = feature_tbl.encode_target(target_cols="label", targets=target_codes)\
```