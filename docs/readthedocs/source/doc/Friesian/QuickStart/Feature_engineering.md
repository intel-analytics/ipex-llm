# Friesian Feature Engineering Overview

## **Overview**
Friesian Table provides a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems.

It provides high-level abstraction to simplify code and accelerates computation on Intel CPU. With Table recommender focused APIs, data scientists and machine learning engineers are able to quickly process datasets of all sizes.

Processed features are feed into Orca Estimator to train recommender models, and then loaded into Redis for online inference. 

## **Key Concepts**
A **Friesian Table** is a distributed collection of data. It is built on top of [spark dataframe](https://spark.apache.org/docs/latest/sql-programming-guide.html#datasets-and-dataframes), with richer abstraction and optimizations under the hood, specifically for recommender systems.

**FeatureTable** is built on top of Friesian Table.
- A FeatureTable provides rich data processing and feature engineering methods specifically for recommender models. 
- FeatureTable.df is then feed into [Orca Estimator](../../Orca/Overview/distributed-training-inference.md) for training purpose.

A **StringIndex** is a **Friesian Table** with unique index values for categorical features, it represents a mapping from categorical values to `id` of integers.
- Frisian FeatureTable could generate StringIndex based feature frequency.
- Integer `id`  starts from 1, reserving 0 for unknown features.
- A StringIndex is then used to transform categorical features of Friesian FeatureTable to integer values. 

A **TargetEncode** is a Friesian Table with representation of categorical data using target encodes. 
- Target encoding allows us to retain actual useful information about the categories while keeping the dimensionality of data the same as the unencoded data. 
- To target encode data, **Friesian FeatureTable** simply replaces each category with the mean target value for samples which have that category.

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
A couple of end to end training pipelines are provided here to showcase how to use **Friesian Table** to generate features for popular recommender models.
See full demo code [here](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/democode/)


[Feature Engineering for W&D (Wide and Deep Learning) using Friesian](Train_wnd.md)