# Friesian Offline Pipeline Overview

## **Overview**
Friesian Table provides a feature engineering and preprocessing library for tabular data that is designed to easily manipulate terabyte scale datasets and train deep learning (DL) based recommender systems.

It provides high-level abstraction to simplify code and accelerates computation on Intel CPU. With Table recommender focused APIs, data scientists and machine learning engineers are able to quickly process datasets of all sizes.

Processed features are feed into Orca Estimator to train recommender models, and loaded into Redis online inference. 

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

## **Create a Friesian Table**

Friesian FeatureTables can be created from spark dataframe, pandas dataframe, parquet file, jason files, csv files and text files.

### **Create a FeatureTable from dataframe**
```python
# Create a FeatureTable from spark dataframe
# spark_df
# +----+----+-----+-----+
# |user|item|count|label|
# +----+----+-----+-----+
# |   a|   b|    1|    1|
# |   b|   a|    2|    0|
# |   a|   c|    3|    1|
# |   c|   c|    2|    0|
# |   b|   a|    1|    1|
# |   a|   d|    1|    1|
# +----+----+-----+-----+
from bigdl.friesian.feature import FeatureTable
feature_tbl = FeatureTable(spark_df)
# feature_tbl.show()
# +----+----+-----+-----+
# |user|item|count|label|
# +----+----+-----+-----+
# |   a|   b|    1|    1|
# |   b|   a|    2|    0|
# |   a|   c|    3|    1|
# |   c|   c|    2|    0|
# |   b|   a|    1|    1|
# |   a|   d|    1|    1|
# +----+----+-----+-----+

# create a FeatureTable from pandas dataframe
```python
# pd_df
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
string_idx_list = feature_tbl.gen_string_idx(["user", "item"], freq_limit=1)
string_idx_list[0].show()
# +----+---+
# |user| id|
# +----+---+
# |   c|  1|
# |   b|  2|
# |   a|  3|
indices = {'a': 1, 'b': 2, 'c': 3}
col_name = 'user'
letter_index = StringIndex.from_dict(indices, col_name)
```

### **Create a TargetEncode**
A TargetCode is generated from FeatureTable.
```python
feature_tbl, target_codes = feature_tbl.target_encode(cat_cols="count", target_cols="label",target_mean=None, smooth=20, kfold=2,
                                                      fold_seed=None, fold_col="__fold__", drop_cat=False, drop_fold=True,
                                                      out_cols=None)
target_codes[0].show()
# +-----+-------------------+------------------+
# |count|target_encode_count|    count_te_label|
# +-----+-------------------+------------------+
# |    1|                  3|0.7101449275362318|
# |    3|                  1|0.6825396825396824|
# |    2|                  2| 0.606060606060606|
# +-----+-------------------+------------------+
feature_tbl.show()
# +-----+----+----+-----+------------------+
# |count|user|item|label|    count_te_label|
# +-----+----+----+-----+------------------+
# |    1|   a|   b|    1|0.6825396825396824|
# |    2|   b|   a|    0|0.6349206349206349|
# |    3|   a|   c|    1|0.6666666666666666|
# |    2|   c|   c|    0|0.6349206349206349|
# |    1|   b|   a|    1|0.6969696969696969|
# |    1|   a|   d|    1|0.6825396825396824|
# +-----+----+----+-----+------------------+
```

## **Quick Start**
FeatureTable provides a library of unified and easy-to-use APIs for data preprocessing and feature engineering. It inherits common spark dataframe functions, and provides complex feature generation for continuous and categorical data. 
Here is an example of using Friesian Table to do feature engineering on movielens data. See full demo code [here]() 

Creating a rating table, user table and an item table
```python
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.feature.dataset import movielens
import pandas as pd

ratings = pd.read_csv(args.data_dir + "/ml-1m/ratings.dat", delimiter="::",
                      names=["user", "item", "rate", "time"])
ratings_tbl = FeatureTable.from_pandas(ratings).cast(["user", "item", "rate", "time"], "int")

user_df = pd.read_csv(data_dir + "/ml-1m/users.dat", delimiter="::", names=["user", "gender", "age", "occupation", "zipcode"])
user_tbl = FeatureTable.from_pandas(user_df).cast(["user", "age", "occupation", "zipcode"], "int")

item_df = pd.read_csv(args.data_dir + "/ml-1m/movies.dat", encoding="ISO-8859-1",
                      delimiter="::", names=["item", "title", "genres"])
item_tbl = FeatureTable.from_pandas(item_df).cast("item", "int")
```

Once data is loaded into FeatureTable, `FeatureTable.apply(in_col, out_col, func, dtype="string")`  is a convenient way to generate new column based on input columns.
```python
ratings_tbl = ratings_tbl.apply(["rate"], "label", lambda x: 1, "int")
```

Dealing with missing data if needed, could `fillna` with provided value or just fill with median. Movielens data does not have missing value, code is for demo purpose. 
```python
# ratings_tbl = ratings_tbl.fill_median(["rate"])
```

Adding negative samples is necessary for implicit feedback for recommender systems.
```python
ratings_tbl = ratings_tbl.add_negative_samples(item_size=item_size, item_col="item", label_col="label", neg_num=1)
```

Continuous features could be scaled and normalized. For example, here user_stats and item_stats are generated as statistic features is generated and min max scaled.
```python
user_stats = ratings_tbl.group_by("user", agg={"item":"count", "rate":"mean"})\
                        .rename({"count(item)": "user_visits", "avg(rate)": "user_mean_rate"})
user_stats, user_min_max = user_stats.min_max_scale(["user_visits", "user_mean_rate"])
item_stats = ratings_tbl.group_by("item", agg={"user":"count", "rate":"mean"})\
                        .rename({"count(user)": "item_visits", "avg(rate)": "item_mean_rate"})
item_stats, user_min_max = item_stats.min_max_scale(["item_visits", "item_mean_rate"])
```

FeatureTable provides different ways to process categorical features, like `one_hot_encode`, `hash_encode`, `category_encode`, and `target_encode`
For example, here `category_encode` transforms each categorical value into an integer. 
```python 
user_tbl, user_idx = user_tbl.category_encode(columns=["gender", "age", "zipcode"])
item_tbl, item_idx = item_tbl.category_encode(["genres"])
```

Crossing some categorical features into one and bucketizing it adds richer categorical features. 
```python
user_tbl = user_tbl.cross_columns(crossed_columns=[["gender", "age"], ["age", "zipcode"]], bucket_sizes=[50, 200])
```

Adding a sequence of history visits of items and padding them into certain length are useful for sequence or attention based models.
```python
ratings_tbl = ratings_tbl
    .add_hist_seq(cols=['item'], user_col="user", sort_col='time',
                  min_len=1, max_len=10, num_seqs=1)
    .pad(cols="item_hist_seq", seq_len=10)
```

Joining user features, item features with rating table gives the full proprocessed data.
```python
user_tbl = user_tbl.join(user_stats, on="user")
item_tbl = item_tbl.join(item_stats, on="item")
full = ratings_tbl.join(user_tbl, on="user").join(item_tbl, on="item")
full.show(3, False)
```

For more details on the BigDL-Friesian offline pipelines, please refer to [examples on github](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example).