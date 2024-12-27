# **Feature Engineering for DIEN using Friesian**
With Friesian Table APIs, data scientists and machine learning engineers are able to quickly process datasets of all sizes to generate features for recommender models. This tutorial showcases how to extract features for Deep Interest Evolution Network [DIEN](https://arxiv.org/pdf/1809.03672.pdf) using [movielens dataset](http://files.grouplens.org/datasets/movielens/).

## **1. Key Concepts**
- A **FeatureTable** is a distributed table for processing RecSys features; it provides both common tabular operations (such as join) and RecSys feature transformations (such as negative sampling, category encoding).

- **DIEN** is a noval recommender models proposed by Alibaba. Specifically, DIEN has an interest extractor layer to capture temporal interests from history behavior sequence, and an interest evolving layer to model interest evolving process that is relative to target item. Know more about [DIEN](https://arxiv.org/pdf/1809.03672.pdf), and it's optimization from [Intel](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/dien).

- An [**Orca Estimator**](../../Orca/Overview/distributed-training-inference.md) provides sklearn-style Estimator APIs to perform distributed [TensorFlow](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch), [Keras](https://github.com/keras-team/keras) and [BigDL](https://github.com/intel-analytics/BigDL) training and inference.

A **FeatureTable** is a distributed collection of data, it provides rich data processing and feature engineering functions for recommender systems.
- Categorical Data Encoding, FeatureTable can encode categorical data into integers, one hot encodings, target encodes.
- Negative Sampling, FeatureTable selects negative examples randomly from the user’s non-interactive product set.
- FeatureTable can extract behavior sequences for users to capture their temporal interests, as well as the evolving process.

## **2. DIEN implemendation on Friesian**
See the full demo code [here](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/democode/train_dien.py)

### **2.1. Initialize OrcaContext**
```python
from bigdl.orca import init_orca_context
sc = init_orca_context("local",  cores=8, memory="8g", init_ray_on_spark=True)
```

### **2.2. Load data into FeatureTable**
Data can be loaded into Friesian FeatureTables from spark dataframe, pandas dataframe, parquet file, jason files, csv files and text files.
Here, ratings, item data are loaded into ratings_tbl, item_tbl.
```python
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.feature.dataset import movielens
import pandas as pd

data_dir = "./movielens"
_ = movielens.get_id_ratings(data_dir)
ratings = pd.read_csv(data_dir + "/ml-1m/ratings.dat", delimiter="::",
                      names=["user", "item", "rate", "time"])
ratings = pd.DataFrame(ratings, columns=["user", "item", "rate", "time"])
ratings_tbl = FeatureTable.from_pandas(ratings).cast(["user", "item", "rate"], "int").cast("time", "long")
item_df = pd.read_csv(data_dir + "/ml-1m/movies.dat", encoding="ISO-8859-1", delimiter="::", names=["item", "title", "genres"])
item_tbl = FeatureTable.from_pandas(item_df).drop("title").rename({"genres": "category"}).cast("item", "int")
ratings_tbl.show(3, False)
item_tbl.show(3, False)
# +----+----+----+---------+
# |user|item|rate|time     |
# +----+----+----+---------+
# |1   |1193|5   |978300760|
# |1   |661 |3   |978302109|
# |1   |914 |3   |978301968|
# +----+----+----+---------+
#
# +----+----------------------------+
# |item|category                    |
# +----+----------------------------+
# |1   |Animation|Children's|Comedy |
# |2   |Adventure|Children's|Fantasy|
# |3   |Comedy|Romance              |
# +----+----------------------------+
```

### **2.3. Data processing using FeatureTable**

#### 2.3.1. Process categorical features
Categorical variables are usually represented as strings and are finite in number, converting these categorical string data into numeric is essential.
One can use `item_tbl.category_encode("category")` to transform the categorical strings into unique integers for each column in column_names, a list of `StringIndex` of mapping from strings to integers is also returned, and should be applied on new feature table when making prediction.

```python
item_tbl, cat_indx = item_tbl.category_encode("category")
item_tbl.show(3, False)
# +----+--------+
# |item|category|
# +----+--------+
# |1   |257     |
# |2   |32      |
# |3   |160     |
# +----+--------+
```
#### 2.3.2. Generate user behavior sequence
[DIEN](https://arxiv.org/pdf/1809.03672.pdf) was proposed capture the user interest from long sequential behavior data. 
Herr, history behavior sequences of items visited are generated to capture users' temporal interests, as well as the evolving process. 
```python
seq_length = 6
full_tbl = ratings_tbl
    .add_hist_seq(cols=['item'], user_col="user",
                  sort_col='time', min_len=1, max_len=seq_length, num_seqs=1)\
    .drop("time", "rate").append_column("item_hist_seq_len", lit(seq_length))
full_tbl.show(3, False)
# +----+----+------------------------------------+-----------------+
# |user|item|item_hist_seq                       |item_hist_seq_len|
# +----+----+------------------------------------+-----------------+
# |148 |3270|[3646, 3821, 1950, 3730, 3507, 2135]|6                |
# |463 |2042|[2413, 2468, 520, 2253, 2379, 2458] |6                |
# |471 |172 |[1320, 327, 1676, 1831, 2034, 256]  |6                |
# +----+----+------------------------------------+-----------------+
```
Each user's item sequences are sorted by `'time'` and most recently visited are chosen to represent behavior sequence. One can set `num_seqs = number` to choose a number of recent behaviors.
`seq_length = 6` showcases what data looks like, it should be a relatively larger number based on user behaviors to capture temporal interests and the evolving process.

Besides using the real behavior as positive instance, DIEN also needs negative instances that samples from item set except the clicked item. One can call `full_tbl.add_neg_hist_seq(item_size, 'item_hist_seq', neg_num=3) ` to randomly generate a sequence of `3` non-clicked samples for each positive item in the behavior sequence.

```python 
item_size = ratings_tbl.get_stats("item", "max")["item"] + 1
full_tbl = full_tbl.add_neg_hist_seq(item_size, 'item_hist_seq', neg_num=3) 
full_tbl.show(3, False)
# +----+----+------------------------------------+-----------------+---------------------------------------------------------------------------------------------------------------------+
# |user|item|item_hist_seq                       |item_hist_seq_len|neg_item_hist_seq                                                                                                    |
# +----+----+------------------------------------+-----------------+---------------------------------------------------------------------------------------------------------------------+
# |148 |3270|[3646, 3821, 1950, 3730, 3507, 2135]|6                |[[238, 291, 3018], [1702, 2990, 1098], [823, 1349, 492], [2885, 492, 906], [3069, 2943, 369], [3499, 1566, 465]]     |
# |463 |2042|[2413, 2468, 520, 2253, 2379, 2458] |6                |[[3294, 2214, 765], [2071, 2259, 706], [1641, 324, 3274], [3870, 2639, 2041], [1316, 2550, 497], [3482, 2685, 1103]] |
# |471 |172 |[1320, 327, 1676, 1831, 2034, 256]  |6                |[[2319, 2152, 2249], [1386, 2443, 2313], [1373, 367, 1442], [928, 1448, 1011], [1950, 1194, 105], [1528, 2458, 2318]]|
# +----+----+------------------------------------+-----------------+---------------------------------------------------------------------------------------------------------------------+
```

#### 2.3.3. Negative Sampling
Friesian FeatureTable randomly sample a small proportion among all the items and mark them as negative evidence. One item is randomly chosen within the `item_size` as negative sample for each positive record when `neg_num=1`, `neg_num` should be larger than 1 if more negative records are needed for training purpose.
```python
full_tbl = full_tbl.add_negative_samples(item_size, item_col='item', neg_num=1) 
full_tbl.show(3, False)
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+-----+
# |user|item_hist_seq                       |item_hist_seq_len|neg_item_hist_seq                                                                                                   |item|label|
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+-----+
# |148 |[3646, 3821, 1950, 3730, 3507, 2135]|6                |[[238, 291, 3018], [1702, 2990, 1098], [823, 1349, 492], [2885, 492, 906], [3069, 2943, 369], [3499, 1566, 465]]    |1310|0    |
# |148 |[3646, 3821, 1950, 3730, 3507, 2135]|6                |[[238, 291, 3018], [1702, 2990, 1098], [823, 1349, 492], [2885, 492, 906], [3069, 2943, 369], [3499, 1566, 465]]    |3270|1    |
# |463 |[2413, 2468, 520, 2253, 2379, 2458] |6                |[[3294, 2214, 765], [2071, 2259, 706], [1641, 324, 3274], [3870, 2639, 2041], [1316, 2550, 497], [3482, 2685, 1103]]|3474|0    |
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+-----+
```

#### 2.3.4. Add value features
For each item in the columns of `["item", "item_hist_seq", "neg_item_hist_seq"]`, add a corresponding category value.
```python
full_tbl = full_tbl.add_value_features(columns=["item", "item_hist_seq", "neg_item_hist_seq"],
                                                dict_tbl=item_tbl, key="item", value="category")
full_tbl.show(3, False)
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+-----+--------+------------------------------+----------------------------------------------------------------------------------------------+
# |user|item_hist_seq                       |item_hist_seq_len|neg_item_hist_seq                                                                                                   |item|label|category|category_hist_seq             |neg_category_hist_seq                                                                         |
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+-----+--------+------------------------------+----------------------------------------------------------------------------------------------+
# |148 |[3646, 3821, 1950, 3730, 3507, 2135]|6                |[[238, 291, 3018], [1702, 2990, 1098], [823, 1349, 492], [2885, 492, 906], [3069, 2943, 369], [3499, 1566, 465]]    |1310|0    |88      |[266, 266, 260, 260, 266, 100]|[[65, 26, 233], [5, 276, 266], [76, 233, 267], [135, 267, 85], [76, 135, 76], [233, 117, 109]]|
# |148 |[3646, 3821, 1950, 3730, 3507, 2135]|6                |[[238, 291, 3018], [1702, 2990, 1098], [823, 1349, 492], [2885, 492, 906], [3069, 2943, 369], [3499, 1566, 465]]    |3270|1    |76      |[266, 266, 260, 260, 266, 100]|[[65, 26, 233], [5, 276, 266], [76, 233, 267], [135, 267, 85], [76, 135, 76], [233, 117, 109]]|
# |463 |[2413, 2468, 520, 2253, 2379, 2458] |6                |[[3294, 2214, 765], [2071, 2259, 706], [1641, 324, 3274], [3870, 2639, 2041], [1316, 2550, 497], [3482, 2685, 1103]]|3474|0    |101     |[267, 178, 266, 236, 266, 111]|[[233, 26, 299], [76, 160, 76], [266, 266, 276], [76, 76, 155], [76, 98, 160], [76, 299, 76]] |
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+-----+--------+------------------------------+----------------------------------------------------------------------------------------------+
```

#### 2.3.5. Pad sequences, transform labels and train-test split
Pad sequences into a fixed length, add zeros or any values needed at the end if a user's history visits has a length smaller than `seq_len`.
Transform labels into array.
```python
full_tbl = full_tbl\
    .pad(cols=['item_hist_seq', 'category_hist_seq',
                   'neg_item_hist_seq', 'neg_category_hist_seq'],
             seq_len=seq_length,
             mask_cols=['item_hist_seq'])\
    .apply("label", "label", lambda x: [1 - float(x), float(x)], "array<float>")

train_tbl, test_tbl = full_tbl.random_split([0.8, 0.2], seed=1)

full_tbl.show(3, False)

# +----+------------------------------------+-----------------+------------------------------------------------------------------------------------------------------------------+----+----------+--------+------------------------------+-------------------------------------------------------------------------------------------------+------------------+
# |user|item_hist_seq                       |item_hist_seq_len|neg_item_hist_seq                                                                                                   |item|label     |category|category_hist_seq             |neg_category_hist_seq                                                                         |item_hist_seq_mask|
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+----------+--------+------------------------------+----------------------------------------------------------------------------------------------+------------------+
# |148 |[3646, 3821, 1950, 3730, 3507, 2135]|6                |[[238, 291, 3018], [1702, 2990, 1098], [823, 1349, 492], [2885, 492, 906], [3069, 2943, 369], [3499, 1566, 465]]    |1310|[1.0, 0.0]|88      |[266, 266, 260, 260, 266, 100]|[[65, 26, 233], [5, 276, 266], [76, 233, 267], [135, 267, 85], [76, 135, 76], [233, 117, 109]]|[1, 1, 1, 1, 1, 1]|
# |148 |[3646, 3821, 1950, 3730, 3507, 2135]|6                |[[238, 291, 3018], [1702, 2990, 1098], [823, 1349, 492], [2885, 492, 906], [3069, 2943, 369], [3499, 1566, 465]]    |3270|[0.0, 1.0]|76      |[266, 266, 260, 260, 266, 100]|[[65, 26, 233], [5, 276, 266], [76, 233, 267], [135, 267, 85], [76, 135, 76], [233, 117, 109]]|[1, 1, 1, 1, 1, 1]|
# |463 |[2413, 2468, 520, 2253, 2379, 2458] |6                |[[3294, 2214, 765], [2071, 2259, 706], [1641, 324, 3274], [3870, 2639, 2041], [1316, 2550, 497], [3482, 2685, 1103]]|3474|[1.0, 0.0]|101     |[267, 178, 266, 236, 266, 111]|[[233, 26, 299], [76, 160, 76], [266, 266, 276], [76, 76, 155], [76, 98, 160], [76, 299, 76]] |[1, 1, 1, 1, 1, 1]|
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+----------+--------+------------------------------+----------------------------------------------------------------------------------------------+------------------+
```

### **2.4. Build DIEN model**
Intel defines the [DIEN](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/dien) using tensorflow based on [ai-matrix](https://github.com/alibaba/ai-matrix/tree/master/macro_benchmark/DIEN) and [x-deeplearning](https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/DIEN/script/README.md)
```python
from friesian.example.dien.dien_train import build_model
model = build_model("DIEN", user_size, item_size, cat_size, 0.001, "FP32")
```

### **2.5. Distributed Training using Orca Estimator**
```python
from bigdl.orca.learn.tf.estimator import Estimator
input_phs = [model.uid_batch_ph, model.mid_his_batch_ph, model.cat_his_batch_ph, model.mask,
             model.seq_len_ph, model.mid_batch_ph, model.cat_batch_ph,
             model.noclk_mid_batch_ph, model.noclk_cat_batch_ph]
feature_cols = ['user', 'item_hist_seq', 'category_hist_seq', 'item_hist_seq_mask',
                'item_hist_seq_len', 'item', 'category',
                'neg_item_hist_seq', 'neg_category_hist_seq']

estimator = Estimator.from_graph(inputs=input_phs, outputs=[model.y_hat],
                                 labels=[model.target_ph], loss=model.loss,
                                 optimizer=model.optim, model_dir=model_dir,
                                 metrics={'loss': model.loss, 'accuracy': model.accuracy})
estimator.fit(train_tbl.df, epochs=1, batch_size=batch_size,
                  feature_cols=feature_cols, label_cols=['label'], validation_data=test_tbl.df)
```