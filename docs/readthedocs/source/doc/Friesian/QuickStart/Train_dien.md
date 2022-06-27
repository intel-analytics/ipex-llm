# **Feature Engineering for DIEN using Friesian**
With Friesian Table APIs, data scientists and machine learning engineers are able to quickly process datasets of all sizes to generate features for recommender models. This tutorial showcases how to extract features for Deep Interest Evolution Network [DIEN](https://arxiv.org/pdf/1809.03672.pdf) using [movielens dataset](http://files.grouplens.org/datasets/movielens/).

## **Key Concepts**
**DIEN** is a noval recommender models proposed by Alibaba. Specifically, DIEN has an interest extractor layer to capture temporal interests from history behavior sequence, and an interest evolving layer to model interest evolving process that is relative to target item. Know more about [DIEN](https://arxiv.org/pdf/1809.03672.pdf), and it's optimization from [Intel](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/dien).

A **FeatureTable** is a distributed collection of data, it provides rich data processing and feature engineering functions for recommender systems.
- Categorical Data Encoding, FeatureTable can encode categorical data into integers, ont hot encodings.
- Negative Sampling, FeatureTable selects negative examples randomly from the user’s non-interactive product set.
- FeatureTable can extract behavior sequences for users to capture their temporal interests, as well as the evolving process.

## Quik Start
See the full demo code [here](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/democode/train_dien.py)

Step 1. Initialize OrcaContext
```python
sc = init_orca_context("local",  cores=8, memory="8g", init_ray_on_spark=True)
```

Step 2. Create a ratings table and an item table
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
Step 3. Encode movie categories 
```python
cat_indx = item_tbl.gen_string_idx("category", freq_limit=1)
item_tbl = item_tbl.encode_string("category", cat_indx)
item_tbl.show(3, False)
# +----+--------+
# |item|category|
# +----+--------+
# |1   |257     |
# |2   |32      |
# |3   |160     |
# +----+--------+
```
FeatureTable provides different ways to encode categorical features, instead of using `gen_string_idx()` and `encode_string()', you can also use `category_encode()` to transform categorical strings to integer values.

Step 4. Generate history behavior sequences for each user.
History behavior sequences of items are generated to capture users' temporal interests, as well as the evolving process. 
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
Each user's item sequences are sorted by `'time'` and most recently visited are chosen to represent behavior sequence. You can set `num_seqs = number` to choose a number of recent behaviors.
`seq_length = 6` showcases what data looks like, it should be a relatively larger number based on user behaviors to capture temporal interests, and it's evolving process.

Step 5. Generate non-clicked sample sequences for each behavior sequence
Besides the clicked behavior sequences, DIEN also needs non-clicked sequences that samples from item set except the clicked item.
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

Step 6. Negative Sampling.
Add a label of 1 for current records, as well as corresponding non-clicked items with label of 0. 
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
`neg_num =1` means for each positive record, add 1 negative record.

Step 7. Add a category value for each item in the data.
For each item value in the columns of `["item", "item_hist_seq", "neg_item_hist_seq"]`, add a corresponding category value.
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

Step 8. Organize data into model needed format.
Pad sequences into a fixed length, add zeros or any values needed at the end if a user's history visits has a length smaller than `seq_len`.
Transform labels into array.
```python
full_tbl = full_tbl\
    .pad(cols=['item_hist_seq', 'category_hist_seq',
                   'neg_item_hist_seq', 'neg_category_hist_seq'],
             seq_len=seq_length,
             mask_cols=['item_hist_seq'])\
    .apply("label", "label", lambda x: [1 - float(x), float(x)], "array<float>")
full_tbl.show(3, False)

# +----+------------------------------------+-----------------+------------------------------------------------------------------------------------------------------------------+----+----------+--------+------------------------------+-------------------------------------------------------------------------------------------------+------------------+
# |user|item_hist_seq                       |item_hist_seq_len|neg_item_hist_seq                                                                                                   |item|label     |category|category_hist_seq             |neg_category_hist_seq                                                                         |item_hist_seq_mask|
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+----------+--------+------------------------------+----------------------------------------------------------------------------------------------+------------------+
# |148 |[3646, 3821, 1950, 3730, 3507, 2135]|6                |[[238, 291, 3018], [1702, 2990, 1098], [823, 1349, 492], [2885, 492, 906], [3069, 2943, 369], [3499, 1566, 465]]    |1310|[1.0, 0.0]|88      |[266, 266, 260, 260, 266, 100]|[[65, 26, 233], [5, 276, 266], [76, 233, 267], [135, 267, 85], [76, 135, 76], [233, 117, 109]]|[1, 1, 1, 1, 1, 1]|
# |148 |[3646, 3821, 1950, 3730, 3507, 2135]|6                |[[238, 291, 3018], [1702, 2990, 1098], [823, 1349, 492], [2885, 492, 906], [3069, 2943, 369], [3499, 1566, 465]]    |3270|[0.0, 1.0]|76      |[266, 266, 260, 260, 266, 100]|[[65, 26, 233], [5, 276, 266], [76, 233, 267], [135, 267, 85], [76, 135, 76], [233, 117, 109]]|[1, 1, 1, 1, 1, 1]|
# |463 |[2413, 2468, 520, 2253, 2379, 2458] |6                |[[3294, 2214, 765], [2071, 2259, 706], [1641, 324, 3274], [3870, 2639, 2041], [1316, 2550, 497], [3482, 2685, 1103]]|3474|[1.0, 0.0]|101     |[267, 178, 266, 236, 266, 111]|[[233, 26, 299], [76, 160, 76], [266, 266, 276], [76, 76, 155], [76, 98, 160], [76, 299, 76]] |[1, 1, 1, 1, 1, 1]|
# +----+------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------------+----+----------+--------+------------------------------+----------------------------------------------------------------------------------------------+------------------+
```

Step 9. **Train and test split**
```python
train_tbl, test_tbl = full_tbl.random_split([0.8, 0.2], seed=1)
```
You can also use `full_tbl.filter(condition)` to split train and test data 

Step 10. Build DIEN model
Intel defines the [DIEN](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/dien) using tensorflow based on [ai-matrix](https://github.com/alibaba/ai-matrix/tree/master/macro_benchmark/DIEN) and [x-deeplearning](https://github.com/alibaba/x-deeplearning/blob/master/xdl-algorithm-solution/DIEN/script/README.md)
```python
from friesian.example.dien.dien_train import build_model
model = build_model("DIEN", user_size, item_size, cat_size, 0.001, "FP32")
```

Step 11. Create estimator and train DIEN model
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