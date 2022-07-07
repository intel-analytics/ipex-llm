# Feature Engineering for W&D (Wide and Deep Learning) using Friesian
With Friesian FeatureTable APIs, data scientists and machine learning engineers are able to quickly process datasets of all sizes to generate features for recommender models. This tutorial showcases how to extract features for [Wide and Deep Learning](https://arxiv.org/abs/1606.07792) (W&D) using [movielens dataset](http://files.grouplens.org/datasets/movielens/).

## **1. Key Concepts**
- A **FeatureTable** is a distributed table for processing RecSys features; it provides both common tabular operations (such as join) and RecSys feature transformations (such as negative sampling, category encoding).

- **Wide and Deep Learning** is a noval recommender model proposed by Google, it includes a wide linear model and a deep neural network. One can combine the benefits of memorization and generalization by jointly training a wide linear model and a deep neural network.
Know more about W&D and it's implementation [here](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/wnd)

- An [**Orca Estimator**](../../Orca/Overview/distributed-training-inference.md) provides sklearn-style Estimator APIs to perform distributed [TensorFlow](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch), [Keras](https://github.com/keras-team/keras) and [BigDL](https://github.com/intel-analytics/BigDL) training and inference.

## **2. W&D implemendation on Friesian**
See the full demo code [here](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/democode/train_wnd.py)

### **2.1. Initialize OrcaContext**
```python
from bigdl.orca import init_orca_context
sc = init_orca_context("local",  cores=8, memory="8g", init_ray_on_spark=True)
```

### **2.2. Load data into FeatureTable**
Data can be loaded into Friesian FeatureTables from spark dataframe, pandas dataframe, parquet file, jason files, csv files and text files.
```python
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.feature.dataset import movielens
import pandas as pd

data_dir = "./movielens"
ratings = movielens.get_id_ratings(data_dir)
ratings = pd.DataFrame(ratings, columns=["user", "item", "rate"])      
ratings_tbl = FeatureTable.from_pandas(ratings)
ratings_tbl.cache()

user_tbl = FeatureTable.read_csv(data_dir + "/ml-1m/users.dat", delimiter=":")\
    .select("_c0", "_c2", "_c4", "_c6", "_c8")\
    .rename({"_c0": "user", "_c2": "gender", "_c4": "age", "_c6": "occupation", "_c8": "zip"})\
    .cast(["user"], "int")

user_tbl.cache()
ratings_tbl.show(3, False)
user_tbl.show(3, False)
# 
# +----+----+----+
# |user|item|rate|
# +----+----+----+
# |1   |1193|5   |
# |1   |661 |3   |
# |1   |914 |3   |
# +----+----+----+
#
# +----+------+---+----------+-------+
# |user|gender|age|occupation|zipcode|
# +----+------+---+----------+-------+
# |1   |F     |1  |10        |48067  |
# |2   |M     |56 |16        |70072  |
# |3   |M     |25 |15        |55117  |
# +----+------+---+----------+-------+
```

### **2.3. Data processing using FeatureTable**
#### 2.3.1. Deal with missing data
For missing values, FeatureTable can replace them with a specified value or just simply drop the records with null values. For numerical columns, `feature_tbl.fill_median(columns)` can fill missing values with medians.
```python
user_tbl = user_tbl.fillna('0', "zipcode")
```
#### 2.3.2. Scale numerical features
Generate continuous features like user stats and normalize them using min max scale, one can call `feature_tbl.transform_min_max_scale(user_min_max_dict)` to apply scaler to a feature table.
```python
user_stats = ratings_tbl.group_by("user", agg={"item": "count", "rate": "mean"}) \
        .rename({"count(item)": "user_visits", "avg(rate)": "user_mean_rate"})
user_stats, user_min_max_dict = user_stats.min_max_scale(["user_visits", "user_mean_rate"])

user_stats.show(3, False)
# 
# +----+-----------+--------------+
# |user|user_visits|user_mean_rate|
# +----+-----------+--------------+
# |148 |0.26329556 |0.6886728     |
# |463 |0.04489974 |0.50274247    |
# |471 |0.037053183|0.6619721     |
# +----+-----------+--------------+
```

#### 2.3.3. Process categorical features
Converting categorical string data into numeric is essential. One can use `feature_tbl.category_encode("column_name")` to transform categorical data into integers.
```python
user_tbl, inx_list = user_tbl.category_encode(["gender", "age", "zip", "occupation"])
user_tbl.show(3, False)
#
# +----+------+---+-----+----------+
# |user|gender|age|zip  |occupation|
# +----+------+---+-----+----------+
# |1   |1     |4  |345  |9         |
# |2   |2     |3  |2775 |21        |
# |3   |2     |2  |1971 |19        |
# +----+------+---+-------+--------+
```
Generate more categorical features by crossing `["gender", "age"]` into `"gender_age"`, `["age", "zipcode"]` into `"age_zipcode"`.
```python
user_tbl = user_tbl.cross_columns([["gender", "age"], ["age", "zip"]], [50, 200])
user_tbl.show(3, False)
# 
# +----+------+---+----+----------+----------+-------+
# |user|gender|age|zip |occupation|gender_age|age_zip|
# +----+------+---+-------+----------+----------+----+
# |1   |1     |4  |345 |9         |36        |113    |
# |2   |2     |3  |2775|21        |46        |159    |
# |3   |2     |2  |1971|19        |45        |135    |
# +----+------+---+----+----------+----------+-------+
```
One can cross multiple categorical columns and hash into a number of buckets by calling `feature_tbl.cross_hash_encode(columns, bins)`

#### 2.3.4. Negative sampling
Friesian FeatureTable randomly choose one item as negative sample for each positive record when `neg_num=1`, `neg_num` should be larger than 1 if more negative records are needed.
```python
item_size = item_stats.select("item").distinct().size()
ratings_tbl = ratings_tbl.add_negative_samples(item_size=item_size, item_col="item", label_col="label", neg_num=1)
ratings_tbl.show(3, False)
#
# +----+----+----+-----+
# |user|rate|item|label|
# +----+----+----+-----+
# |1   |5   |283 |0    |
# |1   |5   |1193|1    |
# |1   |3   |785 |0    |
# +----+----+----+-----+
```
#### 2.3.5. Join features and train test split
Join user features, item features with ratings to create a full feature table, and randomly split into train and test tables.
```python
user_tbl = user_tbl.join(user_stats, on="user")
full = ratings_tbl.join(user_tbl, on="user").join(item_stats, on="item")

train_tbl, test_tbl = full.random_split([0.8, 0.2])
full.show(3, False)
#
# +----+----+----+-----+------+---+----+----------+----------+-------+-----------+--------------+--------------+-----------+
# |item|user|rate|label|gender|age|zip |occupation|gender_age|age_zip|user_visits|user_mean_rate|item_mean_rate|item_visits|
# +----+----+----+-----+------+---+----+----------+----------+-------+-----------+--------------+--------------+-----------+
# |148 |1088|4   |0    |1     |4  |1139|9         |36        |33     |0.5039233  |0.58825946    |0.4456522     |0.006419609|
# |148 |5156|3   |0    |2     |6  |2261|18        |49        |172    |0.13077594 |0.81147605    |0.4456522     |0.006419609|
# |148 |897 |5   |0    |2     |2  |1257|4         |45        |152    |0.07977332 |0.64375305    |0.4456522     |0.006419609|
# +----+----+----+-----+------+---+-------+----------+----------+-----------+-----------+--------------+------+--------------+-----------+
```

### **2.4. Define W&D model**
Define a W&D model using tensorflow APIs
```python
import tensorflow as tf

def build_model(column_info, hidden_units=[40, 20]):
    """Build an W&D model."""
    # wide model 
    wide_base_input_layers = []
    wide_base_layers = []
    for i in range(len(column_info.wide_base_cols)):
        wide_base_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        wide_base_layers.append(tf.keras.backend.one_hot(wide_base_input_layers[i], column_info.wide_base_dims[i] + 1))

    wide_cross_input_layers = []
    wide_cross_layers = []
    for i in range(len(column_info.wide_cross_cols)):
        wide_cross_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        wide_cross_layers.append(tf.keras.backend.one_hot(wide_cross_input_layers[i], column_info.wide_cross_dims[i]))
    if len(wide_base_layers + wide_cross_layers) > 1:
        wide_input = tf.keras.layers.concatenate(wide_base_layers + wide_cross_layers, axis=1)
    else:
        wide_input = (wide_base_layers + wide_cross_layers)[0]
    wide_out = tf.keras.layers.Dense(1)(wide_input)
    
    # deep model
    indicator_input_layers = []
    indicator_layers = []
    for i in range(len(column_info.indicator_cols)):
        indicator_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        indicator_layers.append(tf.keras.backend.one_hot(indicator_input_layers[i], column_info.indicator_dims[i] + 1))

    embed_input_layers = []
    embed_layers = []
    for i in range(len(column_info.embed_in_dims)):
        embed_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        embedding_layer = tf.keras.layers.Embedding(column_info.embed_in_dims[i] + 1, output_dim=column_info.embed_out_dims[i])
        iembed = embedding_layer(embed_input_layers[i])
        flat_embed = tf.keras.layers.Flatten()(iembed)
        embed_layers.append(flat_embed)
    
    continuous_input_layers = []
    continuous_layers = []
    for i in range(len(column_info.continuous_cols)):
        continuous_input_layers.append(tf.keras.layers.Input(shape=[]))
        continuous_layers.append(tf.keras.layers.Reshape(target_shape=(1,))(continuous_input_layers[i]))

    if len(indicator_layers + embed_layers + continuous_layers) > 1:
        deep_concat = tf.keras.layers.concatenate(indicator_layers + embed_layers + continuous_layers, axis=1)
    else:
        deep_concat = (indicator_layers + embed_layers + continuous_layers)[0]
    linear = deep_concat
    for ilayer in range(0, len(hidden_units)):
        linear_mid = tf.keras.layers.Dense(hidden_units[ilayer])(linear)
        bn = tf.keras.layers.BatchNormalization()(linear_mid)
        relu = tf.keras.layers.ReLU()(bn)
        dropout = tf.keras.layers.Dropout(0.1)(relu)
        linear = dropout
    deep_out = tf.keras.layers.Dense(1)(linear)
    added = tf.keras.layers.add([wide_out, deep_out])
    out = tf.keras.layers.Activation("sigmoid")(added)
    model = tf.keras.models.Model(wide_base_input_layers +
                                  wide_cross_input_layers +
                                  indicator_input_layers +
                                  embed_input_layers +
                                  continuous_input_layers,
                                  out)
    return model

def model_creator(conf):
    model = build_model(column_info)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['AUC', 'Recall'])
return model
```
### **2.4. Distributed Training using Orca Estimator**
Build an [Orca Estimator](../../Orca/Overview/distributed-training-inference.md), and `train_tbl.df` is then feed into the estimator to train the model.
```python
from bigdl.orca.learn.tf2.estimator import Estimator

est = Estimator.from_keras(model_creator=model_creator)
est.fit(data=train_tbl.df, steps_per_epoch=train_steps, validation_steps=test_steps, feature_cols=column_info.feature_cols, label_cols=['label'])
```