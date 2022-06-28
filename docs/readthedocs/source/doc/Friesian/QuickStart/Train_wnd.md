# **Feature Engineering for W&D (Wide and Deep Learning) using Friesian**
With Friesian Table APIs, data scientists and machine learning engineers are able to quickly process datasets of all sizes to generate features for recommender models. This tutorial showcases how to extract features for [Wide and Deep Learning](https://arxiv.org/abs/1606.07792) using [movielens dataset](http://files.grouplens.org/datasets/movielens/).

## **Key Concepts**
**Wide and Deep Learning** is a noval recommender model proposed by Google. Specifically, A W&D model is jointly trained wide linear models and deep neural networks---to combine the benefits of memorization and generalization for recommender systems.
Know more about W&D and it's implementation [here](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/wnd)

A **FeatureTable** is a distributed collection of data, it provides rich data processing and feature engineering methods for recommender systems.
- Categorical Data Encoding, FeatureTable can encode categorical data into integers, one hot encodings.
- Cross-product Feature Transformations, FeatureTable crosses categorical columns and hash into certain buckets to generate more features.
- Negative Sampling, FeatureTable selects negative examples randomly from the user’s non-interactive product set.

## Quik Start
See the full demo code [here](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/democode/train_wnd.py)

Step 1. Initialize OrcaContext
```python
sc = init_orca_context("local",  cores=8, memory="8g", init_ray_on_spark=True)
```

Step 2. Creating a ratings table, user table and an item table
```python
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.feature.dataset import movielens
import pandas as pd

data_dir = "./movielens"
ratings = movielens.get_id_ratings(data_dir)
ratings = pd.DataFrame(ratings, columns=["user", "item", "rate"])      
ratings_tbl = FeatureTable.from_pandas(ratings)

user_df = pd.read_csv(data_dir + "/ml-1m/users.dat", delimiter="::", names=["user", "gender", "age", "occupation", "zipcode"])
user_tbl = FeatureTable.from_pandas(user_df).cast(["user"], "int")

item_df = pd.read_csv(data_dir + "/ml-1m/movies.dat", encoding="ISO-8859-1", delimiter="::", names=["item", "title", "genres"])
item_tbl = FeatureTable.from_pandas(item_df).cast("item", "int")
ratings_tbl.show(3, False)
user_tbl.show(3, False)
item_tbl.show(3, False)
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
#
# +----+-----------------------+----------------------------+
# |item|title                  |genres                      |
# +----+-----------------------+----------------------------+
# |1   |Toy Story (1995)       |Animation|Children's|Comedy |
# |2   |Jumanji (1995)         |Adventure|Children's|Fantasy|
# |3   |Grumpier Old Men (1995)|Comedy|Romance              |
# +----+-----------------------+----------------------------+
```

Step 3. Deal with missing data
FeatureTable can replace null values with a specified value or just simply drop the records with null values.
```python
user_tbl = user_tbl.fillna('0', "zipcode")
```

Step 4. Generate continuous features and normalize them
```python
user_stats = ratings_tbl.group_by("user", agg={"item": "count", "rate": "mean"}) \
        .rename({"count(item)": "user_visits", "avg(rate)": "user_mean_rate"})
user_stats, user_min_max = user_stats.min_max_scale(["user_visits", "user_mean_rate"])

item_stats = ratings_tbl.group_by("item", agg={"user": "count", "rate": "mean"}) \
        .rename({"count(user)": "item_visits", "avg(rate)": "item_mean_rate"})
user_stats.show(3, False)
item_stats.show(3, False)
# 
# +----+-----------+--------------+
# |user|user_visits|user_mean_rate|
# +----+-----------+--------------+
# |148 |0.26329556 |0.6886728     |
# |463 |0.04489974 |0.50274247    |
# |471 |0.037053183|0.6619721     |
# +----+-----------+--------------+
#
# +----+--------------+-----------+
# |item|item_mean_rate|item_visits|
# +----+--------------+-----------+
# |1580|0.6849882     |0.7402976  |
# |2366|0.66402113    |0.2203093  |
# |1088|0.57787484    |0.20017508 |
# +----+--------------+-----------+
```

Step 5. Encode categorical features
```python
user_tbl, inx_list = user_tbl.category_encode(["gender", "age", "zipcode", "occupation"])
item_tbl, item_list = item_tbl.category_encode(["genres"])
user_tbl.show(3, False)
item_tbl.show(3, False)
#
# +----+------+---+-------+----------+
# |user|gender|age|zipcode|occupation|
# +----+------+---+-------+----------+
# |1   |1     |4  |345    |9         |
# |2   |2     |3  |2775   |21        |
# |3   |2     |2  |1971   |19        |
# +----+------+---+-------+----------+
#
# +----+------+
# |item|genres|
# +----+------+
# |1   |257   |
# |2   |32    |
# |3   |160   |
# +----+------+
```

Step 6. Negative Sampling.
Add a label of 1 for current records, as well as corresponding non-clicked items with label of 0. 
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

Step 7. Add more categorical features by crossing different columns
Cross ["gender", "age"] into "gender_age", ["age", "zipcode"] into "age_zipcode" to generate more features for wide model.
```python
user_tbl = user_tbl.cross_columns([["gender", "age"], ["age", "zipcode"]], [50, 200])
user_tbl.show(3, False)
# 
# +----+------+---+-------+----------+----------+-----------+
# |user|gender|age|zipcode|occupation|gender_age|age_zipcode|
# +----+------+---+-------+----------+----------+-----------+
# |1   |1     |4  |345    |9         |36        |113        |
# |2   |2     |3  |2775   |21        |46        |159        |
# |3   |2     |2  |1971   |19        |45        |135        |
# +----+------+---+-------+----------+----------+-----------+
# 
```
FeatureTable can also cross multiple columns into one column and hash into a number of buckets by calling `cross_hash_encode`

Step 8. Join all features together.
```python
user_tbl = user_tbl.join(user_stats, on="user")
item_tbl = item_tbl.join(item_stats, on="item")
full = ratings_tbl.join(user_tbl, on="user").join(item_tbl, on="item")
full.show(3, False)
#
# +----+----+----+-----+------+---+-------+----------+----------+-----------+-----------+--------------+------+--------------+-----------+
# |item|user|rate|label|gender|age|zipcode|occupation|gender_age|age_zipcode|user_visits|user_mean_rate|genres|item_mean_rate|item_visits|
# +----+----+----+-----+------+---+-------+----------+----------+-----------+-----------+--------------+------+--------------+-----------+
# |148 |1088|4   |0    |1     |4  |1139   |9         |36        |33         |0.5039233  |0.58825946    |76    |0.4456522     |0.006419609|
# |148 |5156|3   |0    |2     |6  |2261   |18        |49        |172        |0.13077594 |0.81147605    |76    |0.4456522     |0.006419609|
# |148 |897 |5   |0    |2     |2  |1257   |4         |45        |152        |0.07977332 |0.64375305    |76    |0.4456522     |0.006419609|
# +----+----+----+-----+------+---+-------+----------+----------+-----------+-----------+--------------+------+--------------+-----------+

Step 9. Train and test split
```python
train_tbl, test_tbl = full.random_split([0.8, 0.2], seed=1)
```

Step 10. Summarize features and their dimensions for W&D
```python
from friesian.example.wnd.train.wnd_train_recsys import ColumnFeatureInfo, model_creator
wide_cols = ["gender", "age", "occupation", "zipcode", "genres"]
wide_cross_cols = ["gender_age", "age_zipcode"]
indicator_cols = ["gender", "age", "occupation", "genres"]
embed_cols = ["user", "item"]
num_cols = ["user_visits", "user_mean_rate", "item_visits", "item_mean_rate"]
cat_cols = wide_cols + wide_cross_cols + embed_cols
stats = full.get_stats(cat_cols, "max")
wide_dims = [stats[key] for key in wide_cols]
wide_cross_dims = [stats[key] for key in wide_cross_cols]
embed_dims = [stats[key] for key in embed_cols]
indicator_dims = [stats[key] for key in indicator_cols]
column_info = ColumnFeatureInfo(wide_base_cols=wide_cols,
                                wide_base_dims=wide_dims,
                                wide_cross_cols=wide_cross_cols,
                                wide_cross_dims=wide_cross_dims,
                                indicator_cols=indicator_cols,
                                indicator_dims=indicator_dims,
                                embed_cols=embed_cols,
                                embed_in_dims=embed_dims,
                                embed_out_dims=[8] * len(embed_dims),
                                continuous_cols=num_cols,
                                label="label")
```

Step 11. Create wide_and_deep model, and estimator 
```python
from bigdl.orca.learn.tf2.estimator import Estimator
from friesian.example.wnd.train.wnd_train_recsys import  model_creator
conf = {"column_info": column_info, "hidden_units": [20, 10], "lr": 0.001}
est = Estimator.from_keras(model_creator=model_creator, config=conf)
```

Step 12. Train wide_and_deep model 
```python
train_count, test_count = train_tbl.size(), test_tbl.size()
est.fit(data=train_tbl.df,
        epochs=1,
        batch_size=batch_size,
        steps_per_epoch=train_count // batch_size,
        validation_data=test_tbl.df,
        validation_steps=test_count // batch_size,
        feature_cols=column_info.feature_cols,
        label_cols=['label'])
```