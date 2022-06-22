### Preprocess and Train Two Tower Model Using Movielens Data

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
```

Step 3. Dealing with missing data
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
```

Step 5. Encode categorical features
```python
user_tbl, inx_list = user_tbl.category_encode(["gender", "age", "zipcode", "occupation"])
item_tbl, item_list = item_tbl.category_encode(["genres"])
```

Step 6. Add negative samples into ratings table
```python
item_size = item_stats.select("item").distinct().size()
ratings_tbl = ratings_tbl.add_negative_samples(item_size=item_size, item_col="item", label_col="label", neg_num=1)
```

Step 7. Join all features together into ratings table.
```python
user_tbl = user_tbl.join(user_stats, on="user")
item_tbl = item_tbl.join(item_stats, on="item")
full = ratings_tbl.join(user_tbl, on="user").join(item_tbl, on="item")
```

Step 8. Train and test split
```python
cat_features = ["user", "item", "zipcode", "gender", "age", "occupation", "zipcode", "genres"]
full = full.select("label", *cat_features,
                   array("user_visits", "user_mean_rate").alias("user_num"),
                   array("item_visits", "item_mean_rate").alias("item_num"))
train_tbl, test_tbl = full.random_split([0.8, 0.2], seed=1)
```
Step 9. Prepare feature stats and create two tower model
```python
from friesian.example.two_tower.model import ColumnInfoTower, TwoTowerModel, get_1tower_model
stats = full.get_stats(cat_features, "max")
user_info = ColumnInfoTower(indicator_cols=["gender", "age", "occupation"],
                            indicator_dims=[stats["gender"], stats["age"], stats["occupation"]],
                            embed_cols=["user", "zipcode"],
                            embed_in_dims=[stats["user"], stats["zipcode"]],
                            embed_out_dims=[16, 16],
                            numerical_cols=["user_num"],
                            numerical_dims=[2],
                            name="user")
item_info = ColumnInfoTower(indicator_cols=["genres"],
                            indicator_dims=[stats["genres"]],
                            embed_cols=["item"],
                            embed_in_dims=[stats["item"]],
                            embed_out_dims=[16],
                            numerical_cols=["item_num"],
                            numerical_dims=[2],
                            name="item")
two_tower = TwoTowerModel(user_info, item_info)
def model_creator(config):
    model = two_tower.build_model()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'Recall', 'AUC'])
    return model
```
Step 10. Create estimator and train the two_tower model
```python
from bigdl.orca.learn.tf2.estimator import Estimator

estimator = Estimator.from_keras(model_creator=model_creator)

train_count, test_count = train_tbl.size(), test_tbl.size()
feature_cols = user_info.get_name_list() + item_info.get_name_list()

estimator.fit(data=train_tbl.df,
              epochs=1,
              batch_size=batch_size,
              steps_per_epoch=train_count // batch_size,
              validation_data=test_tbl.df,
              validation_steps=test_count // batch_size,
              feature_cols=feature_cols,
              label_cols=['label'])
```
