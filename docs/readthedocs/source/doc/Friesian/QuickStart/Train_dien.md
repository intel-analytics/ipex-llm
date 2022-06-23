### ** Feature Engineering for DIEN using Friesian**

Step 1. Initialize OrcaContext
```python
sc = init_orca_context("local",  cores=8, memory="8g", init_ray_on_spark=True)
```

Step 2. Creating a ratings table and an item table
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
```

Step 3. Generate indices for movie category 
```python
cat_indx = item_tbl.gen_string_idx("category", freq_limit=1)
item_tbl = item_tbl.encode_string("category", cat_indx)
```

Step 4. Generate historical visits of items for each user.
```python
full_tbl = ratings_tbl
    .add_hist_seq(cols=['item'], user_col="user",
                  sort_col='time', min_len=1, max_len=seq_length, num_seqs=1)\
    .append_column("item_hist_seq_len", lit(seq_length))
```

Step 5. Generate non click items for each history visit
```python 
item_size = ratings_tbl.get_stats("item", "max")["item"] + 1
full_tbl = full_tbl.add_neg_hist_seq(item_size, 'item_hist_seq', neg_num=5) 
```

Step 6. Add negative samples.
```python
full_tbl = full_tbl.add_negative_samples(item_size, item_col='item', neg_num=1) 
```

Step 7. Add a category value for each item in the data
```python
full_tbl = full_tbl.add_value_features(columns=["item", "item_hist_seq", "neg_item_hist_seq"],
                                                dict_tbl=item_tbl, key="item", value="category")
```

Step 8. Pad all the sequence data into a length and add a mask column based on historical item visits
```python
full_tbl = full_tbl.pad(cols=['item_hist_seq', 'category_hist_seq',
                   'neg_item_hist_seq', 'neg_category_hist_seq'],
             seq_len=seq_length,
             mask_cols=['item_hist_seq']) 
```
Step 9. Organize labels into array 
```python
full_tbl = full_tbl.apply("label", "label", lambda x: [1 - float(x), float(x)], "array<float>")
```

Step 10. **Train and test split**
```python
train_tbl, test_tbl = full_tbl.random_split([0.8, 0.2], seed=1)
```

Step 10. Build model DIEN
```python
from friesian.example.dien.dien_train import build_model
model = build_model("DIEN", user_size, item_size, cat_size, 0.001, "FP32")
```

Step 11. Create estimator 
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
```

Step 12. Train DIEN model 
```python
estimator.fit(train_tbl.df, epochs=1, batch_size=batch_size,
                  feature_cols=feature_cols, label_cols=['label'], validation_data=test_tbl.df)
```