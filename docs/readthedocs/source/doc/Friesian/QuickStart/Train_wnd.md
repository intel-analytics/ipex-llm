### ** Preprocess and Train Wide and Deep Model Using Movielens Data**
Step 1. to Step 6. are same as [preprocess and train two tower model](Train_2tower.md)
Step 7. Add more categorical features by crossing different columns 
```python
user_tbl = user_tbl.cross_columns([["gender", "age"], ["age", "zipcode"]], [50, 200])
```
Step 8. Join all features together into ratings table.
```python
user_tbl = user_tbl.join(user_stats, on="user")
item_tbl = item_tbl.join(item_stats, on="item")
full = ratings_tbl.join(user_tbl, on="user").join(item_tbl, on="item")
```

Step 9. Train and test split
```python
train_tbl, test_tbl = full.random_split([0.8, 0.2], seed=1)
```

Step 10. Prepare feature dimensions for wide and deep model
```python
from friesian.example.wnd.train.wnd_train_recsys import ColumnFeatureInfo, model_creator
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