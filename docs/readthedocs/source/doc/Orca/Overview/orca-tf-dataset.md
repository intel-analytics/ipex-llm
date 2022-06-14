# Orca TF Dataset

Orca TF Dataset represents a distributed [TF Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). It can be used as a bridge between [Orca xShards](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/data-parallel-processing.html#xshards-distributed-data-parallel-python-processing) (or [Friesian FeatureTable](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Friesian/feature.html)) and [Orca TF Estimator](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-training-inference.html#tensorflow-keras-estimator) input, it also allows users to do basic data pre-processing. 


## Key Concepts

An **Orca TF Dataset** is a distributed Tensorflow tf.data.Dataset.
- Each element of the Orca TF Dataset is a tf.data.Dataset.
- An Orca TF Dataset has a collection of elements partitioned across the cluster nodes that can be operated on in parallel.
- After the Orca TF Dataset is created, map functions can be applied to the Dataset in parallel. And the Orca TF estimator can do model training, validation, and inference using the created Orca TF Dataset.

## Creating an Orca TF Dataset

An Orca TF Dataset can be created from an Orca xShards or a Friesian FeatureTable.

### From Orca xShards

An Orca TF Dataset can be created from an Orca xShards using `DataSet.from_tensor_slices(xshards)`

```python
sample_xshards
# [{'movie_title': array(["One Flew Over the Cuckoo's Nest (1975)",
#        'James and the Giant Peach (1996)', 'My Fair Lady (1964)',
#        'Erin Brockovich (2000)', "Bug's Life, A (1998)"], dtype='<U38'), 
#   'user_id': array([1, 1, 1, 1, 1], dtype=int32), 
#   'user_rating': array([5, 3, 3, 4, 5], dtype=int32)}]

ds = DataSet.from_tensor_slices(sample_xshards)
# List all elements in ds 
# {'movie_title': b"One Flew Over the Cuckoo's Nest (1975)", 'user_id': 1, 'user_rating': 5}
# {'movie_title': b'James and the Giant Peach (1996)', 'user_id': 1, 'user_rating': 3}
# {'movie_title': b'My Fair Lady (1964)', 'user_id': 1, 'user_rating': 3}
# {'movie_title': b'Erin Brockovich (2000)', 'user_id': 1, 'user_rating': 4}
# {'movie_title': b"Bug's Life, A (1998)", 'user_id': 1, 'user_rating': 5}
```

### From Friesian FeatureTable

An Orca TF Dataset can be created from a Friesian FeaturetTable using `Dataset.from_feature_table(tbl)`

```python
tbl = FeatureTable.read_csv("/path/to/input_file", delimiter=":", header=False, names=["userid", "title", "rating"])
tbl.show(5, False)
# +------+--------------------------------------+------+
# |userid|title                                 |rating|
# +------+--------------------------------------+------+
# |1     |One Flew Over the Cuckoo's Nest (1975)|5     |
# |1     |James and the Giant Peach (1996)      |3     |
# |1     |My Fair Lady (1964)                   |3     |
# |1     |Erin Brockovich (2000)                |4     |
# |1     |Bug's Life, A (1998)                  |5     |
# +------+--------------------------------------+------+

ds = Dataset.from_feature_table(tbl)
# List all elements in ds 
# {'userid': b'1', 'title': b"One Flew Over the Cuckoo's Nest (1975)", 'rating': 5}
# {'userid': b'1', 'title': b'James and the Giant Peach (1996)', 'rating': 3}
# {'userid': b'1', 'title': b'My Fair Lady (1964)', 'rating': 3}
# {'userid': b'1', 'title': b'Erin Brockovich (2000)', 'rating': 4}
# {'userid': b'1', 'title': b"Bug's Life, A (1998)", 'rating': 5}
```

## Preprocess Orca TF Dataset using map function

You can use the map function to do basic data pre-processing on an Orca TF Dataset.
```python
# Preprocess the ds using map function
ds = ds.map(lambda x: {
    "movie_title": x["title"],
    "user_id": x["userid"],
    "user_rating": x["rating"],
    "a": (x["userid"], x["title"])
})
# List all elements in ds
# {'movie_title': b"One Flew Over the Cuckoo's Nest (1975)", 'user_id': b'1', 'user_rating': 5, 'a': (b'1', b"One Flew Over the Cuckoo's Nest (1975)")}
# {'movie_title': b'James and the Giant Peach (1996)', 'user_id': b'1', 'user_rating': 3, 'a': (b'1', b'James and the Giant Peach (1996)')}
# {'movie_title': b'My Fair Lady (1964)', 'user_id': b'1', 'user_rating': 3, 'a': (b'1', b'My Fair Lady (1964)')}
# {'movie_title': b'Erin Brockovich (2000)', 'user_id': b'1', 'user_rating': 4, 'a': (b'1', b'Erin Brockovich (2000)')}
# {'movie_title': b"Bug's Life, A (1998)", 'user_id': b'1', 'user_rating': 5, 'a': (b'1', b"Bug's Life, A (1998)")}
```

## Model training, validation and inference using Orca TF Dataset

Note that the Orca TF Dataset will be automatically **batched** in the estimator, and if the input dataset is a training dataset, it will be **shuffled** before fitting.

```python
from bigdl.orca.learn.tf2.estimator import Estimator

est = Estimator.from_keras(model_creator=model_creator,
                           verbose=True,
                           config=config, backend="tf2")
est.fit(ds, 1, batch_size=32, steps_per_epoch=steps)
est.evaluate(ds, 32, num_steps=steps)
pred_shards = est.predict(ds)
```

## Quick Start

**Orca TF Dataset** usage follows a common pattern:
1. Create an Orca TF Dataset from input xShards or Friesian FeatureTable.
2. Apply TF dataset map functions to preprocess the data.
3. Feed into Orca TF Estimator to do model training, validation, and inference.

Let's see an example of using Orca TF Dataset. You can run this example via [Friesian basic ranking example](https://github.com/intel-analytics/BigDL/blob/main/python/friesian/colab-notebook/examples/basic_ranking.ipynb)

First, init an orca context and create a Friesian FeatureTable from the input CSV files. Then we can create the Orca TF Dataset from the Friesian FeatureTable.

```python
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca.data.tf.data import Dataset

# Init an orca context
init_orca_context("local", cores=4, memory="4g", init_ray_on_spark=True)

# Read the input csv files
tbl = FeatureTable.read_csv("/path/to/input_file", delimiter=":", header=False, names=["userid", "title", "rating"]).dropna(columns=None)
tbl = tbl.cast(["rating"], "int")
tbl = tbl.cast(["userid"], "string")
tbl.show(5, False)
# +------+--------------------------------------+------+
# |userid|title                                 |rating|
# +------+--------------------------------------+------+
# |1     |One Flew Over the Cuckoo's Nest (1975)|5     |
# |1     |James and the Giant Peach (1996)      |3     |
# |1     |My Fair Lady (1964)                   |3     |
# |1     |Erin Brockovich (2000)                |4     |
# |1     |Bug's Life, A (1998)                  |5     |
# +------+--------------------------------------+------+

# Generate unique index value of categorical features and encode these columns with generated string indices.
str_idx = tbl.gen_string_idx(["userid", "title"])
user_id_size = str_idx[0].size()
title_size = str_idx[1].size()
tbl = tbl.encode_string(["userid", "title"], str_idx)
tbl.show(5, False)
# +------+------+-----+
# |rating|userid|title|
# +------+------+-----+
# |5     |4395  |1855 |
# |3     |4395  |136  |
# |3     |4395  |2973 |
# |4     |4395  |816  |
# |5     |4395  |1582 |
# +------+------+-----+

train_count = tbl.size()
steps = math.ceil(train_count / 8192)
print("train size: ", train_count, ", steps: ", steps)

# Create an Orca TF Dataset from a Friesian FeatureTable
ds = Dataset.from_feature_table(tbl)
# List all elements in ds 
# {'userid': 4395, 'title': 1855, 'rating': 5}
# {'userid': 4395, 'title': 136, 'rating': 3}
# {'userid': 4395, 'title': 2973, 'rating': 3}
# {'userid': 4395, 'title': 816, 'rating': 4}
# {'userid': 4395, 'title': 1582, 'rating': 5}
```

Once the Orca TF Dataset is created, we can perform some data preprocessing using the map function. Since the model use `input["movie_title"], input["user_id"] and input["user_rating"]` in the model `call` and `compute_loss` function, we should change the key name of the Dataset.

```python
# Preprocess the ds using map function
ds = ds.map(lambda x: {
    "movie_title": x["title"],
    "user_id": x["userid"],
    "user_rating": x["rating"],
    "a": (x["userid"], x["title"])
})
# List all elements in ds
# {'movie_title': 1855, 'user_id': 4395, 'user_rating': 5, 'a': (4395, 1855)}
# {'movie_title': 136, 'user_id': 4395, 'user_rating': 3, 'a': (4395, 136)}
# {'movie_title': 2973, 'user_id': 4395, 'user_rating': 3, 'a': (4395, 2973)}
# {'movie_title': 816, 'user_id': 4395, 'user_rating': 4, 'a': (4395, 816)}
# {'movie_title': 1582, 'user_id': 4395, 'user_rating': 5, 'a': (4395, 1582)}
```

Then we can use this dataset as input for estimator `fit`, `evaluate` and `predict`.

```python
# Define the SampleRankingModel
class SampleRankingModel(tfrs.models.Model):
    def __init__(self, user_id_num, movie_title_num):
        super().__init__()
        embedding_dim = 32
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.user_embedding = tf.keras.layers.Embedding(user_id_num + 1, embedding_dim)
        self.movie_embedding = tf.keras.layers.Embedding(movie_title_num + 1, embedding_dim)
        self.ratings = tf.keras.Sequential([
              # Learn multiple dense layers.
              tf.keras.layers.Dense(256, activation="relu"),
              tf.keras.layers.Dense(64, activation="relu"),
              # Make rating predictions in the final layer.
              tf.keras.layers.Dense(1)
          ])

    def call(self, features):
        embeddings = tf.concat([self.user_embedding(features["user_id"]),
                               self.movie_embedding(features["movie_title"])], axis=1)
        return self.ratings(embeddings)

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        labels = inputs["user_rating"]
        rating_predictions = self(inputs)
        return self.task(labels=labels, predictions=rating_predictions)


def model_creator(config):
    model = SampleRankingModel(user_id_num, movie_title_num)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                  optimizer=tf.keras.optimizers.Adagrad(config["lr"]))
    return model

config = {
    "lr": 0.1
}

est = Estimator.from_keras(model_creator=model_creator,
                           verbose=True,
                           config=config, backend="tf2")
est.fit(ds, 1, batch_size=32, steps_per_epoch=steps)
est.evaluate(ds, 32, num_steps=steps)
pred_shards = est.predict(ds)
# Collect the predict results to driver.
pred_collect = pred_shards.collect()
stop_orca_context()
```
