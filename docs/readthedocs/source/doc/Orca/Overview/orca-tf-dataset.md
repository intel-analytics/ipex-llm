# Orca TF Dataset

Orca TF Dataset represents a distributed [TF Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). It can be used as a bridge between [Orca xShards](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/data-parallel-processing.html#xshards-distributed-data-parallel-python-processing) (or [Friesian FeatureTable](https://bigdl.readthedocs.io/en/latest/doc/PythonAPI/Friesian/feature.html)) and [Orca TF Estimator](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-training-inference.html#tensorflow-keras-estimator) input, it also allows users to do basic data pre-processing. 


## Key Concepts

An **Orca TF Dataset** is a distributed Tensorflow tf.data.Dataset.
- Each element of the Orca TF Dataset is a tf.data.Dataset.
- An Orca TF Dataset has a collection of elements partitioned across the cluster nodes that can be operated on in parallel.
- After the Orca TF Dataset is created, map functions can be applied to the Dataset in parallel. And the Orca TF estimator can do model training, validation, and inference using the created Orca TF Dataset.

## Creating an Orca TF Dataset

An Orca TF Dataset can be created from an Orca xShards, a Spark DataFrame or a Friesian FeatureTable.

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

### From Spark DataFrame

An Orca TF Dataset can be created from a Spark DataFrame using `DataSet.from_spark_df(df)`

```python
spark = OrcaContext.get_spark_session()
df = spark.read.options(header=True, inferSchema=True, delimiter=":").csv("/path/to/input_file")
df.show(5, False)
# +------+--------------------------------------+------+
# |userid|title                                 |rating|
# +------+--------------------------------------+------+
# |1     |One Flew Over the Cuckoo's Nest (1975)|5     |
# |1     |James and the Giant Peach (1996)      |3     |
# |1     |My Fair Lady (1964)                   |3     |
# |1     |Erin Brockovich (2000)                |4     |
# |1     |Bug's Life, A (1998)                  |5     |
# +------+--------------------------------------+------+

ds = DataSet.from_spark_df(df)
# List all elements in ds 
# {'userid': b'1', 'title': b"One Flew Over the Cuckoo's Nest (1975)", 'rating': 5}
# {'userid': b'1', 'title': b'James and the Giant Peach (1996)', 'rating': 3}
# {'userid': b'1', 'title': b'My Fair Lady (1964)', 'rating': 3}
# {'userid': b'1', 'title': b'Erin Brockovich (2000)', 'rating': 4}
# {'userid': b'1', 'title': b"Bug's Life, A (1998)", 'rating': 5}
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
                           config=config, backend="ray")
est.fit(ds, 1, batch_size=32, steps_per_epoch=steps)
est.evaluate(ds, 32, num_steps=steps)
pred_shards = est.predict(ds)
```

## Quick Start

**Orca TF Dataset** usage follows a common pattern:
1. Create an Orca TF Dataset from input xShards or Friesian FeatureTable.
2. Apply TF dataset map functions to preprocess the data.
3. Feed into Orca TF Estimator to do model training, validation, and inference.

Let's see an example of using Orca TF Dataset. You can run this example via [Orca basic ranking example](https://github.com/intel-analytics/BigDL/tree/main/python/orca/example/learn/tf2/basic_ranking)

First, init an orca context and create a Spark dataframe from the input CSV files. Then we can convert the dataframe to an Orca XShards, and create the Orca TF Dataset from the Orca XShards.

```python
import math
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca.data.tf.data import Dataset
from pyspark.sql.functions import col, mean, stddev

# Init an orca context
init_orca_context("local", cores=4, memory="4g", init_ray_on_spark=True)
spark = OrcaContext.get_spark_session()

# Read the input csv files
df = spark.read.options(header=True, inferSchema=True, delimiter=":").csv("/path/to/input_file")
df = df.withColumn("rating", col("rating").cast("float"))
df = df.withColumn("userid", col("userid").cast("string"))
df.show(5, False)
# +-------+------+------+---------+--------------------------------------+----------------------------+
# |movieid|userid|rating|timestamp|title                                 |genres                      |
# +-------+------+------+---------+--------------------------------------+----------------------------+
# |1193   |1     |5.0   |978300760|One Flew Over the Cuckoo's Nest (1975)|Drama                       |
# |661    |1     |3.0   |978302109|James and the Giant Peach (1996)      |Animation|Children's|Musical|
# |914    |1     |3.0   |978301968|My Fair Lady (1964)                   |Musical|Romance             |
# |3408   |1     |4.0   |978300275|Erin Brockovich (2000)                |Drama                       |
# |2355   |1     |5.0   |978824291|Bug's Life, A (1998)                  |Animation|Children's|Comedy |
# +-------+------+------+---------+--------------------------------------+----------------------------+

# Generate vocabularies for the StringLookup layers
user_id_vocab = df.select("userid").distinct().rdd.map(lambda row: row["userid"]).collect()
movie_title_vocab = df.select("title").distinct().rdd.map(lambda row: row["title"]).collect()

# Calculate mean and standard deviation for normalization
df_stats = df.select(
    mean(col('timestamp')).alias('mean'),
    stddev(col('timestamp')).alias('std')
).collect()
mean = df_stats[0]['mean']
stddev = df_stats[0]['std']

train_count = df.count()
steps = math.ceil(train_count / 8192)
print("train size: ", train_count, ", steps: ", steps)

# Create an Orca TF Dataset from a Spark DataFrame
ds = DataSet.from_spark_df(df)
# List all elements in ds 
# {'movieid': 1193, 'userid': b'1', 'rating': 5.0, 'timestamp': 978300760, 'title': b"One Flew Over the Cuckoo's Nest (1975)", 'genres': b'Drama'}
# {'movieid': 661, 'userid': b'1', 'rating': 3.0, 'timestamp': 978302109, 'title': b'James and the Giant Peach (1996)', 'genres': b"Animation|Children's|Musical"}
# {'movieid': 914, 'userid': b'1', 'rating': 3.0, 'timestamp': 978301968, 'title': b'My Fair Lady (1964)', 'genres': b'Musical|Romance'}
# {'movieid': 3408, 'userid': b'1', 'rating': 4.0, 'timestamp': 978300275, 'title': b'Erin Brockovich (2000)', 'genres': b'Drama'}
# {'movieid': 2355, 'userid': b'1', 'rating': 5.0, 'timestamp': 978824291, 'title': b"Bug's Life, A (1998)", 'genres': b"Animation|Children's|Comedy"}
```

Once the Orca TF Dataset is created, we can perform some data preprocessing using the map function. Since the model use `input["movie_title"]`, `input["user_id"]` and `input["user_rating"]` in the model `call`, `train_step` and `test_step` function, we should change the key name of the Dataset. Also, we normalize the continuous feature timestamp here.
```python
def preprocess(x):
    return {
        "movie_title": x["title"],
        "user_id": x["userid"],
        "user_rating": x["rating"],
        # Normalize continuous timestamp
        "timestamp": (tf.cast(x["timestamp"], tf.float32) - mean) / stddev
    }

# Preprocess the ds using map function
ds = ds.map(preprocess)
# List all elements in ds
# {'movie_title': b"One Flew Over the Cuckoo's Nest (1975)", 'user_id': b'1', 'user_rating': 5.0, 'timestamp': 0.49397522}
# {'movie_title': b'James and the Giant Peach (1996)', 'user_id': b'1', 'user_rating': 3.0, 'timestamp': 0.4940853}
# {'movie_title': b'My Fair Lady (1964)', 'user_id': b'1', 'user_rating': 3.0, 'timestamp': 0.49407482}
# {'movie_title': b'Erin Brockovich (2000)', 'user_id': b'1', 'user_rating': 4.0, 'timestamp': 0.4939385}
# {'movie_title': b"Bug's Life, A (1998)", 'user_id': b'1', 'user_rating': 5.0, 'timestamp': 0.5368723}
```

Then we can use this dataset as input for estimator `fit`, `evaluate` and `predict`.

```python
# Define the SampleRankingModel
class SampleRankingModel(tf.keras.Model):
    def __init__(self, unique_user_ids, unique_movie_titles):
        super().__init__()
        embedding_dim = 32

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim)])
        self.movie_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dim)])
        self.ratings = tf.keras.Sequential([
              # Learn multiple dense layers.
              tf.keras.layers.Dense(256, activation="relu"),
              tf.keras.layers.Dense(64, activation="relu"),
              # Make rating predictions in the final layer.
              tf.keras.layers.Dense(1)
          ])

    def call(self, features):
        embeddings = tf.concat([self.user_embedding(features["user_id"]),
                                self.movie_embedding(features["movie_title"]),
                                tf.reshape(features["timestamp"], (-1, 1))], axis=1)
        return self.ratings(embeddings)

    def train_step(self, data):
        y = data["user_rating"]

        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        y = data["user_rating"]

        y_pred = self(data, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def model_creator(config):
    model = SampleRankingModel(unique_user_ids=user_id_vocab,
                               unique_movie_titles=movie_title_vocab)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()],
                  optimizer=tf.keras.optimizers.Adagrad(config["lr"]))
    return model

config = {
    "lr": 0.1
}

est = Estimator.from_keras(model_creator=model_creator,
                           verbose=True,
                           config=config, backend="ray")
# Train the model using Orca TF Dataset.
est.fit(ds, 1, batch_size=32, steps_per_epoch=steps)
# Evaluate the model on the test set.
est.evaluate(ds, 32, num_steps=steps)
pred_shards = est.predict(ds)
# Collect the predict results to driver.
pred_collect = pred_shards.collect()
stop_orca_context()
```
