# Orca in 5 minutes

### Overview

Most AI projects start with a Python notebook running on a single laptop; however, one usually needs to go through a mountain of pains to scale it to handle larger data set in a distributed fashion. The  _**Orca**_ library seamlessly scales out your single node Python notebook across large clusters (so as to process distributed Big Data).

---

### TensorFlow Bite-sized Example

First of all, follow the steps [here](install.md#to-use-basic-orca-features) to install Orca in your environment.

This section uses **TensorFlow 2.x**, and you should also install TensorFlow before running this example:
```bash
pip install tensorflow
```

First, initialize [Orca Context](orca-context.md):

```python
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext

# cluster_mode can be "local", "k8s" or "yarn"
sc = init_orca_context(cluster_mode="local", cores=4, memory="10g", num_nodes=1)
```

Next, perform [data-parallel processing in Orca](data-parallel-processing.md) (supporting standard Spark Dataframes, TensorFlow Dataset, PyTorch DataLoader, Pandas, etc.). Here to make things simple, we just generate some random data with Spark DataFrame:

```python
import random
from pyspark.sql.functions import array
from pyspark.sql.types import StructType, StructField, IntegerType
from bigdl.orca import OrcaContext

spark = OrcaContext.get_spark_session()

num_users, num_items = 200, 100
rdd = sc.range(0, 512).map(
    lambda x: [random.randint(0, num_users-1), random.randint(0, num_items-1), random.randint(0, 1)])
schema = StructType([StructField("user", IntegerType(), False),
                     StructField("item", IntegerType(), False),
                     StructField("label", IntegerType(), False)])
df = spark.createDataFrame(rdd, schema)
train, test = df.randomSplit([0.8, 0.2], seed=1)
```

Finally, use [sklearn-style Estimator APIs in Orca](distributed-training-inference.md) to perform distributed _TensorFlow_, _PyTorch_, _Keras_ and _BigDL_ training and inference:

```python
from tensorflow import keras
from bigdl.orca.learn.tf2.estimator import Estimator

def model_creator(config):
  user_input = keras.layers.Input(shape=(1,), dtype="int32", name="use_input")
  item_input = keras.layers.Input(shape=(1,), dtype="int32", name="item_input")

  mlp_embed_user = keras.layers.Embedding(input_dim=num_users, output_dim=config["embed_dim"],
                               input_length=1)(user_input)
  mlp_embed_item = keras.layers.Embedding(input_dim=num_items, output_dim=config["embed_dim"],
                               input_length=1)(item_input)

  user_latent = keras.layers.Flatten()(mlp_embed_user)
  item_latent = keras.layers.Flatten()(mlp_embed_item)

  mlp_latent = keras.layers.concatenate([user_latent, item_latent], axis=1)
  predictions = keras.layers.Dense(2, activation="sigmoid")(mlp_latent)
  model = keras.models.Model(inputs=[user_input, item_input], outputs=predictions)
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

est = Estimator.from_keras(model_creator=model_creator, backend="spark", config={"embed_dim": 8})
est.fit(data=train,
        batch_size=64,
        epochs=4,
        feature_cols=['user', 'item'],
        label_cols=['label'],
        steps_per_epoch=int(train.count()/64),
        validation_data=test,
        validation_steps=int(test.count()/64))

stop_orca_context()
```
