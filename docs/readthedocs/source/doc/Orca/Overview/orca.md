# Orca in 5 minutes

### Overview

The  _**Orca**_ library in BigDL can seamlessly scale out your single node Python notebook across large clusters to process large-scale data.

This page demonstrates how to scale the distributed training and inference of a standard TensorFlow model to a large cluster with minimum code changes to your notebook using Orca. We use [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) for recommendation as an example.

---

### TensorFlow Bite-sized Example

Before running this example, follow the steps [here](install.md) to prepare the environment and install Orca in your environment.

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

Next, perform [data-parallel processing in Orca](data-parallel-processing.md) (supporting standard Spark DataFrames, TensorFlow Dataset, PyTorch DataLoader, Pandas, etc.). Here to make things simple, we just generate some random data with Spark DataFrame:

```python
import random
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
train_df, test_df = df.randomSplit([0.8, 0.2], seed=1)
```

Finally, use [sklearn-style Estimator APIs in Orca](distributed-training-inference.md) to perform distributed _TensorFlow_, _PyTorch_, _Keras_ and _BigDL_ training and inference:

```python
from bigdl.orca.learn.tf2.estimator import Estimator

# Define the NCF model in standard TensorFlow API
def model_creator(config):
    from tensorflow import keras

    user_input = keras.layers.Input(shape=(1,), dtype="int32", name="use_input")
    item_input = keras.layers.Input(shape=(1,), dtype="int32", name="item_input")

    mlp_embed_user = keras.layers.Embedding(input_dim=config["num_users"], output_dim=config["embed_dim"],
                                            input_length=1)(user_input)
    mlp_embed_item = keras.layers.Embedding(input_dim=config["num_items"], output_dim=config["embed_dim"],
                                            input_length=1)(item_input)

    user_latent = keras.layers.Flatten()(mlp_embed_user)
    item_latent = keras.layers.Flatten()(mlp_embed_item)

    mlp_latent = keras.layers.concatenate([user_latent, item_latent], axis=1)
    predictions = keras.layers.Dense(1, activation="sigmoid")(mlp_latent)
    model = keras.models.Model(inputs=[user_input, item_input], outputs=predictions)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


batch_size = 64
train_steps = int(train_df.count() / batch_size)
val_steps = int(test_df.count() / batch_size)

est = Estimator.from_keras(model_creator=model_creator, backend="spark",
                           config={"embed_dim": 8, "num_users": num_users, "num_items": num_items})

# Distributed training
est.fit(data=train_df,
        batch_size=batch_size,
        epochs=4,
        feature_cols=['user', 'item'],
        label_cols=['label'],
        steps_per_epoch=train_steps,
        validation_data=test_df,
        validation_steps=val_steps)

# Distributed inference
prediction_df = est.predict(test_df,
                            batch_size=batch_size,
                            feature_cols=['user', 'item'])
```

Stop [Orca Context](orca-context.md) after you finish your program:

```python
stop_orca_context()
```
