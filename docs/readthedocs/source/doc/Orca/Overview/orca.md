# Orca in 5 minutes

### Overview

Most AI projects start with a Python notebook running on a single laptop; however, one usually needs to go through a mountain of pains to scale it to handle larger data set in a distributed fashion. The  _**Orca**_ library seamlessly scales out your single node Python notebook across large clusters (so as to process distributed Big Data).

---

### TensorFlow Bite-sized Example

First of all, follow the steps [here](install.md) to install Orca in your environment.

This section uses TensorFlow 1.15, and you should also install TensorFlow before running this example:
```bash
pip install tensorflow==1.15
```

First, initialize [Orca Context](orca-context.md):

```python
from bigdl.orca import init_orca_context, OrcaContext

# cluster_mode can be "local", "k8s" or "yarn"
sc = init_orca_context(cluster_mode="local", cores=4, memory="10g", num_nodes=1)
```

Next, perform [data-parallel processing in Orca](data-parallel-processing.md) (supporting standard Spark Dataframes, TensorFlow Dataset, PyTorch DataLoader, Pandas, etc.):

```python
from pyspark.sql.functions import array

spark = OrcaContext.get_spark_session()
df = spark.read.parquet(file_path)
df = df.withColumn('user', array('user')) \
       .withColumn('item', array('item'))
```

Finally, use [sklearn-style Estimator APIs in Orca](distributed-training-inference.md) to perform distributed _TensorFlow_, _PyTorch_, _Keras_ and _BigDL_ training and inference:

```python
from tensorflow import keras
from bigdl.orca.learn.tf.estimator import Estimator

user = keras.layers.Input(shape=[1])
item = keras.layers.Input(shape=[1])
feat = keras.layers.concatenate([user, item], axis=1)
predictions = keras.layers.Dense(2, activation='softmax')(feat)
model = keras.models.Model(inputs=[user, item], outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

est = Estimator.from_keras(keras_model=model)
est.fit(data=df,
        batch_size=64,
        epochs=4,
        feature_cols=['user', 'item'],
        label_cols=['label'])
```
