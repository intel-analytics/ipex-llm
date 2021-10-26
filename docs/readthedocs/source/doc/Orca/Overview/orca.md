# The Orca Library

## 1. Overview

Most AI projects start with a Python notebook running on a single laptop; however, one usually needs to go through a mountain of pains to scale it to handle larger data set in a distributed fashion. The  _**Orca**_ library seamlessly scales out your single node Python notebook across large clusters (so as to process distributed Big Data).

## 2. Install
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment.
```bash
conda create -n py37 python=3.7  # "py37" is conda environment name, you can use any name you like.
conda activate py37
pip install bigdl-orca
pip install tensorflow==1.15
```

When installing bigdl-orca with pip, you can specify the extras key `[ray]` to additionally install the additional dependencies
essential for running Ray (i.e. `ray==1.2.0`, `psutil`, `aiohttp==3.7.0`, `aioredis==1.1.0`, `setproctitle`, `hiredis==1.1.0`, `async-timeout==3.0.1`):
```bash
pip install bigdl-orca[ray]
```

## 3. Run

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

## Get Started

See [TensorFlow](../QuickStart/orca-tf-quickstart.md) and [PyTorch](../QuickStart/orca-pytorch-quickstart.md) quickstart for more details.

