# Use Spark Dataframe for Deep Learning

---

![](../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/ncf_dataframe.ipynb) &nbsp;![](../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/ncf_dataframe.ipynb)

---

**In this guide we will describe how to use Apache Spark Dataframes to scale-out data processing for distribtued deep learning.**

The dataset used in this guide is [movielens-1M](https://grouplens.org/datasets/movielens/1m/), which contains 1 million ratings of 5 levels from 6000 users on 4000 movies. We will read the data into Spark Dataframe and directly use the Spark Dataframe as the input to the distributed training.

### **1. Read input data into Spark DataFrame**

First, read the input data into Spark Dataframes.

```python
from zoo.orca import OrcaContext

spark = OrcaContext.get_spark_session()
# read csv with specifying column names
df = spark.read.csv(new_rating_files, sep=':', inferSchema=True).toDF(
  "user", "item", "label", "timestamp")
```

### **2. Process data using Spark Dataframe**

Next, process the data using Spark Dataframe operations.

```python
# update label starting from 0. That's because ratings go from 1 to 5, while the matrix column index goes from 0 to 4
df = df.withColumn('label', df.label-1)

# split to train/test dataset
train_data, test_data = df.randomSplit([0.8, 0.2], 100)
```

### **3. Define NCF model**

This example defines NCF model in the _Creator Function_ using TensroFlow 2 APIs as follows.

```python
def model_creator(config):
    from tensorflow import keras
    import tensorflow as tf
    
    embedding_size=16
    user = keras.layers.Input(dtype=tf.int32, shape=(None,))
    item = keras.layers.Input(dtype=tf.int32, shape=(None,))
    label = keras.layers.Input(dtype=tf.int32, shape=(None,))

    with tf.name_scope("GMF"):
        user_embed_GMF = keras.layers.Embedding(max_user_id + 1, embedding_size)(user)
        item_embed_GMF = keras.layers.Embedding(max_item_id + 1, embedding_size)(item)
        GMF = keras.layers.Multiply()([user_embed_GMF, item_embed_GMF])

    with tf.name_scope("MLP"):
        user_embed_MLP = keras.layers.Embedding(max_user_id + 1, embedding_size)(user)
        item_embed_MLP = keras.layers.Embedding(max_item_id + 1, embedding_size)(item)
        interaction = concat([user_embed_MLP, item_embed_MLP], axis=-1)
        layer1_MLP = keras.layers.Dense(units=embedding_size * 2, activation='relu')(interaction)
        layer1_MLP = keras.layers.Dropout(rate=0.2)(layer1_MLP)
        layer2_MLP = keras.layers.Dense(units=embedding_size, activation='relu')(layer1_MLP)
        layer2_MLP = keras.layers.Dropout(rate=0.2)(layer2_MLP)
        layer3_MLP = keras.layers.Dense(units=embedding_size // 2, activation='relu')(layer2_MLP)
        layer3_MLP = keras.layers.Dropout(rate=0.2)(layer3_MLP)

    # Concate the two parts together
    with tf.name_scope("concatenation"):
        concatenation = tf.concat([GMF, layer3_MLP], axis=-1)
        outputs = keras.layers.Dense(units=5, activation='softmax')(concatenation)

    model = keras.Model(inputs=[user, item], outputs=outputs)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    return model
```

### **4. Fit with Orca Estimator**

Finally, run distributed model training/inference on the Spark Dataframes directly.

```python
from zoo.orca.learn.tf2 import Estimator

# create an Estimator
est = Estimator.from_keras(model_creator=model_creator) # the model accept two inputs and one label

# fit with Estimator
stats = est.fit(train_data,
                epochs=epochs,
                batch_size=batch_size,
                feature_cols=['user', 'item'], # specifies which column(s) to be used as inputs
                label_cols=['label'], # specifies which column(s) to be used as labels
                steps_per_epoch=800000 // batch_size,
                validation_data=test_data,
                validation_steps=200000 // batch_size)

checkpoint_path = os.path.join(model_dir, "NCF.ckpt")
est.save(checkpoint_path)

# evaluate with Estimator
stats = est.evaluate(test_data,
                     feature_cols=['user', 'item'], # specifies which column(s) to be used as inputs
                     label_cols=['label'], # specifies which column(s) to be used as labels
                     num_steps=100000 // batch_size)
est.shutdown()
print(stats)
```

