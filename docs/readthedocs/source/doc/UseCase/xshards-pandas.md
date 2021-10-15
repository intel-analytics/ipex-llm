# Use Distributed Pandas for Deep Learning

---

![](../../../image/colab_logo_32px.png)[Run in Google Colab](https://colab.research.google.com/github/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/ncf_xshards_pandas.ipynb) &nbsp;![](../../../image/GitHub-Mark-32px.png)[View source on GitHub](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/colab-notebook/orca/quickstart/ncf_xshards_pandas.ipynb)

---

**In this guide we will describe how to use [XShards](../Orca/Overview/data-parallel-processing.md) to scale-out Pandas data processing for distribtued deep learning.** 

### **1. Read input data into XShards of Pandas DataFrame**

First, read CVS, JSON or Parquet files into an `XShards` of Pandas Dataframe (i.e., a distributed and sharded dataset where each partition contained a Pandas Dataframe), as shown below:

```python
from zoo.orca.data.pandas import read_csv
full_data = read_csv(new_rating_files, sep=':', header=None,
                     names=['user', 'item', 'label'], usecols=[0, 1, 2],
                     dtype={0: np.int32, 1: np.int32, 2: np.int32})
```

### **2. Process Pandas Dataframes using XShards**

Next, use XShards to efficiently process large-size Pandas Dataframes in a distributed and data-parallel fashion. You may run standard Python code on each partition in a data-parallel fashion using `XShards.transform_shard`, as shown below:

```python
# update label starting from 0. That's because ratings go from 1 to 5, while the matrix columns go from 0 to 4
def update_label(df):
  df['label'] = df['label'] - 1
  return df

full_data = full_data.transform_shard(update_label)
```

```python
from sklearn.model_selection import train_test_split

# split to train/test dataset
def split_train_test(data):
  train, test = train_test_split(data, test_size=0.2, random_state=100)
  return train, test

train_data, test_data = full_data.transform_shard(split_train_test).split()
```

### **3. Define NCF model**

Define the NCF model using TensorFlow 1.15 APIs:

```python
import tensorflow as tf

class NCF(object):
    def __init__(self, embed_size, user_size, item_size):
        self.user = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.item = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.label = tf.placeholder(dtype=tf.int32, shape=(None,))
        
        with tf.name_scope("GMF"):
            user_embed_GMF = tf.contrib.layers.embed_sequence(self.user, vocab_size=user_size + 1,
                                                              embed_dim=embed_size)
            item_embed_GMF = tf.contrib.layers.embed_sequence(self.item, vocab_size=item_size + 1,
                                                              embed_dim=embed_size)
            GMF = tf.multiply(user_embed_GMF, item_embed_GMF)

        with tf.name_scope("MLP"):
            user_embed_MLP = tf.contrib.layers.embed_sequence(self.user, vocab_size=user_size + 1,
                                                              embed_dim=embed_size)
            item_embed_MLP = tf.contrib.layers.embed_sequence(self.item, vocab_size=item_size + 1,
                                                              embed_dim=embed_size)
            interaction = tf.concat([user_embed_MLP, item_embed_MLP], axis=-1)
            layer1_MLP = tf.layers.dense(inputs=interaction, units=embed_size * 2)
            layer1_MLP = tf.layers.dropout(layer1_MLP, rate=0.2)
            layer2_MLP = tf.layers.dense(inputs=layer1_MLP, units=embed_size)
            layer2_MLP = tf.layers.dropout(layer2_MLP, rate=0.2)
            layer3_MLP = tf.layers.dense(inputs=layer2_MLP, units=embed_size // 2)
            layer3_MLP = tf.layers.dropout(layer3_MLP, rate=0.2)

        # Concate the two parts together
        with tf.name_scope("concatenation"):
            concatenation = tf.concat([GMF, layer3_MLP], axis=-1)
            self.logits = tf.layers.dense(inputs=concatenation, units=5)
            self.logits_softmax = tf.nn.softmax(self.logits)
            self.class_number = tf.argmax(self.logits_softmax, 1)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label, logits=self.logits, name='loss'))

        with tf.name_scope("optimzation"):
            self.optim = tf.train.AdamOptimizer(1e-3, name='Adam')
            self.optimizer = self.optim.minimize(self.loss)

embedding_size=16
model = NCF(embedding_size, max_user_id, max_item_id)
```
### **4. Fit with Orca Estimator**

Finally, directly run distributed model training/inference on the XShards of Pandas DataFrames.

```python
from zoo.orca.learn.tf.estimator import Estimator

# create an Estimator.
estimator = Estimator.from_graph(
            inputs=[model.user, model.item], # the model accept two inputs and one label
            outputs=[model.class_number],
            labels=[model.label],
            loss=model.loss,
            optimizer=model.optim,
            model_dir=model_dir,
            metrics={"loss": model.loss})

# fit the Estimator
estimator.fit(data=train_data,
              batch_size=1280,
              epochs=1,
              feature_cols=['user', 'item'], # specifies which column(s) to be used as inputs
              label_cols=['label'], # specifies which column(s) to be used as labels
              validation_data=test_data)
```
