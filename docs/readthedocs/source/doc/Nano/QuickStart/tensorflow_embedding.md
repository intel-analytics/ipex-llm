# BigDL-Nano TensorFlow SparseEmbedding and SparseAdam
**In this guide we demonstrates how to use `SparseEmbedding` and `SparseAdam` to obtain stroger performance with sparse gradient.**

### **Step 0: Prepare Environment**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment. Please refer to the [install guide](../../UserGuide/python.md) for more details.

```bash
conda create py37 python==3.7.10 setuptools==58.0.4
conda activate py37
# nightly bulit version
pip install --pre --upgrade bigdl-nano[tensorflow]
# set env variables for your conda environment
source bigdl-nano-init
pip install tensorflow-datasets
```

### **Step 1: Import BigDL-Nano**
The optimizations in BigDL-Nano are delivered through BigDL-Nano’s `Model` and `Sequential` classes. For most cases, you can just replace your `tf.keras.Model` to `bigdl.nano.tf.keras.Model` and `tf.keras.Sequential` to `bigdl.nano.tf.keras.Sequential` to benefits from BigDL-Nano.
```python
from bigdl.nano.tf.keras import Model, Sequential
```

### **Step 2: Load the data**
We demonstrate with imdb_reviews, a large movie review dataset.
```python
import tensorflow_datasets as tfds
(raw_train_ds, raw_val_ds, raw_test_ds), info = tfds.load(
    "imdb_reviews",
    split=['train[:80%]', 'train[80%:]', 'test'],
    as_supervised=True,
    batch_size=32,
    shuffle_files=False,
    with_info=True
)
```

### **Step 3: Parepre the Data**
In particular, we remove <br /> tags.
```python
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import string
import re

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

max_features = 20000
embedding_dim = 128
sequence_length = 500

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(text_ds)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)
```

### **Step 4: Build Model**
`bigdl.nano.tf.keras.Embedding` is a slightly modified version of `tf.keras.Embedding` layer, this embedding layer only applies regularizer to the output of the embedding layer, so that the gradient to embeddings is sparse. `bigdl.nano.tf.optimzers.Adam` is a variant of the `Adam` optimizer that handles sparse updates more efficiently. 
Here we create two models, one using normal Embedding layer and Adam optimizer, the other using `SparseEmbedding` and `SparseAdam`.
```python
from tensorflow.keras import layers
from bigdl.nano.tf.keras.layers import Embedding
from bigdl.nano.tf.optimizers import SparseAdam

from tensorflow.keras import layers
from bigdl.nano.tf.keras.layers import Embedding
from bigdl.nano.tf.optimizers import SparseAdam

inputs = tf.keras.Input(shape=(None,), dtype="int64")

# Embedding layer can only be used as the first layer in a model,
# you need to provide the argument inputShape (a Single Shape, does not include the batch dimension).
x = Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an SparseAdam optimizer.
model.compile(loss="binary_crossentropy", optimizer=SparseAdam(), metrics=["accuracy"])
```

### **Step 5: Training**
```python
# Fit the model using the train and val datasets.
model.fit(train_ds, validation_data=val_ds, epochs=3)

model.evaluate(test_ds)
```

You can find the detailed result of training from [here](https://github.com/intel-analytics/BigDL/blob/main/python/nano/notebooks/tensorflow/tutorial/tensorflow_embedding.ipynb)