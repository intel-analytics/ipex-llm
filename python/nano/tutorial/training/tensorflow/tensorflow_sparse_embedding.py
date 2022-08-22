#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This example shows how to use sparse adam optimizer and embedding layer with bigdl-nano


import os
import re
import string

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

# import `Embedding`, `SparseAdam` and `Model` from bigdl-nano
from bigdl.nano.tf.keras.layers import Embedding
from bigdl.nano.tf.optimizers import SparseAdam
from bigdl.nano.tf.keras import Model


max_features = 20000
embedding_dim = 128


def create_datasets():
    (raw_train_ds, raw_val_ds, raw_test_ds), info = tfds.load(
        "imdb_reviews",
        data_dir="/tmp/data",
        split=['train[:80%]', 'train[80%:]', 'test'],
        as_supervised=True,
        batch_size=32,
        with_info=True
    )

    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
        return tf.strings.regex_replace(
            stripped_html, f"[{re.escape(string.punctuation)}]", ""
        )

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=500,
    )
    
    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    # Vectorize the data
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    return train_ds, val_ds, test_ds


def make_backbone():
    inputs = tf.keras.Input(shape=(None, embedding_dim))
    x = layers.Dropout(0.5)(inputs)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = Model(inputs, predictions)
    return model


def make_model():
    inputs = tf.keras.Input(shape=(None,), dtype="int64")
    
    # use `Embedding` layer in `bigdl.nano.tf.keras.layers`
    x = Embedding(max_features, embedding_dim)(inputs)

    predictions = make_backbone()(x)
    model = Model(inputs, predictions)

    # use `SparseAdam` optimizer in `bigdl.nano.tf.optimizers`
    model.compile(loss="binary_crossentropy", optimizer=SparseAdam(), metrics=["accuracy"])

    return model


if __name__=='__main__':
    num_epochs = int(os.environ.get('NUM_EPOCHS', 10))

    train_ds, val_ds, test_ds = create_datasets()

    # Use sparse adam optimizer and embedding layer
    #
    # Sparse embedding represents the gradient matrix by a sparse tensor and 
    # only calculating gradients for embedding vectors which will be non zero.
    # It can be used to speed up and reduce memory usage
    # 
    # Use `Embedding` in `bigdl.nano.tf.keras.layers` to create a sparse embedding layer,
    # then use `SparseAdam` in `bigdl.nano.tf.optimizers` as the model's optimizer.
    #
    model = make_model()

    model.fit(train_ds, validation_data=val_ds, epochs=num_epochs)

    his = model.evaluate(test_ds)
