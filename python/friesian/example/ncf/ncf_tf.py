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
#

import numpy as np
import tensorflow as tf
from pandas import read_csv
from sklearn.model_selection import train_test_split


def build_model(num_users, num_items, layers=[20, 10], include_mf=True, mf_embed=20):
    num_layer = len(layers)
    user_input = tf.keras.layers.Input(shape=(1,), dtype="int32", name="user_input")
    item_input = tf.keras.layers.Input(shape=(1,), dtype="int32", name="item_input")

    mlp_embed_user = tf.keras.layers.Embedding(input_dim=num_users + 1,
                                               output_dim=int(layers[0] / 2),
                                               input_length=1)(user_input)
    mlp_embed_item = tf.keras.layers.Embedding(input_dim=num_items + 1,
                                               output_dim=int(layers[0] / 2),
                                               input_length=1)(item_input)
    user_latent = tf.keras.layers.Flatten()(mlp_embed_user)
    item_latent = tf.keras.layers.Flatten()(mlp_embed_item)

    mlp_latent = tf.keras.layers.concatenate([user_latent, item_latent], axis=1)
    for idx in range(1, num_layer):
        layer = tf.keras.layers.Dense(layers[idx], activation="relu",
                      name="layer%d" % idx)
        mlp_latent = layer(mlp_latent)

    if include_mf:
        mf_embed_user = tf.keras.layers.Embedding(input_dim=num_users + 1,
                                                  output_dim=mf_embed,
                                                  input_length=1)(user_input)
        mf_embed_item = tf.keras.layers.Embedding(input_dim=num_items + 1,
                                                  output_dim=mf_embed,
                                                  input_length=1)(item_input)
        mf_user_flatten = tf.keras.layers.Flatten()(mf_embed_user)
        mf_item_flatten = tf.keras.layers.Flatten()(mf_embed_item)

        mf_latent = tf.keras.layers.multiply([mf_user_flatten, mf_item_flatten])
        concated_model = tf.keras.layers.concatenate([mlp_latent, mf_latent], axis=1)
        prediction = tf.keras.layers.Dense(1, activation="relu", name="prediction")(concated_model)
    else:
        prediction = tf.keras.layers.Dense(1, activation="relu", name="prediction")(mlp_latent)

    model = tf.keras.Model([user_input, item_input], prediction)
    return model


full_data = read_csv("/home/kai/Downloads/ml-1m/ratings.dat", sep="::", header=None, names=["user", "item", "label"],
                     usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
print(full_data.head())
user_set = set(full_data["user"].unique())
item_set = set(full_data["item"].unique())
min_user_id = min(user_set)
max_user_id = max(user_set)
min_item_id = min(item_set)
max_item_id = max(item_set)
print(min_user_id, max_user_id, min_item_id, max_item_id)
train, test = train_test_split(full_data, test_size=0.2, random_state=100)

model = build_model(max_user_id, max_item_id)
model.summary()
optimizer = tf.keras.optimizers.Adam(1e-2)
model.compile(optimizer=optimizer, loss="mean_squared_error")
model.fit([train.user, train.item], train.label,
          batch_size=800,
          epochs=5,
          validation_data=([test.user, test.item], test.label))
tf.saved_model.save(model, "./model")
