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
import tensorflow as tf


def ncf_model(embedding_size, user_num, item_num, dropout, lr):
    user = tf.keras.layers.Input(dtype=tf.int32, shape=())
    item = tf.keras.layers.Input(dtype=tf.int32, shape=())

    with tf.name_scope("GMF"):
        user_embed_GMF = tf.keras.layers.Embedding(user_num, embedding_size, name='gmf_user')(user)
        item_embed_GMF = tf.keras.layers.Embedding(item_num, embedding_size, name='gmf_item')(item)
        GMF = tf.keras.layers.Multiply()([user_embed_GMF, item_embed_GMF])

    with tf.name_scope("MLP"):
        user_embed_MLP = tf.keras.layers.Embedding(
            user_num, embedding_size * 4, name='mlp_user')(user)
        item_embed_MLP = tf.keras.layers.Embedding(
            item_num, embedding_size * 4, name='mlp_item')(item)
        interaction = tf.concat([user_embed_MLP, item_embed_MLP], axis=-1)

        layer1_MLP = tf.keras.layers.Dense(
            units=embedding_size * 4, activation='relu')(interaction)
        layer1_MLP = tf.keras.layers.Dropout(rate=dropout)(layer1_MLP)
        layer2_MLP = tf.keras.layers.Dense(
            units=embedding_size * 2, activation='relu')(layer1_MLP)
        layer2_MLP = tf.keras.layers.Dropout(rate=dropout)(layer2_MLP)
        layer3_MLP = tf.keras.layers.Dense(
            units=embedding_size, activation='relu')(layer2_MLP)
        layer3_MLP = tf.keras.layers.Dropout(rate=dropout)(layer3_MLP)

    with tf.name_scope("concatenation"):
        concatenation = tf.concat([GMF, layer3_MLP], axis=-1)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(concatenation)

    model = tf.keras.Model(inputs=[user, item], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
    return model
