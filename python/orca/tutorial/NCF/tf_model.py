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


def ncf_model(user_num, item_num, factor_num, dropout, lr, num_layers,
              sparse_feats_input_dims, sparse_feats_embed_dims, num_dense_feats):
    user = tf.keras.layers.Input(dtype=tf.int32, shape=())
    item = tf.keras.layers.Input(dtype=tf.int32, shape=())

    if not isinstance(sparse_feats_embed_dims, list):
        sparse_feats_embed_dims = [sparse_feats_embed_dims] * len(sparse_feats_input_dims)

    with tf.name_scope("GMF"):
        user_embed_GMF = tf.keras.layers.Embedding(user_num, factor_num, name='gmf_user')(user)
        item_embed_GMF = tf.keras.layers.Embedding(item_num, factor_num, name='gmf_item')(item)
        GMF = tf.keras.layers.Multiply()([user_embed_GMF, item_embed_GMF])

    with tf.name_scope("MLP"):
        user_embed_MLP = tf.keras.layers.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1)), name='mlp_user')(user)
        item_embed_MLP = tf.keras.layers.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1)), name='mlp_item')(item)

        cat_feature_input_layers = []
        cat_feature_layers = []
        for in_dim, out_dim in zip(sparse_feats_input_dims, sparse_feats_embed_dims):
            input_layer = tf.keras.layers.Input(shape=(), dtype=tf.int32)
            cat_feature_input_layers.append(input_layer)
            cat_feature_layers.append(tf.keras.layers.Embedding(in_dim, out_dim)(input_layer))

        num_feature_input_layers = []
        num_feature_layers = []
        for i in range(num_dense_feats):
            num_feature_input_layers.append(tf.keras.layers.Input(shape=1))
            num_feature_layers.append(num_feature_input_layers[i])

        all_feature_input_layers = cat_feature_input_layers + num_feature_input_layers
        all_feature_layers = cat_feature_layers + num_feature_layers

        interaction = tf.concat([user_embed_MLP, item_embed_MLP] + all_feature_layers, axis=-1)
        output_size = factor_num * (2 ** (num_layers - 1))
        for i in range(num_layers):
            layer_MLP = tf.keras.layers.Dense(units=output_size, activation='relu')(interaction)
            interaction = tf.keras.layers.Dropout(rate=dropout)(layer_MLP)
            output_size //= 2

    with tf.name_scope("concatenation"):
        concatenation = tf.concat([GMF, interaction], axis=-1)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(concatenation)

    model = tf.keras.Model(inputs=[user, item] + all_feature_input_layers, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
    return model
