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


def ncf_model(factor_num, user_num, item_num, dropout, lr,
              categorical_features_in_dim, categorical_features_out_dim, num_feature_dim):
    user = tf.keras.layers.Input(dtype=tf.int32, shape=())
    item = tf.keras.layers.Input(dtype=tf.int32, shape=())

    if not isinstance(categorical_features_out_dim, list):
        categorical_features_out_dim = [categorical_features_out_dim for _ in categorical_features_in_dim]

    cat_feature_input_layers = []
    cat_feature_layers = []
    for i, (in_dim, out_dim) in enumerate(zip(categorical_features_in_dim, categorical_features_out_dim)):
        cat_feature_input_layers.append(tf.keras.layers.Input(shape=(), dtype=tf.int32))
        cat_feature_layers.append(
            tf.keras.layers.Embedding(in_dim + 1, out_dim)(cat_feature_input_layers[i]))

    num_feature_input_layers = []
    num_feature_layers = []
    for i, in_dim in enumerate(num_feature_dim):
        num_feature_input_layers.append(tf.keras.layers.Input(shape=in_dim))
        num_feature_layers.append(num_feature_input_layers[i])

    add_feature_input_layers = cat_feature_input_layers + num_feature_input_layers
    add_feature_layers = cat_feature_layers + num_feature_layers

    with tf.name_scope("GMF"):
        user_embed_GMF = tf.keras.layers.Embedding(user_num, factor_num, name='gmf_user')(user)
        item_embed_GMF = tf.keras.layers.Embedding(item_num, factor_num, name='gmf_item')(item)
        GMF = tf.keras.layers.Multiply()([user_embed_GMF, item_embed_GMF])

    with tf.name_scope("MLP"):
        user_embed_MLP = tf.keras.layers.Embedding(
            user_num, factor_num * 4, name='mlp_user')(user)
        item_embed_MLP = tf.keras.layers.Embedding(
            item_num, factor_num * 4, name='mlp_item')(item)
        interaction = tf.concat([user_embed_MLP, item_embed_MLP] + add_feature_layers, axis=-1)

        layer1_MLP = tf.keras.layers.Dense(
            units=factor_num * 4, activation='relu')(interaction)
        layer1_MLP = tf.keras.layers.Dropout(rate=dropout)(layer1_MLP)
        layer2_MLP = tf.keras.layers.Dense(
            units=factor_num * 2, activation='relu')(layer1_MLP)
        layer2_MLP = tf.keras.layers.Dropout(rate=dropout)(layer2_MLP)
        layer3_MLP = tf.keras.layers.Dense(
            units=factor_num, activation='relu')(layer2_MLP)
        layer3_MLP = tf.keras.layers.Dropout(rate=dropout)(layer3_MLP)

    with tf.name_scope("concatenation"):
        concatenation = tf.concat([GMF, layer3_MLP], axis=-1)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(concatenation)

    model = tf.keras.Model(inputs=[user, item] + add_feature_input_layers, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
    return model
