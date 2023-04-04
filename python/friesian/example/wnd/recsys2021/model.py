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

from bigdl.dllib.keras.layers import *
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, concatenate, Lambda
import tensorflow as tf


class ColumnInfoTower(object):
    """
    The same data information shared by the Tower model and its feature generation part.

    Each instance could contain the following fields:
    indicator_cols: Data of indicator_cols will be fed into model as one-hot vectors.
                    List of String. Default is an empty list.
    indicator_dims: Dimensions of indicator_cols. The dimensions of the data in
                    indicator_cols should be within the range of indicator_dims.
                    List of int. Default is an empty list.
    embed_cols: Data of embed_cols will be fed into the  model as embeddings.
                List of String. Default is an empty list.
    embed_in_dims: Input dimension of the data in embed_cols. The dimensions of the data in
                   embed_cols should be within the range of embed_in_dims.
                   List of int. Default is an empty list.
    embed_out_dims: The out dimensions of embeddings. List of int. Default is an empty list.
    numerical_cols: Data of numerical_cols will be treated as continuous values
                    List of String. Default is an empty list.
    numerical_dims: dimention of numerical_cols.
    name: string
    """

    def __init__(self, indicator_cols=None, indicator_dims=None,
                 embed_cols=None, embed_in_dims=None, embed_out_dims=None,
                 numerical_cols=None, numerical_dims=None, name=None, bigdl_type="float"):
        self.indicator_cols = [] if not indicator_cols else indicator_cols
        self.indicator_dims = [] if not indicator_dims else [int(d) for d in indicator_dims]
        self.embed_cols = [] if not embed_cols else embed_cols
        self.embed_in_dims = [] if not embed_in_dims else [int(d) for d in embed_in_dims]
        self.embed_out_dims = [] if not embed_out_dims else [int(d) for d in embed_out_dims]
        self.numerical_cols = [] if not numerical_cols else numerical_cols
        self.numerical_dims = [] if not numerical_dims else [int(d) for d in numerical_dims]
        self.name = name
        self.bigdl_type = bigdl_type

    def __reduce__(self):
        return ColumnInfoTower, (self.indicator_cols, self.indicator_dims,
                                 self.embed_cols, self.embed_in_dims, self.embed_out_dims,
                                 self.numerical_cols, self.numerical_dims, self.name)

    def get_name_list(self):
        name_list = self.indicator_cols + self.embed_cols + self.numerical_cols
        return name_list

    def __str__(self):
        return "ColumnInfoTower {indicator_cols: %s, indicator_dims: %s, embed_cols: %s, " \
               "embed_in_dims: %s, embed_out_dims: %s, numerical_cols: %s, name: %s}" \
               % (self.indicator_cols, self.indicator_dims, self.embed_cols,
                  self.embed_in_dims, self.embed_out_dims, self.numerical_cols,
                  self.name)


class TwoTowerModel(object):
    """
    The Two Tower model used for recommendation retrieval.

    # Arguments
    user_col_info: user column information.
    item_col_info: item column information.
    hidden_layers: Units of hidden layers for each tower.
                   Tuple of positive int. Default is [1024, 512, 128].
    """
    def __init__(self, user_col_info, item_col_info, hidden_layers=[1024, 512, 128]):
        self.user_col_info = user_col_info
        self.item_col_info = item_col_info
        self.hidden_layers = hidden_layers

    def build_model(self):
        hidden_layers = self.hidden_layers

        # one tower
        def build_1tower(col_info):
            indicator_input_layers = []
            indicator_layers = []
            for i in range(len(col_info.indicator_cols)):
                indicator_input_layers.append(Input(shape=(),
                                                    name=col_info.name + col_info.indicator_cols[i],
                                                    dtype="int32"))
                indicator_layers.append(tf.keras.backend.one_hot(indicator_input_layers[i],
                                                                 col_info.indicator_dims[i] + 1))

            embed_input_layers = []
            embed_layers = []
            for i in range(len(col_info.embed_in_dims)):
                embed_input_layers.append(Input(shape=(),
                                                name=col_info.name + col_info.embed_cols[i],
                                                dtype="int32"))
                iembed = Embedding(col_info.embed_in_dims[i] + 1,
                                   output_dim=col_info.embed_out_dims[i])(embed_input_layers[i])
                flat_embed = Flatten()(iembed)
                embed_layers.append(flat_embed)

            numerical_input_layers = []
            for i in range(len(col_info.numerical_cols)):
                numerical_input_layers.append(
                    Input(shape=(col_info.numerical_dims[i]),
                          name=col_info.name + col_info.numerical_cols[i]))

            cocated = indicator_layers + embed_layers + numerical_input_layers
            concated = cocated[0] if len(cocated) == 1 else concatenate(cocated, axis=1)

            linear = Dense(hidden_layers[0], activation="relu")(concated)
            for ilayer in range(1, len(hidden_layers)):
                linear_mid = Dense(hidden_layers[ilayer], activation="relu")(linear)
                linear = linear_mid
            last_linear = linear

            nomorlized = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1),
                                name=col_info.name + "_embed_output")
            out = nomorlized(last_linear)
            input = indicator_input_layers + embed_input_layers + numerical_input_layers

            return input, out

        user_input, user_out = build_1tower(col_info=self.user_col_info)
        item_input, item_out = build_1tower(col_info=self.item_col_info)

        # dot operation to compute cosine similarity
        doted = tf.keras.layers.Dot(axes=1)

        out = doted([user_out, item_out])

        model = tf.keras.Model(user_input + item_input, out)
        return model


def get_1tower_model(model, column_info):
    """
    Conect layers of each tower to output embeddings of user or item
    """
    col_list = column_info.get_name_list()
    input_names = [column_info.name + x for x in col_list]
    inputs = [model.get_layer(x).input for x in input_names]
    output = model.get_layer(column_info.name + "_embed_output").output
    one_tower_model = tf.keras.models.Model(inputs, output)
    return one_tower_model
