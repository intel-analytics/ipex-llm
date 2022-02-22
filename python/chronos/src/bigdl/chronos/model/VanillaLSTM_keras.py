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
from bigdl.orca.automl.model.base_keras_model import KerasBaseModel
from collections.abc import Iterable
import numpy as np
import tensorflow as tf
from bigdl.nano.tf import keras
from bigdl.nano.tf.keras import Sequential, Model
from keras.layers import LSTM, Reshape, Dense, Input


class LSTMModel(Model):
    def __init__(self, input_dim, hidden_dim, layer_num, dropout, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.lstm_list = Sequential([Input(shape=(None, input_dim))])
        for layer in range(self.layer_num - 1):
            self.lstm_list.add(LSTM(self.hidden_dim[layer],
                                    return_sequences=True,
                                    dropout=dropout[layer],
                                    activation="linear",
                                    name="lstm_" + str(layer+1)))
        self.lstm_list.add(LSTM(self.hidden_dim[-1],
                                dropout=dropout[-1],
                                activation="linear",
                                name="lstm_" + str(self.layer_num)))
        self.fc = Dense(output_dim)
        self.out_shape = Reshape((1, output_dim), input_shape=(output_dim,))


    def call(self, input_seq):
        lstm_out = input_seq
        out = self.lstm_list(lstm_out)
        out = self.fc(out)
        out = self.out_shape(out)
        return out


def model_creator(config):
    layer_num = config.get('layer_num', 2)
    hidden_dim = config.get('hidden_dim', [32, 16])
    dropout = config.get('dropout', [0.2, 0.1])
    if isinstance(hidden_dim, list):
        assert len(hidden_dim) == layer_num, \
            "length of hidden_dim should be equal to layer_num"
    if isinstance(dropout, list):
        assert len(dropout) == layer_num, \
            "length of dropout should be equal to layer_num"
    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim]*layer_num
    if isinstance(dropout, (float, int)):
        dropout = [dropout]*layer_num
    model = LSTMModel(input_dim=config['input_feature_num'],
                      hidden_dim=hidden_dim,
                      layer_num=layer_num,
                      dropout=dropout,
                      output_dim=config['output_feature_num'])

    learning_rate = config.get('lr', 1e-3)
    optimizer = getattr(tf.keras.optimizers, config.get('optim', "Adam"))(learning_rate)
    model.compile(loss=config.get("loss", "mse"),
                  optimizer=optimizer,
                  metrics=[config.get("metric", "mse")])
    return model


class VanillaLSTM(KerasBaseModel):
    def __init__(self, check_optional_config=False, future_seq_len=1):
        super(VanillaLSTM, self).__init__(model_creator=model_creator,
                                          check_optional_config=check_optional_config)

    def _check_config(self, **config):
        super()._check_config(**config)
        assert isinstance(config["input_dim"], int), "'input_dim' should be int"
        assert isinstance(config["output_dim"], int), "'output_dim' should be int"
        lstm_name = "lstm_units"
        dropout_name = "dropouts"
        if lstm_name in config:
            if not check_iter_type(config[lstm_name], (int, np.integer)):
                raise ValueError(f"{lstm_name} should be int or an list/tuple of ints. "
                                 f"Got {config[lstm_name]}")
        if dropout_name in config:
            if not check_iter_type(config[dropout_name], (float, np.float)):
                raise ValueError(f"{dropout_name} should be float or a list/tuple of floats. "
                                 f"Got {config[dropout_name]}")
        if lstm_name in config and dropout_name in config:
            if (isinstance(config[lstm_name], int) and isinstance(config[dropout_name], Iterable)) \
                or (isinstance(config[lstm_name], Iterable) and
                    isinstance(config[dropout_name], Iterable) and
                    len(config[lstm_name]) != len(config[dropout_name])):
                raise ValueError(f"{lstm_name} should have the same elements num as {dropout_name}")

    def _get_required_parameters(self):
        return {"input_dim",
                "output_dim"
                } | super()._get_required_parameters()

    def _get_optional_parameters(self):
        return {"lstm_units",
                "dropouts",
                "optim",
                "lr"
                } | super()._get_optional_parameters()
