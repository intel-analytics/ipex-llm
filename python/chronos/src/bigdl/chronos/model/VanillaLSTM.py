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


def model_creator(config):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
    import tensorflow as tf

    inp = Input(shape=(None, config["input_dim"]))
    if "lstm_1_units" in config and "lstm_2_units" in config:
        lstm_units = (config["lstm_1_units"], config["lstm_2_units"])
    else:
        lstm_units = config.get("lstm_units", [32, 32])
    if "dropout_1" in config and "dropout_2" in config:
        dropout_rates = (config["dropout_1"], config["dropout_2"])
    else:
        dropout_rates = config.get("dropouts", 0.2)
    lstm_units = [lstm_units] if not isinstance(lstm_units, Iterable) else lstm_units
    for i, unit in enumerate(lstm_units):
        return_sequences = True if i != len(lstm_units) - 1 else False
        dropout_rate = dropout_rates[i] if isinstance(dropout_rates, Iterable) else dropout_rates
        lstm_input = inp if i == 0 else dropout
        lstm = LSTM(units=unit, return_sequences=return_sequences)(lstm_input)
        dropout = Dropout(rate=dropout_rate)(lstm)
    out = Dense(config["output_dim"])(dropout)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss=config.get("loss", "mse"),
                  optimizer=getattr(tf.keras.optimizers, config.get("optim", "Adam"))
                  (learning_rate=config.get("lr", 0.001)),
                  metrics=[config.get("metric", "mse")])
    return model


def check_iter_type(obj, type):
    return isinstance(obj, type) or \
        (isinstance(obj, Iterable) and all(isinstance(o, type) for o in obj))


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
