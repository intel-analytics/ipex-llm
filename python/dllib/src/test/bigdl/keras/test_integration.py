#
# Copyright 2018 Analytics Zoo Authors.
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

import pytest

import numpy as np

from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *

np.random.seed(1337)  # for reproducibility


class TestKerasAPI():

    def test_sequential(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(8, )))
        model.add(Dense(10))
        model.add(Dense(12))
        input_shape = model.get_input_shape()
        output_shape = model.get_output_shape()
        np.testing.assert_allclose((8,), input_shape[1:])
        np.testing.assert_allclose((12,), output_shape[1:])

    def test_graph(self):
        input = Input(shape=(8, ))
        dense = Dense(10)(input)
        output = Dense(12)(dense)
        model = Model(input, output)
        input_shape = model.get_input_shape()
        output_shape = model.get_output_shape()
        np.testing.assert_allclose((8,), input_shape[1:])
        np.testing.assert_allclose((12,), output_shape[1:])

    def test_training(self):
        model = Sequential()
        model.add(Dense(4, activation="relu", input_shape=(10, )))
        x = np.random.random([300, 10])
        y = np.random.random([300, ])
        model.compile(optimizer="sgd", loss="mae", metrics=["accuracy"])
        model.fit(x, y, batch_size=112, nb_epoch=2, validation_data=(x, y))
        model.evaluate(x, y, batch_size=112)
        model.predict(x)


if __name__ == "__main__":
   pytest.main([__file__])
