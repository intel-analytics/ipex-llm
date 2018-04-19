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

from test.zoo.pipeline.utils.test_utils import ZooTestCase
import zoo.pipeline.api.keras.layers as ZLayer
import keras.layers as KLayer
import keras.backend as K
from bigdl.keras.converter import WeightsConverter

np.random.seed(1337)  # for reproducibility


class TestLayer(ZooTestCase):

    def test_embedding(self):
        input_data = np.random.randint(1000, size=(32, 10))
        blayer = ZLayer.Embedding(1000, 64, input_shape=(10, ))
        klayer = KLayer.Embedding(1000, 64, input_length=10)
        self.compare_layer(klayer, blayer, input_data,
                           WeightsConverter.convert_embedding)

    def test_batchnormalization(self):
        K.set_image_dim_ordering("th")
        input_data = np.random.random_sample([2, 5, 32, 32])
        blayer = ZLayer.BatchNormalization(axis=1, input_shape=(5, 32, 32))
        klayer = KLayer.BatchNormalization(axis=1, input_shape=(5, 32, 32))
        self.compare_layer(klayer, blayer, input_data,
                           WeightsConverter.convert_batchnormalization)
        K.set_image_dim_ordering("tf")
        input_data2 = np.random.random_sample([2, 32, 32, 4])
        blayer = ZLayer.BatchNormalization(axis=-1, dim_ordering="tf", input_shape=(32, 32, 4))
        klayer = KLayer.BatchNormalization(axis=-1, input_shape=(32, 32, 4))
        self.compare_layer(klayer, blayer, input_data2,
                           WeightsConverter.convert_batchnormalization)


if __name__ == "__main__":
   pytest.main([__file__])
