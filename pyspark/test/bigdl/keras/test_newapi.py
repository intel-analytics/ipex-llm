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

import pytest

from test.bigdl.test_utils import BigDLTestCase
import bigdl.nn.keras.layer as BLayer
import keras.layers as KLayer
import keras.backend as K
from keras.models import Sequential as KSequential
import numpy as np
from numpy.testing import assert_allclose
from bigdl.keras.converter import WeightsConverter

np.random.seed(1337)  # for reproducibility


class TestLayer(BigDLTestCase):

    def __compare_results(self, klayer, blayer, weight_converter, input_data, rtol=1e-6, atol=1e-6):
        bmodel = BLayer.Sequential()
        bmodel.add(blayer)
        kmodel = KSequential()
        kmodel.add(klayer)
        if isinstance(blayer, BLayer.BatchNormalization):
            k_running_mean = K.eval(klayer.running_mean)
            k_running_std = K.eval(klayer.running_std)
            blayer.set_running_mean(k_running_mean)
            blayer.set_running_std(k_running_std)
        bmodel.set_weights(weight_converter(klayer, kmodel.get_weights()))
        boutput = bmodel.forward(input_data)
        koutput = kmodel.predict(input_data)
        assert_allclose(boutput, koutput, rtol=rtol, atol=atol)

    def test_embedding(self):
        input_data = np.random.randint(1000, size=(32, 10))
        blayer = BLayer.Embedding(1000, 64, input_shape=(10, ))
        klayer = KLayer.Embedding(1000, 64, input_length=10)
        self.__compare_results(klayer, blayer, WeightsConverter.convert_embedding, input_data)

    def test_batchnormalization(self):
        K.set_image_dim_ordering("th")
        input_data = np.random.random([2, 3, 32, 32])
        blayer = BLayer.BatchNormalization(beta_init=np.zeros(3), input_shape=(3, 32, 32))
        klayer = KLayer.BatchNormalization(axis=1, input_shape=(3, 32, 32))
        self.__compare_results(klayer, blayer, WeightsConverter.convert_batchnormalization, input_data)


if __name__ == "__main__":
    pytest.main([__file__])
