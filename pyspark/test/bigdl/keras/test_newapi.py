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
import numpy as np
from bigdl.keras.converter import WeightsConverter

np.random.seed(1337)  # for reproducibility


class TestLayer(BigDLTestCase):

    def test_embedding(self):
        input_data = np.random.randint(1000, size=(32, 10))
        blayer = BLayer.Embedding(1000, 64, input_shape=(10, ))
        klayer = KLayer.Embedding(1000, 64, input_length=10)
        self.compare_newapi(klayer, blayer, input_data,
                            WeightsConverter.convert_embedding)

    def test_batchnormalization(self):
        K.set_image_dim_ordering("th")
        input_data = np.random.random_sample([2, 5, 32, 32])
        blayer = BLayer.BatchNormalization(input_shape=(5, 32, 32))
        klayer = KLayer.BatchNormalization(axis=1, input_shape=(5, 32, 32))
        self.compare_newapi(klayer, blayer, input_data,
                            WeightsConverter.convert_batchnormalization)
        K.set_image_dim_ordering("tf")
        input_data2 = np.random.random_sample([2, 32, 32, 4])
        blayer = BLayer.BatchNormalization(dim_ordering="tf", input_shape=(32, 32, 4))
        klayer = KLayer.BatchNormalization(axis=-1, input_shape=(32, 32, 4))
        self.compare_newapi(klayer, blayer, input_data2,
                            WeightsConverter.convert_batchnormalization)

    def test_merge_sum(self):
        b1 = BLayer.InputLayer(input_shape=(3, 5))
        b2 = BLayer.InputLayer(input_shape=(3, 5))
        blayer = BLayer.Merge(layers=[b1, b2], mode="sum")
        k1 = KLayer.InputLayer(input_shape=(3, 5))
        k2 = KLayer.InputLayer(input_shape=(3, 5))
        klayer = KLayer.Merge(layers=[k1, k2], mode="sum")
        input_data = [np.random.random([2, 3, 5]), np.random.random([2, 3, 5])]
        self.compare_newapi(klayer, blayer, input_data)

    def test_merge_mul(self):
        b1 = BLayer.InputLayer(input_shape=(3, 5))
        b2 = BLayer.InputLayer(input_shape=(3, 5))
        blayer = BLayer.Merge(layers=[b1, b2], mode="mul")
        k1 = KLayer.InputLayer(input_shape=(3, 5))
        k2 = KLayer.InputLayer(input_shape=(3, 5))
        klayer = KLayer.Merge(layers=[k1, k2], mode="mul")
        input_data = [np.random.random([2, 3, 5]), np.random.random([2, 3, 5])]
        self.compare_newapi(klayer, blayer, input_data)

    def test_merge_ave(self):
        b1 = BLayer.InputLayer(input_shape=(2, 5, 8))
        b2 = BLayer.InputLayer(input_shape=(2, 5, 8))
        blayer = BLayer.Merge(layers=[b1, b2], mode="ave")
        k1 = KLayer.InputLayer(input_shape=(2, 5, 8))
        k2 = KLayer.InputLayer(input_shape=(2, 5, 8))
        klayer = KLayer.Merge(layers=[k1, k2], mode="ave")
        input_data = [np.random.random([3, 2, 5, 8]), np.random.random([3, 2, 5, 8])]
        self.compare_newapi(klayer, blayer, input_data)

    def test_merge_max(self):
        b1 = BLayer.InputLayer(input_shape=(2, 5, 8))
        b2 = BLayer.InputLayer(input_shape=(2, 5, 8))
        blayer = BLayer.Merge(layers=[b1, b2], mode="max")
        k1 = KLayer.InputLayer(input_shape=(2, 5, 8))
        k2 = KLayer.InputLayer(input_shape=(2, 5, 8))
        klayer = KLayer.Merge(layers=[k1, k2], mode="max")
        input_data = [np.random.random([3, 2, 5, 8]), np.random.random([3, 2, 5, 8])]
        self.compare_newapi(klayer, blayer, input_data)

    def test_merge_concat(self):
        b1 = BLayer.InputLayer(input_shape=(2, 5, 11))
        b2 = BLayer.InputLayer(input_shape=(2, 5, 8))
        blayer = BLayer.Merge(layers=[b1, b2], mode="concat")
        k1 = KLayer.InputLayer(input_shape=(2, 5, 11))
        k2 = KLayer.InputLayer(input_shape=(2, 5, 8))
        klayer = KLayer.Merge(layers=[k1, k2], mode="concat")
        input_data = [np.random.random([3, 2, 5, 11]), np.random.random([3, 2, 5, 8])]
        self.compare_newapi(klayer, blayer, input_data)

    def test_merge_dot(self):
        b1 = BLayer.InputLayer(input_shape=(4, ))
        b2 = BLayer.InputLayer(input_shape=(4, ))
        blayer = BLayer.Merge(layers=[b1, b2], mode="dot")
        k1 = KLayer.InputLayer(input_shape=(4, ))
        k2 = KLayer.InputLayer(input_shape=(4, ))
        klayer = KLayer.Merge(layers=[k1, k2], mode="dot")
        input_data = [np.random.random([2, 4]), np.random.random([2, 4])]
        self.compare_newapi(klayer, blayer, input_data)

    def test_merge_cos(self):
        b1 = BLayer.InputLayer(input_shape=(3, ))
        b2 = BLayer.InputLayer(input_shape=(3, ))
        blayer = BLayer.Merge(layers=[b1, b2], mode="cos")
        k1 = KLayer.InputLayer(input_shape=(3, ))
        k2 = KLayer.InputLayer(input_shape=(3, ))
        klayer = KLayer.Merge(layers=[k1, k2], mode="cos")
        input_data = [np.random.random([2, 3]), np.random.random([2, 3])]
        self.compare_newapi(klayer, blayer, input_data)

    def test_lenet_shape(self):
        from bigdl.examples.lenet.lenet import build_model
        model = build_model(10)
        input_shape = model.get_input_shape()
        np.testing.assert_allclose((28, 28, 1), input_shape[1:])
        output_shape = model.get_output_shape()
        np.testing.assert_allclose((10, ), output_shape[1:])

    def test_graph(self):
        x1 = BLayer.Input(input_shape=(8, ))
        x2 = BLayer.Input(input_shape=(6, ))
        y1 = BLayer.Dense(10)(x1)
        y2 = BLayer.Dense(10)(x2)
        model = BLayer.Model([x1, x2], [y1, y2])
        input_shapes = model.get_input_shape()
        np.testing.assert_allclose((8, ), input_shapes[0][1:])
        np.testing.assert_allclose((6, ), input_shapes[1][1:])


if __name__ == "__main__":
    pytest.main([__file__])
