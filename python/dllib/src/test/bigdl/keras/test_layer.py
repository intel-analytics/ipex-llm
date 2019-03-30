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
from zoo.pipeline.api.keras.models import Model as ZModel, Sequential as ZSequential
import keras.layers as KLayer
from keras.engine import merge as kmerge, Model as KModel
from keras.models import Sequential as KSequential
import keras.backend as K
from bigdl.keras.converter import WeightsConverter
from zoo.pipeline.api.keras import regularizers

np.random.seed(1337)  # for reproducibility


class TestLayer(ZooTestCase):

    def test_embedding(self):
        input_data = np.random.randint(1000, size=(32, 10))
        zlayer = ZLayer.Embedding(1000, 64, input_shape=(10, ))
        klayer = KLayer.Embedding(1000, 64, input_length=10)
        self.compare_layer(klayer, zlayer, input_data,
                           WeightsConverter.convert_embedding)

    def test_batchnormalization(self):
        print("Running batch normal test")
        K.set_image_dim_ordering("th")
        input_data = np.random.random_sample([2, 5, 32, 32])
        zlayer = ZLayer.BatchNormalization(axis=1, input_shape=(5, 32, 32))
        klayer = KLayer.BatchNormalization(axis=1, input_shape=(5, 32, 32))
        self.compare_layer(klayer, zlayer, input_data,
                           WeightsConverter.convert_batchnormalization)
        K.set_image_dim_ordering("tf")
        input_data2 = np.random.random_sample([2, 32, 32, 4])
        zlayer = ZLayer.BatchNormalization(axis=-1, dim_ordering="tf", input_shape=(32, 32, 4))
        klayer = KLayer.BatchNormalization(axis=-1, input_shape=(32, 32, 4))
        self.compare_layer(klayer, zlayer, input_data2,
                           WeightsConverter.convert_batchnormalization)
        K.set_image_dim_ordering("th")
        input_data = np.random.random_sample([2, 5])
        zlayer = ZLayer.BatchNormalization(axis=1, input_shape=(5,))
        klayer = KLayer.BatchNormalization(axis=1, input_shape=(5,))
        self.compare_layer(klayer, zlayer, input_data,
                           WeightsConverter.convert_batchnormalization)

    def test_merge_sum(self):
        z1 = ZLayer.InputLayer(input_shape=(3, 5))
        z2 = ZLayer.InputLayer(input_shape=(3, 5))
        zlayer = ZLayer.Merge(layers=[z1, z2], mode="sum")
        k1 = KLayer.InputLayer(input_shape=(3, 5))
        k2 = KLayer.InputLayer(input_shape=(3, 5))
        klayer = KLayer.Merge(layers=[k1, k2], mode="sum")
        input_data = [np.random.random([2, 3, 5]), np.random.random([2, 3, 5])]
        self.compare_layer(klayer, zlayer, input_data)

    def test_merge_mul(self):
        z1 = ZLayer.InputLayer(input_shape=(3, 5))
        z2 = ZLayer.InputLayer(input_shape=(3, 5))
        zlayer = ZLayer.Merge(layers=[z1, z2], mode="mul")
        k1 = KLayer.InputLayer(input_shape=(3, 5))
        k2 = KLayer.InputLayer(input_shape=(3, 5))
        klayer = KLayer.Merge(layers=[k1, k2], mode="mul")
        input_data = [np.random.random([2, 3, 5]), np.random.random([2, 3, 5])]
        self.compare_layer(klayer, zlayer, input_data)

    def test_merge_ave(self):
        z1 = ZLayer.InputLayer(input_shape=(2, 5, 8))
        z2 = ZLayer.InputLayer(input_shape=(2, 5, 8))
        zlayer = ZLayer.Merge(layers=[z1, z2], mode="ave")
        k1 = KLayer.InputLayer(input_shape=(2, 5, 8))
        k2 = KLayer.InputLayer(input_shape=(2, 5, 8))
        klayer = KLayer.Merge(layers=[k1, k2], mode="ave")
        input_data = [np.random.random([3, 2, 5, 8]), np.random.random([3, 2, 5, 8])]
        self.compare_layer(klayer, zlayer, input_data)

    def test_merge_max(self):
        z1 = ZLayer.InputLayer(input_shape=(2, 5, 8))
        z2 = ZLayer.InputLayer(input_shape=(2, 5, 8))
        zlayer = ZLayer.Merge(layers=[z1, z2], mode="max")
        k1 = KLayer.InputLayer(input_shape=(2, 5, 8))
        k2 = KLayer.InputLayer(input_shape=(2, 5, 8))
        klayer = KLayer.Merge(layers=[k1, k2], mode="max")
        input_data = [np.random.random([3, 2, 5, 8]), np.random.random([3, 2, 5, 8])]
        self.compare_layer(klayer, zlayer, input_data)

    def test_merge_concat(self):
        z1 = ZLayer.InputLayer(input_shape=(2, 5, 11))
        z2 = ZLayer.InputLayer(input_shape=(2, 5, 8))
        zlayer = ZLayer.Merge(layers=[z1, z2], mode="concat")
        k1 = KLayer.InputLayer(input_shape=(2, 5, 11))
        k2 = KLayer.InputLayer(input_shape=(2, 5, 8))
        klayer = KLayer.Merge(layers=[k1, k2], mode="concat")
        input_data = [np.random.random([3, 2, 5, 11]), np.random.random([3, 2, 5, 8])]
        self.compare_layer(klayer, zlayer, input_data)

    def test_merge_dot(self):
        z1 = ZLayer.InputLayer(input_shape=(4, ))
        z2 = ZLayer.InputLayer(input_shape=(4, ))
        zlayer = ZLayer.Merge(layers=[z1, z2], mode="dot")
        k1 = KLayer.InputLayer(input_shape=(4, ))
        k2 = KLayer.InputLayer(input_shape=(4, ))
        klayer = KLayer.Merge(layers=[k1, k2], mode="dot")
        input_data = [np.random.random([2, 4]), np.random.random([2, 4])]
        self.compare_layer(klayer, zlayer, input_data)

    def test_merge_cos(self):
        z1 = ZLayer.InputLayer(input_shape=(3, ))
        z2 = ZLayer.InputLayer(input_shape=(3, ))
        zlayer = ZLayer.Merge(layers=[z1, z2], mode="cos")
        k1 = KLayer.InputLayer(input_shape=(3, ))
        k2 = KLayer.InputLayer(input_shape=(3, ))
        klayer = KLayer.Merge(layers=[k1, k2], mode="cos")
        input_data = [np.random.random([2, 3]), np.random.random([2, 3])]
        self.compare_layer(klayer, zlayer, input_data)

    def convert_two_dense(self, kmodel, weights):
        return [weights[2].T, weights[3], weights[0].T, weights[1]]

    def test_merge_method_sum(self):
        zx1 = ZLayer.Input(shape=(8, ))
        zx2 = ZLayer.Input(shape=(6, ))
        zy1 = ZLayer.Dense(10)(zx1)
        zy2 = ZLayer.Dense(10)(zx2)
        zz = ZLayer.merge([zy1, zy2], mode="sum")
        zmodel = ZModel([zx1, zx2], zz, name="graph1")

        kx1 = KLayer.Input(shape=(8, ))
        kx2 = KLayer.Input(shape=(6, ))
        ky1 = KLayer.Dense(10)(kx1)
        ky2 = KLayer.Dense(10)(kx2)
        kz = kmerge([ky1, ky2], mode="sum")
        kmodel = KModel([kx1, kx2], kz)

        input_data = [np.random.random([2, 8]), np.random.random([2, 6])]
        self.compare_layer(kmodel, zmodel, input_data, self.convert_two_dense)

    def test_merge_method_model_concat(self):
        zx1 = ZLayer.Input(shape=(4, ))
        zx2 = ZLayer.Input(shape=(5, ))
        zy1 = ZLayer.Dense(6, activation="sigmoid")(zx1)
        zbranch1 = ZModel(zx1, zy1)(zx1)
        zbranch2 = ZLayer.Dense(8)(zx2)
        zz = ZLayer.merge([zbranch1, zbranch2], mode="concat")
        zmodel = ZModel([zx1, zx2], zz)

        kx1 = KLayer.Input(shape=(4, ))
        kx2 = KLayer.Input(shape=(5, ))
        ky1 = KLayer.Dense(6, activation="sigmoid")(kx1)
        kbranch1 = KModel(kx1, ky1)(kx1)
        kbranch2 = KLayer.Dense(8)(kx2)
        kz = KLayer.merge([kbranch1, kbranch2], mode="concat")
        kmodel = KModel([kx1, kx2], kz)

        input_data = [np.random.random([2, 4]), np.random.random([2, 5])]
        self.compare_layer(kmodel, zmodel, input_data, self.convert_two_dense)

    def test_merge_method_seq_concat(self):
        zx1 = ZLayer.Input(shape=(10, ))
        zx2 = ZLayer.Input(shape=(10, ))
        zy1 = ZLayer.Dense(12, activation="sigmoid")(zx1)
        zbranch1_node = ZModel(zx1, zy1)(zx1)
        zbranch2 = ZSequential()
        zbranch2.add(ZLayer.Dense(12, input_dim=10))
        zbranch2_node = zbranch2(zx2)
        zz = ZLayer.merge([zbranch1_node, zbranch2_node], mode="concat")
        zmodel = ZModel([zx1, zx2], zz)

        kx1 = KLayer.Input(shape=(10, ))
        kx2 = KLayer.Input(shape=(10, ))
        ky1 = KLayer.Dense(12, activation="sigmoid")(kx1)
        kbranch1_node = KModel(kx1, ky1)(kx1)
        kbranch2 = KSequential()
        kbranch2.add(KLayer.Dense(12, input_dim=10))
        kbranch2_node = kbranch2(kx2)
        kz = KLayer.merge([kbranch1_node, kbranch2_node], mode="concat")
        kmodel = KModel([kx1, kx2], kz)

        input_data = [np.random.random([2, 10]), np.random.random([2, 10])]
        self.compare_layer(kmodel, zmodel, input_data, self.convert_two_dense)

    def test_reshape(self):
        a = np.random.random((2, 2, 3, 4))
        i1 = ZLayer.Input(shape=(2, 3, 4))
        s = ZLayer.Reshape((-1, 2, 12))(i1)
        m = ZModel(i1, s)
        # predict should not generate exception
        y = m.predict(a, distributed=False)

    def test_regularizer(self):
        model = ZSequential()
        model.add(ZLayer.Dense(16, W_regularizer=regularizers.l2(0.001),
                               activation='relu', input_shape=(10000,)))
        model.summary()
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])


if __name__ == "__main__":
    pytest.main([__file__])
