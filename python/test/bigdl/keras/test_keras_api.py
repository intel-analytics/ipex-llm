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
from bigdl.keras.converter import WeightsConverter
from bigdl.dataset.dataset import *
from bigdl.nn.keras.topology import Model as BModel
from bigdl.nn.keras.topology import Sequential as BSequential
from keras.engine import merge as kmerge, Model as KModel
from keras.models import Sequential as KSequential

np.random.seed(1337)  # for reproducibility


class TestKerasAPI(BigDLTestCase):

    def test_embedding(self):
        input_data = np.random.randint(1000, size=(32, 10))
        blayer = BLayer.Embedding(1000, 64, input_shape=(10, ))
        klayer = KLayer.Embedding(1000, 64, input_length=10)
        self.compare_newapi(klayer, blayer, input_data,
                            WeightsConverter.convert_embedding)

    def test_batchnormalization(self):
        K.set_image_dim_ordering("th")
        input_data = np.random.random_sample([2, 5, 32, 32])
        blayer = BLayer.BatchNormalization(axis=1, input_shape=(5, 32, 32))
        klayer = KLayer.BatchNormalization(axis=1, input_shape=(5, 32, 32))
        self.compare_newapi(klayer, blayer, input_data,
                            WeightsConverter.convert_batchnormalization)
        K.set_image_dim_ordering("tf")
        input_data2 = np.random.random_sample([2, 32, 32, 4])
        blayer = BLayer.BatchNormalization(axis=-1, dim_ordering="tf", input_shape=(32, 32, 4))
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
        x1 = BLayer.Input(shape=(8, ))
        x2 = BLayer.Input(shape=(6, ))
        y1 = BLayer.Dense(10)(x1)
        y2 = BLayer.Dense(10)(x2)
        model = BModel([x1, x2], [y1, y2])
        input_shapes = model.get_input_shape()
        output_shapes = model.get_output_shape()
        np.testing.assert_allclose((8, ), input_shapes[0][1:])
        np.testing.assert_allclose((6, ), input_shapes[1][1:])
        np.testing.assert_allclose((10, ), output_shapes[0][1:])
        np.testing.assert_allclose((10, ), output_shapes[1][1:])

    def test_train(self):
        x = np.random.random([32, 10])
        y = np.random.random([32, ])
        model = BSequential()
        model.add(BLayer.Dense(5, input_shape=(10, )))
        model.compile(optimizer="sgd", loss="mse", metrics=["accuracy"])
        model.fit(x, y, batch_size=8, nb_epoch=2, validation_data=(x, y))
        model.evaluate(x, y, batch_size=8)
        model.predict(x)

    def test_train_dataset(self):
        images = []
        labels = []
        for i in range(0, 8):
            features = np.random.uniform(0, 1, (200, 200, 3))
            label = np.array([2])
            images.append(features)
            labels.append(label)
        image_frame = DistributedImageFrame(self.sc.parallelize(images),
                                            self.sc.parallelize(labels))

        transformer = Pipeline([BytesToMat(), Resize(256, 256), CenterCrop(224, 224),
                                ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                MatToTensor(), ImageFrameToSample(target_keys=['label'])])
        data_set = DataSet.image_frame(image_frame).transform(transformer)

        model = BSequential()
        model.add(BLayer.Convolution2D(1, 5, 5, input_shape=(3, 224, 224)))
        model.add(BLayer.Reshape((1*220*220, )))
        model.add(BLayer.Dense(20, activation="softmax"))
        model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(data_set, batch_size=8, nb_epoch=2, validation_data=data_set)

    def convert_two_dense_model(self, kmodel, weights):
        return [weights[2].T, weights[3], weights[0].T, weights[1]]

    def test_merge_method_sum(self):
        bx1 = BLayer.Input(shape=(8, ))
        bx2 = BLayer.Input(shape=(6, ))
        by1 = BLayer.Dense(10)(bx1)
        by2 = BLayer.Dense(10)(bx2)
        bz = BLayer.merge([by1, by2], mode="sum")
        bmodel = BModel([bx1, bx2], bz, name="graph1")

        kx1 = KLayer.Input(shape=(8, ))
        kx2 = KLayer.Input(shape=(6, ))
        ky1 = KLayer.Dense(10)(kx1)
        ky2 = KLayer.Dense(10)(kx2)
        kz = kmerge([ky1, ky2], mode="sum")
        kmodel = KModel([kx1, kx2], kz)

        input_data = [np.random.random([2, 8]), np.random.random([2, 6])]
        self.compare_newapi(kmodel, bmodel, input_data, self.convert_two_dense_model)

    def test_merge_method_model_concat(self):
        bx1 = BLayer.Input(shape=(4, ))
        bx2 = BLayer.Input(shape=(5, ))
        by1 = BLayer.Dense(6, activation="sigmoid")(bx1)
        bbranch1 = BModel(bx1, by1)(bx1)
        bbranch2 = BLayer.Dense(8)(bx2)
        bz = BLayer.merge([bbranch1, bbranch2], mode="concat")
        bmodel = BModel([bx1, bx2], bz)

        kx1 = KLayer.Input(shape=(4, ))
        kx2 = KLayer.Input(shape=(5, ))
        ky1 = KLayer.Dense(6, activation="sigmoid")(kx1)
        kbranch1 = KModel(kx1, ky1)(kx1)
        kbranch2 = KLayer.Dense(8)(kx2)
        kz = KLayer.merge([kbranch1, kbranch2], mode="concat")
        kmodel = KModel([kx1, kx2], kz)

        input_data = [np.random.random([2, 4]), np.random.random([2, 5])]
        self.compare_newapi(kmodel, bmodel, input_data, self.convert_two_dense_model)

    def test_merge_method_seq_concat(self):
        bx1 = BLayer.Input(shape=(10, ))
        bx2 = BLayer.Input(shape=(10, ))
        by1 = BLayer.Dense(12, activation="sigmoid")(bx1)
        bbranch1_node = BModel(bx1, by1)(bx1)
        bbranch2 = BSequential()
        bbranch2.add(BLayer.Dense(12, input_dim=10))
        bbranch2_node = bbranch2(bx2)
        bz = BLayer.merge([bbranch1_node, bbranch2_node], mode="concat")
        bmodel = BModel([bx1, bx2], bz)

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
        self.compare_newapi(kmodel, bmodel, input_data, self.convert_two_dense_model)


if __name__ == "__main__":
    pytest.main([__file__])
