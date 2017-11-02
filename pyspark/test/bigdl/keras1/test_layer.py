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
from __future__ import print_function

import pytest
from keras.layers import *

np.random.seed(1337)  # for reproducibility
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers import Dense, Input
from bigdl.keras1.converter import *
from test.bigdl.test_utils import BigDLTestCase


class TestLayer(BigDLTestCase):

    def test_dense(self):
        input_data = np.random.random_sample([1, 10])
        dense = Dense(2, init='one', activation="relu", input_shape=(10, ))
        self.modelTestSingleLayer(input_data, dense, dump_weights=True, functional_api=False)
        self.modelTestSingleLayer(input_data, dense, dump_weights=True, functional_api=True)

    def test_timedistributeddense(self):
        input_data = np.random.random_sample([2, 4, 5])
        layer = TimeDistributedDense(6, input_shape=(4, 5))
        self.modelTestSingleLayer(input_data, layer, dump_weights=True, functional_api=False)
        self.modelTestSingleLayer(input_data, layer, dump_weights=True, functional_api=True)

    def test_embedding(self):
        # Test index start from 0
        input_data = np.array([[0, 1, 2, 99], [0, 4, 5, 99]])
        layer = Embedding(input_dim=100,  # index [0,99]
                          output_dim=64,  # vector dim
                          init='uniform',
                          input_length=None,
                          W_regularizer=None, activity_regularizer=None,
                          W_constraint=None,
                          mask_zero=False,
                          weights=None, dropout=0.)
        self.modelTestSingleLayer(input_data, layer, dump_weights=True, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, dump_weights=True, functional_api=False)
        # Random input
        input_data2 = np.random.randint(100, size=(10, 128))  # batch: 20, seqlen 128
        self.modelTestSingleLayer(input_data2, layer, dump_weights=True, functional_api=True)
        self.modelTestSingleLayer(input_data2, layer, dump_weights=True, functional_api=False)

        # TODO: add test that exception would be raised if input_lenght == 6
        with pytest.raises(Exception) as excinfo:
            layer = Embedding(input_dim=100,  # index [0,99]
                              output_dim=64,  # vector dim
                              init='uniform',
                              input_length=111,
                              W_regularizer=None, activity_regularizer=None,
                              W_constraint=None,
                              mask_zero=False,
                              weights=None, dropout=0.)
            input_data = np.random.randint(100, size=(10, 128))  # batch: 20, seqlen 128
            self.modelTestSingleLayer(input_data, layer, dump_weights=True)
        assert str(excinfo.value) == """The input_length doesn't match: 128 vs 111"""

    # TODO: Add more test case? activation
    def test_conv1D(self):
        input_data = np.random.random_sample([1, 10, 32])
        layer = Convolution1D(64, 3, border_mode='valid', input_shape=(10, 32))
        self.modelTestSingleLayer(input_data, layer, dump_weights=True, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, dump_weights=True, functional_api=False)

        layer2 = Convolution1D(64, 3, border_mode='same', input_shape=(10, 32))
        self.modelTestSingleLayer(input_data, layer2, dump_weights=True, functional_api=True)
        self.modelTestSingleLayer(input_data, layer2, dump_weights=True, functional_api=False)

        layer3 = Convolution1D(64, 3, border_mode='same', activation="relu", input_shape=(10, 32))
        self.modelTestSingleLayer(input_data, layer3, dump_weights=True, functional_api=True)
        self.modelTestSingleLayer(input_data, layer3, dump_weights=True, functional_api=False)

    def _load_keras(self, json_path, hdf5_path):
        with open(json_path, "r") as jp:
            kmodel = model_from_json(jp.read())
        kmodel.load_weights(hdf5_path)
        bmodel = DefinitionLoader.from_json_path(json_path)
        WeightLoader.load_weights(bmodel, kmodel, hdf5_path)  # TODO: refactor reability of this api
        return kmodel, bmodel

    def test_conv2D(self):
        image_dim_orders = ["tf", "th"]
        modes = ["valid", "same"]
        for order in image_dim_orders:
            keras.backend.set_image_dim_ordering(order)
            print("Testing with %s order" % keras.backend.image_dim_ordering())
            for mode in modes:
                print("Testing with mode %s" % mode)
                input_data = np.random.random_sample([1, 3, 128, 128])
                layer = Convolution2D(64, 1, 20,
                                      border_mode=mode,
                                      input_shape=(128, 128, 3))
                self.modelTestSingleLayer(input_data,
                                          layer, functional_api=True,
                                          dump_weights=True, rtol=1e-5, atol=1e-5)
                self.modelTestSingleLayer(input_data,
                                          layer, functional_api=False,
                                          dump_weights=True, rtol=1e-5, atol=1e-5)
        # Test if alias works or not
        layer = Conv2D(64, 3, 1,
                       border_mode="valid",
                       input_shape=(3, 128, 128))
        self.modelTestSingleLayer(input_data,
                                  layer, functional_api=True,
                                  dump_weights=True, rtol=1e-5, atol=1e-5)
        self.modelTestSingleLayer(input_data,
                                  layer, functional_api=False,
                                  dump_weights=True, rtol=1e-5, atol=1e-5)

    def test_maxpooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = MaxPooling2D(pool_size=[3, 3], strides=[2, 2],
                             border_mode="valid", input_shape=(3, 20, 20))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_maxpooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = MaxPooling1D(pool_length=2, stride=None, border_mode='valid', input_shape=(3, 20))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_globalmaxpooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = GlobalMaxPooling2D(input_shape=(3, 20, 20))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_globalmaxpooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = GlobalMaxPooling1D(input_shape=(3, 20))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_averagepooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = AveragePooling2D(pool_size=[3, 3], strides=[2, 2],
                                 border_mode="valid", input_shape=(3, 20, 20))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_averagepooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = AveragePooling1D(pool_length=2, stride=None,
                                 border_mode='valid', input_shape=(3, 20))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_globalaveragepooling2d(self):
        input_data = np.random.random_sample([1, 3, 20, 20])
        layer = GlobalAveragePooling2D(input_shape=(3, 20, 20))
        self.modelTestSingleLayer(input_data, layer, rtol=1e-6, atol=1e-6, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, rtol=1e-6, atol=1e-6, functional_api=False)

    def test_globalaveragepooling1d(self):
        input_data = np.random.random_sample([1, 3, 20])
        layer = GlobalAveragePooling1D(input_shape=(3, 20))  # TODO: add dim_ordering as parameter?
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_batchnormalization(self):
        # TODO: test training stage result, as the calc logic is not the same for mode 0
        image_dim_orders = ["tf", "th"]
        modes = ["valid", "same"]
        for order in image_dim_orders:
            keras.backend.set_image_dim_ordering(order)
            print("Testing with %s order" % keras.backend.image_dim_ordering())
            for mode in modes:
                print("Testing with mode %s" % mode)
                input_data = np.random.random_sample([2, 3, 20, 20])
                layer = BatchNormalization(input_shape=(3, 20, 20))
                self.modelTestSingleLayer(input_data, layer,
                                          dump_weights=True, functional_api=True)
                self.modelTestSingleLayer(input_data, layer,
                                          dump_weights=True, functional_api=False)

    def test_flatten(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = Flatten(input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_reshape(self):
        input_data = np.random.random_sample([1, 3, 5, 4])
        layer = Reshape(target_shape=(3, 20), input_shape=(3, 5, 4))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_repeatvector(self):
        input_data = np.random.random_sample([2, 3])
        layer = RepeatVector(4, input_shape=(3, ))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_merge_concat(self):
        # input_data1 = np.random.random_sample([2, 3, 5])
        # input_data2 = np.random.random_sample([2, 3, 6])
        # model1 = Sequential()
        # model1.add(Dense(20, input_dim=2))
        # model1.add(Dense(20, input_dim=2))
        #
        # model2 = Sequential()
        # model2.add(Input(input_dim=32))
        #
        # merged_model = Sequential()
        # merged_model.add(Merge([model1, model2], mode='concat', concat_axis=0))

        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 8))
        inputLayer3 = InputLayer(input_shape=(3, 6, 9))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='concat', concat_axis=3)
        # the index including batch and start from zero, and it's the index to be merge
        input_data = [np.random.random_sample([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 8]),
                      np.random.random([2, 3, 6, 9])]
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_merge_sum(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 7))
        inputLayer3 = InputLayer(input_shape=(3, 6, 7))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='sum')
        # the index including batch and start from zero, and it's the index to be merge
        input_data = [np.random.random_sample([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7])]
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_merge_mul(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 7))
        inputLayer3 = InputLayer(input_shape=(3, 6, 7))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='mul')
        input_data = [np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7])]
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_merge_max(self):
        inputLayer1 = InputLayer(input_shape=(3, 6, 7))
        inputLayer2 = InputLayer(input_shape=(3, 6, 7))
        inputLayer3 = InputLayer(input_shape=(3, 6, 7))

        layer = Merge([inputLayer1, inputLayer2, inputLayer3], mode='max')
        input_data = [np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7]),
                      np.random.random([2, 3, 6, 7])]
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_merge_dot(self):
        # use batch_input_shape for merge dot
        inputLayer1 = InputLayer(batch_input_shape=(2, 3))
        inputLayer2 = InputLayer(batch_input_shape=(2, 3))

        layer = Merge([inputLayer1, inputLayer2], mode='dot')
        input_data = [np.random.random([2, 3]),
                      np.random.random([2, 3])]
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_elu(self):
        input_data = np.random.random_sample([10, 2, 3, 4])
        layer = ELU(alpha=1.0, input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_prelu(self):
        input_data = np.random.random_sample([1, 2, 3, 4])
        layer = PReLU(input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_leakyrelu(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = LeakyReLU(alpha=0.5, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_parametricsoftplus(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = ParametricSoftplus(alpha_init=0.4, beta_init=2.5, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_thresholdedrelu(self):
        input_data = np.random.random_sample([1, 2, 3])
        layer = ThresholdedReLU(theta=0.2, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_zeropadding1d(self):
        input_data = np.random.uniform(0, 1, [3, 2, 3])
        layer1 = ZeroPadding1D(padding=3, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer1, functional_api=True)
        self.modelTestSingleLayer(input_data, layer1, functional_api=False)
        layer2 = ZeroPadding1D(padding=(2, 3), input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer2, functional_api=True)
        self.modelTestSingleLayer(input_data, layer2, functional_api=False)
        layer3 = ZeroPadding1D(padding={'left_pad': 1, 'right_pad': 2}, input_shape=(2, 3))
        self.modelTestSingleLayer(input_data, layer3, functional_api=True)
        self.modelTestSingleLayer(input_data, layer3, functional_api=False)

    def test_zeropadding2d(self):
        input_data = np.random.uniform(0, 1, [1, 2, 3, 4])
        layer1 = ZeroPadding2D(padding=(2, 3), input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer1, functional_api=True)
        self.modelTestSingleLayer(input_data, layer1, functional_api=False)
        layer2 = ZeroPadding2D(padding=(2, 3, 4, 1), input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer2, functional_api=True)
        self.modelTestSingleLayer(input_data, layer2, functional_api=False)
        layer3 = ZeroPadding2D(
            padding={'top_pad': 1, 'bottom_pad': 2, 'left_pad': 3, 'right_pad': 4},
            input_shape=(2, 3, 4))
        self.modelTestSingleLayer(input_data, layer3, functional_api=True)
        self.modelTestSingleLayer(input_data, layer3, functional_api=False)

    def test_zeropadding3d(self):
        input_data = np.random.uniform(0, 1, [3, 2, 4, 1, 5])
        layer = ZeroPadding3D(padding=(1, 2, 3), input_shape=(2, 4, 1, 5))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_cropping1d(self):
        input_data = np.random.uniform(0, 1, [3, 10, 10])
        layer = Cropping1D(cropping=(1, 2))
        self.modelTestSingleLayer(input_data, layer, functional_api=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False)

    def test_simplernn(self):
        input_data = np.random.random([3, 4, 5])
        layer = SimpleRNN(5, input_shape=(4, 5), return_sequences=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=True,
                                  dump_weights=True, rtol=1e-6, atol=1e-6)
        self.modelTestSingleLayer(input_data, layer, functional_api=False,
                                  dump_weights=True, rtol=1e-6, atol=1e-6)
        layer2 = SimpleRNN(3, input_shape=(4, 5), return_sequences=False)
        self.modelTestSingleLayer(input_data, layer2, functional_api=True,
                                  dump_weights=True, rtol=1e-6, atol=1e-6)
        self.modelTestSingleLayer(input_data, layer2, functional_api=False,
                                  dump_weights=True, rtol=1e-6, atol=1e-6)
        layer3 = SimpleRNN(3, input_shape=(4, 5), activation='relu')
        self.modelTestSingleLayer(input_data, layer3, functional_api=True,
                                  dump_weights=True, rtol=1e-6, atol=1e-6)
        self.modelTestSingleLayer(input_data, layer3, functional_api=False,
                                  dump_weights=True, rtol=1e-6, atol=1e-6)

    def test_lstm(self):
        input_data = np.random.random([3, 4, 5])
        layer = LSTM(5, input_shape=(4, 5), return_sequences=True, inner_activation='sigmoid')
        self.modelTestSingleLayer(input_data, layer, functional_api=True, dump_weights=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False, dump_weights=True)
        layer2 = LSTM(3, input_shape=(4, 5), return_sequences=False, inner_activation='sigmoid')
        self.modelTestSingleLayer(input_data, layer2, functional_api=True, dump_weights=True)
        self.modelTestSingleLayer(input_data, layer2, functional_api=False, dump_weights=True)

    def test_gru(self):
        input_data = np.random.random([3, 4, 5])
        layer = GRU(4, input_shape=(4, 5), return_sequences=True, inner_activation='sigmoid')
        self.modelTestSingleLayer(input_data, layer, functional_api=True, dump_weights=True)
        self.modelTestSingleLayer(input_data, layer, functional_api=False, dump_weights=True)
        layer2 = GRU(8, input_shape=(4, 5), return_sequences=False, inner_activation='sigmoid')
        self.modelTestSingleLayer(input_data, layer2, functional_api=True, dump_weights=True)
        self.modelTestSingleLayer(input_data, layer2, functional_api=False, dump_weights=True)

    # TODO: Support share weights training.
    def test_multiple_inputs_share_weights(self):
        with pytest.raises(Exception) as excinfo:
            input_node1 = Input(shape=[3, 16, 16])
            input_node2 = Input(shape=[3, 32, 32])
            conv2d = Convolution2D(5, 3, 3,
                                   border_mode='same')
            conv1 = conv2d(input_node1)
            conv2 = conv2d(input_node2)
            out1 = Flatten()(conv1)
            out2 = Flatten()(conv2)
            model1 = Model(input=[input_node1, input_node2], output=[out1, out2])
            tensor1, tensor2 = model1([input_node1, input_node2])
            out3 = Dense(7)(tensor1)
            out4 = Dense(8)(tensor2)
            model2 = Model(input=[input_node1, input_node2], output=[out3, out4])
            def_path, w_path = self._dump_keras(model2)
            bigdl_model = DefinitionLoader.from_json_path(def_path)
        assert str(excinfo.value) == """Convolution2D doesn't support multiple inputs with shared weights"""  # noqa

if __name__ == "__main__":
    pytest.main([__file__])
