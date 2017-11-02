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
import numpy as np
from keras.models import Sequential, Model
from bigdl.keras1.converter import WeightLoader, WeightsConverter
np.random.seed(1337)  # for reproducibility
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers import Dense, Dropout, Input
from keras.models import model_from_json
from keras.optimizers import RMSprop
from bigdl.util.common import create_tmp_path
from bigdl.keras1.converter import DefinitionLoader
import numpy as np
from unittest import TestCase
import keras


class TestModels:

    @staticmethod
    def kmodel_graph_1_layer():
        input1 = Input(shape=(3,))
        dense = Dense(2)(input1)
        kmodel = Model(input=input1, output=dense)
        kmodel.compile(loss='categorical_crossentropy',
                       optimizer=RMSprop(),
                       metrics=['accuracy'])
        input_data = np.random.sample([1, 3])
        output_data = np.random.sample([1, 2])
        return kmodel, input_data, output_data

    @staticmethod
    def kmodel_graph_2_layer():
        input1 = Input(shape=(3,))
        dense = Dense(2)(input1)
        kmodel = Model(input=input1, output=dense)
        kmodel.compile(loss='categorical_crossentropy',
                       optimizer=RMSprop(),
                       metrics=['accuracy'])
        input_data = np.random.sample([1, 3])
        output_data = np.random.sample([1, 2])
        return kmodel, input_data, output_data

    @staticmethod
    def kmodel_graph_4_layer():
        input1 = Input(shape=(20,))
        dense = Dense(10)(input1)
        activation = Activation('relu')(dense)
        dense2 = Dense(10, activation='relu')(activation)
        dense3 = Dense(5)(dense2)
        # activation2 = Activation('softmax')(dense3)
        kmodel = Model(input=input1, output=dense3)
        kmodel.compile(loss='categorical_crossentropy',
                       optimizer=RMSprop(),
                       metrics=['accuracy'])
        input_data = np.random.sample([1, 20])
        output_data = np.random.sample([1, 5])
        return kmodel, input_data, output_data

    @staticmethod
    def kmodel_graph_5_layer():
        input1 = Input(shape=(20,))
        dense = Dense(10)(input1)
        activation = Activation('relu')(dense)
        dense2 = Dense(10, activation='relu')(activation)
        dense3 = Dense(5)(dense2)
        activation2 = Activation('softmax')(dense3)
        kmodel = Model(input=input1, output=activation2)
        kmodel.compile(loss='categorical_crossentropy',
                       optimizer='Adagrad',
                       metrics=['accuracy'])
        input_data = np.random.random([4, 20])
        output_data = np.random.random([4, 5])
        return kmodel, input_data, output_data

    @staticmethod
    def kmodel_seq_lenet_mnist():
        # assuming channel first
        input_shape = [1, 28, 28]
        b_input_shape = input_shape[:]
        nb_samples = 4
        b_input_shape.insert(0, nb_samples)
        nb_classes = 10

        # input image dimensions
        img_rows, img_cols = 28, 28
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='Adagrad',
                      metrics=['accuracy'])
        input_data = np.random.random_sample(b_input_shape)
        output_data = np.random.randint(1, 5, [nb_samples, 1])
        return model, input_data, output_data


class BigDLTestCase(TestCase):

    def __generate_model(self, input_data, output_layer):
        def without_batch(batch_shape):
            return batch_shape[1:]
        if isinstance(output_layer, keras.engine.Merge):  # it's a list in case of Merge Layer
            assert isinstance(input_data, list)
            input_tensor = [Input(shape=without_batch(i.shape)) for i in input_data]
        else:
            input_tensor = Input(shape=without_batch(input_data.shape))
        out_tensor = output_layer(input_tensor)
        return Model(input=input_tensor, output=out_tensor)

    def __generate_sequence(self, input_data, output_layer):
        seq = Sequential()
        if not isinstance(output_layer, keras.engine.Merge):
            seq.add(keras.layers.InputLayer(batch_input_shape=input_data.shape))
        seq.add(output_layer)
        return seq

    def _load_keras(self, json_path, hdf5_path):
        with open(json_path, "r") as jp:
            kmodel = model_from_json(jp.read())
        kmodel.load_weights(hdf5_path)
        bmodel = DefinitionLoader.from_json_path(json_path)
        WeightLoader.load_weights(bmodel, kmodel, hdf5_path)  # TODO: refactor reability of this api
        return kmodel, bmodel

    def _dump_keras(self, keras_model, dump_weights=False):
        keras_model_path = create_tmp_path()
        keras_model_json_path = keras_model_path + ".json"
        keras_model_hdf5_path = keras_model_path + ".hdf5"
        with open(keras_model_json_path, "w") as json_file:
            json_file.write(keras_model.to_json())
        print("json path: " + keras_model_json_path)
        if dump_weights:
            keras_model.save(keras_model_hdf5_path)
            print("hdf5 path: " + keras_model_hdf5_path)
        return keras_model_json_path, keras_model_hdf5_path

    def assert_allclose(self, a, b, rtol=1e-6, atol=1e-6, msg=None):
        # from tensorflow
        self.assertEqual(a.shape, b.shape, "Shape mismatch: expected %s, got %s." %
                         (a.shape, b.shape))
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            cond = np.logical_or(
                np.abs(a - b) > atol + rtol * np.abs(b), np.isnan(a) != np.isnan(b))
            if a.ndim:
                x = a[np.where(cond)]
                y = b[np.where(cond)]
                print("not close where = ", np.where(cond))
            else:
                # np.where is broken for scalars
                x, y = a, b
            print("not close lhs = ", x)
            print("not close rhs = ", y)
            print("not close dif = ", np.abs(x - y))
            print("not close tol = ", atol + rtol * np.abs(y))
            print("dtype = %s, shape = %s" % (a.dtype, a.shape))
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)

    def __generate_keras_model(self, functional_api, input_data, output_layer):
        if functional_api:
            keras_model = self.__generate_model(input_data, output_layer)
        else:
            keras_model = self.__generate_sequence(input_data, output_layer)
        return keras_model

    def modelTest(self,
                  input_data,
                  keras_model,
                  dump_weights=False,
                  is_training=False,
                  rtol=1e-7,
                  atol=1e-7):
        # weight_converter is a function keras [ndarray]-> bigdl [ndarray]
        keras_model_json_path, keras_model_hdf5_path = self._dump_keras(keras_model, dump_weights)
        bigdl_model = DefinitionLoader.from_json_path(keras_model_json_path)
        bigdl_model.training(is_training)
        bigdl_output = bigdl_model.forward(input_data)
        keras_output = keras_model.predict(input_data)
        # TODO: we should verify bigdl_output and keras_output here
        #  init result is not the same, so we disable the verification  for now
        # self.assert_allclose(bigdl_output,
        #                      keras_output,
        #                      rtol=rtol,
        #                      atol=atol)
        if dump_weights:  # load weights if possible
            WeightLoader.load_weights(bigdl_model, keras_model, keras_model_hdf5_path)
            bweights = bigdl_model.get_weights()
            bweights_from_keras = WeightsConverter.get_bigdl_weigths_from_keras(keras_model)

            # bweights and bweights_from_keras are all list
            assert isinstance(bweights, list)
            assert len(bweights) == len(bweights_from_keras)
            for i in range(len(bweights)):
                self.assert_allclose(bweights[i], bweights_from_keras[i], rtol, atol)

        bigdl_output2 = bigdl_model.forward(input_data)
        self.assert_allclose(bigdl_output2,
                             keras_output,
                             rtol=rtol,
                             atol=atol)

    def modelTestSingleLayer(self,
                             input_data,
                             output_layer,  # a keras layer
                             functional_api=True,
                             dump_weights=False,
                             is_training=False,
                             rtol=1e-7,
                             atol=1e-7):
        keras_model = self.__generate_keras_model(functional_api=functional_api,
                                                  input_data=input_data,
                                                  output_layer=output_layer)
        self.modelTest(input_data,
                       keras_model,
                       dump_weights=dump_weights,
                       is_training=is_training,
                       rtol=rtol,
                       atol=atol)
