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
np.random.seed(1337)  # for reproducibility
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
from bigdl.util.common import *
from bigdl.keras.converter import *
from bigdl.keras.converter import WeightLoader, WeightsConverter
import numpy as np
from unittest import TestCase
import keras
from bigdl.examples.keras.keras_utils import *


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
    def kmodel_graph_activation_is_layer():
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
    def kmodel_seq_lenet_mnist():
        # Part of the code is from keras example
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

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        keras.backend.set_image_dim_ordering("th")
        sparkConf = create_spark_conf().setMaster("local[4]").setAppName("test model")
        self.sc = get_spark_context(sparkConf)
        self.sqlContext = SQLContext(self.sc)
        init_engine()

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        keras.backend.set_image_dim_ordering("th")
        self.sc.stop()

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
        kmodel.load_weights_from_hdf5(hdf5_path)
        bmodel = DefinitionLoader.from_json_path(json_path)
        WeightLoader.load_weights_from_hdf5(bmodel, kmodel, hdf5_path)
        return kmodel, bmodel

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

    def __generate_random_weights(self, weights):
        # weights: a list of ndarray; weights from keras
        # Randomly generate a new list with the same shape to serve as the new weights for testing
        new_weights = []
        for weight in weights:
            new_weights.append(np.random.random(list(weight.shape)))
        return new_weights

    def modelTest(self,
                  input_data,
                  keras_model,
                  random_weights=True,
                  dump_weights=False,
                  is_training=False,
                  rtol=1e-6,
                  atol=1e-6):
        if random_weights:
            # Randomly generate weights instead of using initial weights
            kweights = keras_model.get_weights()
            new_kweights = self.__generate_random_weights(kweights)
            keras_model.set_weights(new_kweights)
        # weight_converter is a function keras [ndarray]-> bigdl [ndarray]
        keras_model_json_path, keras_model_hdf5_path = dump_keras(keras_model,
                                                                  dump_weights=dump_weights)

        # Use Theano backend to load as a bigdl model
        self.__set_keras_backend("theano")
        bigdl_model = DefinitionLoader.from_json_path(keras_model_json_path)
        bigdl_model.training(is_training)
        bigdl_output = bigdl_model.forward(input_data)

        # Use TensorFlow backend to compare results
        self.__set_keras_backend("tensorflow")
        keras_output = keras_model.predict(input_data)
        # TODO: we should verify bigdl_output and keras_output here
        #  init result is not the same, so we disable the verification  for now
        # self.assert_allclose(bigdl_output,
        #                      keras_output,
        #                      rtol=rtol,
        #                      atol=atol)
        if dump_weights:  # load weights if possible
            WeightLoader.load_weights_from_hdf5(bigdl_model, keras_model, keras_model_hdf5_path)

        bigdl_output2 = bigdl_model.forward(input_data)
        self.assert_allclose(bigdl_output2,
                             keras_output,
                             rtol=rtol,
                             atol=atol)

    def modelTestSingleLayerWithOrdersModes(self,
                                            input_data,
                                            output_layer_creator,  # a keras layer
                                            dim_orderings=["tf", "th"],
                                            border_modes=["valid", "same"],
                                            random_weights=True,
                                            dump_weights=False,
                                            is_training=False,
                                            rtol=1e-6,
                                            atol=1e-6):
        for dim_ordering in dim_orderings:
            print("Testing with dim_ordering %s" % dim_ordering)
            keras.backend.set_image_dim_ordering(dim_ordering)
            for border_mode in border_modes:
                print("Testing with border_mode %s" % border_mode)
                output_layer = output_layer_creator()
                if not hasattr(output_layer, "dim_ordering") and "1D" not in output_layer.__class__.__name__:  # noqa
                    raise Exception("cannot set dim order for %s" % output_layer)
                output_layer.dim_ordering = dim_ordering
                if hasattr(output_layer, "border_mode"):
                    output_layer.border_mode = border_mode
                elif border_mode is not None:
                    raise Exception("cannot set border_mode for %s" % output_layer)
                self.modelTestSingleLayer(input_data,
                                          output_layer,  # a keras layer
                                          random_weights,
                                          dump_weights,
                                          is_training,
                                          rtol,
                                          atol)

    def modelTestSingleLayer(self,
                             input_data,
                             output_layer,  # a keras layer
                             random_weights=True,
                             dump_weights=False,
                             is_training=False,
                             rtol=1e-6,
                             atol=1e-6,
                             functional_apis=[True, False]):
        for api in functional_apis:
            self._do_modelTestSingleLayer(
                input_data,
                output_layer,  # a keras layer
                functional_api=api,
                random_weights=random_weights,
                dump_weights=dump_weights,
                is_training=is_training,
                rtol=rtol,
                atol=atol)

    def _do_modelTestSingleLayer(self,
                                 input_data,
                                 output_layer,  # a keras layer
                                 functional_api=True,
                                 random_weights=True,
                                 dump_weights=False,
                                 is_training=False,
                                 rtol=1e-6,
                                 atol=1e-6):
        self.__set_keras_backend("tensorflow")
        keras_model = self.__generate_keras_model(functional_api=functional_api,
                                                  input_data=input_data,
                                                  output_layer=output_layer)
        self.modelTest(input_data,
                       keras_model,
                       random_weights=random_weights,
                       dump_weights=dump_weights,
                       is_training=is_training,
                       rtol=rtol,
                       atol=atol)

    # Set Keras Backend without affecting the current dim_ordering
    def __set_keras_backend(self, backend):
        if K.backend() != backend:
            current_dim = K.image_dim_ordering()
            os.environ['KERAS_BACKEND'] = backend
            from six.moves import reload_module
            reload_module(K)
            K.set_image_dim_ordering(current_dim)
            assert K.backend() == backend
            assert K.image_dim_ordering() == current_dim
        # Make theano backend compatible with Python3
        if backend == "theano":
            from theano import ifelse

    def compare_loss(self, y_a, y_b, kloss, bloss, rtol=1e-6, atol=1e-6):
        # y_a: input/y_pred; y_b: target/y_true
        keras_output = np.mean(K.eval(kloss(K.variable(y_b), K.variable(y_a))))
        bigdl_output = bloss.forward(y_a, y_b)
        np.testing.assert_allclose(bigdl_output, keras_output, rtol=rtol, atol=atol)

    # Compare forward results with Keras for new Keras-like API layers.
    def compare_newapi(self, klayer, blayer, input_data, weight_converter=None,
                       is_training=False, rtol=1e-6, atol=1e-6):
        from keras.models import Sequential as KSequential
        from bigdl.nn.keras.topology import Sequential as BSequential
        bmodel = BSequential()
        bmodel.add(blayer)
        kmodel = KSequential()
        kmodel.add(klayer)
        koutput = kmodel.predict(input_data)
        from bigdl.nn.keras.layer import BatchNormalization
        if isinstance(blayer, BatchNormalization):
            k_running_mean = K.eval(klayer.running_mean)
            k_running_std = K.eval(klayer.running_std)
            blayer.set_running_mean(k_running_mean)
            blayer.set_running_std(k_running_std)
        if kmodel.get_weights():
            bmodel.set_weights(weight_converter(klayer, kmodel.get_weights()))
        bmodel.training(is_training)
        boutput = bmodel.forward(input_data)
        self.assert_allclose(boutput, koutput, rtol=rtol, atol=atol)

    def compare_model(self, bmodel, kmodel, input_data, rtol=1e-5, atol=1e-5):
        WeightLoader.load_weights_from_kmodel(bmodel, kmodel)
        bmodel.training(is_training=False)
        bigdl_output = bmodel.forward(input_data)
        keras_output = kmodel.predict(input_data)
        self.assert_allclose(bigdl_output, keras_output, rtol=rtol, atol=atol)
