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

from multiprocessing import Process
import unittest
import numpy as np
import os

from bigdl.ppml.fl import *
from bigdl.ppml.fl.estimator import Estimator
from bigdl.ppml.fl.nn.fl_server import FLServer
from bigdl.ppml.fl.nn.tensorflow.utils import set_one_like_parameter
from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.nn.tensorflow.estimator import TensorflowEstimator

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer
from tensorflow.keras import Model, Input

import shutil


resource_path = os.path.join(os.path.dirname(__file__), "../../resources")


class TestSaveLoad(FLTest):
    fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    server_model_path = '/tmp/vfl_server_model'
    client_model_path = '/tmp/vfl_client_model.h5'
    tf.config.run_functions_eagerly(True) # enable step-by-step debug
    tf.random.set_seed(22)
    def setUp(self) -> None:        
        self.fl_server = FLServer()
        self.fl_server.set_port(self.port)
        self.fl_server.build()
        self.fl_server.start()
    
    def tearDown(self) -> None:
        self.fl_server.stop()
        if os.path.exists(TestSaveLoad.server_model_path):
            shutil.rmtree(TestSaveLoad.server_model_path)
        if os.path.exists(TestSaveLoad.client_model_path):
            os.remove(TestSaveLoad.client_model_path)

    def test_mnist(self) -> None:
        """
        following code is copied from pytorch quick start
        link: https://www.tensorflow.org/tutorials/quickstart/advanced
        """
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Add a channels dimension
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")
        train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train[:5000], y_train[:5000])).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

        model = build_whole_model()
        set_one_like_parameter(model)
        
        
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        init_fl_context(1, self.target)
        vfl_model_1 = build_client_model()
        vfl_model_2 = build_server_model()
        vfl_client_ppl = Estimator.from_keras(client_model=vfl_model_1,
                                              loss_fn=loss_object,
                                              optimizer_cls=tf.keras.optimizers.SGD,
                                              optimizer_args={'lr':1e-3},
                                              server_model=vfl_model_2,
                                              server_model_path=TestSaveLoad.server_model_path,
                                              client_model_path=TestSaveLoad.client_model_path)
        
        vfl_client_ppl.fit(train_ds)
        self.fl_server.stop()
        self.setUp()
        client_model_loaded = tf.keras.models.load_model(TestSaveLoad.client_model_path)
        ppl_from_file = Estimator.from_keras(client_model=client_model_loaded,
                                             loss_fn=loss_object,
                                             optimizer_cls=tf.keras.optimizers.SGD,
                                             optimizer_args={'lr':1e-3})
        ppl_from_file.load_server_model(TestSaveLoad.server_model_path)
        ppl_from_file.fit(train_ds)
        assert ppl_from_file.loss_history[-1] < 2.10, \
            f"Validation failed, incremental training loss does not meet requirement, \
            required < 2.10, current {ppl_from_file.loss_history[-1]}"

def build_client_model():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, 3, activation='relu')(inputs)
    outputs = Flatten()(x)
    return Model(inputs=inputs, outputs=outputs, name="vfl_client_model")

def build_server_model():
    inputs = Input(shape=(21632))
    x = Dense(128, activation='relu')(inputs)
    outputs = Dense(10)(x)
    return Model(inputs=inputs, outputs=outputs, name="vfl_server_model")

def build_whole_model():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, 3, activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10)(x)
    return Model(inputs=inputs, outputs=outputs, name="vfl_whole_model")



if __name__ == '__main__':
    unittest.main()
