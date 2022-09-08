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




resource_path = os.path.join(os.path.dirname(__file__), "../../resources")


class TestCorrectness(FLTest):
    fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    tf.config.run_functions_eagerly(True) # enable step-by-step debug
    def setUp(self) -> None:
        self.fl_server = FLServer()
        self.fl_server.set_port(self.port)
        self.fl_server.build()
        self.fl_server.start()
    
    def tearDown(self) -> None:
        self.fl_server.stop()

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
        optimizer = tf.keras.optimizers.Adam()
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)
            return loss

        @tf.function
        def test_step(images, labels):
            # training=False is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)

            test_loss(t_loss)
            test_accuracy(labels, predictions)
        
        tensorflow_loss_history = []
        EPOCHS = 1
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
            size = len(train_ds)
            for batch, (images, labels) in enumerate(train_ds):
                loss = train_step(images, labels)                
                if batch % 10 == 0:
                    tensorflow_loss_history.append(np.array(loss))
                    logging.info(f"loss: {loss:>7f}  [{batch:>5d}/{size:>5d}]  \
                            epoch {epoch}/{EPOCHS}")

            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)
            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result()}, '
                f'Accuracy: {train_accuracy.result() * 100}, '
                f'Test Loss: {test_loss.result()}, '
                f'Test Accuracy: {test_accuracy.result() * 100}'
            )
        
        # TODO: set fixed parameters
        init_fl_context(1, self.target)
        vfl_model_1 = build_client_model()
        set_one_like_parameter(vfl_model_1)
        vfl_model_2 = build_server_model()
        set_one_like_parameter(vfl_model_2)
        vfl_client_ppl = Estimator.from_keras(client_model=vfl_model_1,
                                              loss_fn=loss_object,
                                              optimizer_cls=tf.keras.optimizers.Adam,
                                              optimizer_args={},
                                              server_model=vfl_model_2)
        
        vfl_client_ppl.fit(train_ds)
        assert np.allclose(tensorflow_loss_history, vfl_client_ppl.loss_history), \
            "Validation failed, correctness of PPML and native Pytorch not the same"

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

class NeuralNetwork(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

class NeuralNetworkPart1(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        return x

class NeuralNetworkPart2(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)
    
    def call(self, x):
        x = x[0]
        x = self.d1(x)
        return self.d2(x)


if __name__ == '__main__':
    unittest.main()
