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

import multiprocessing
import threading
from multiprocessing import Process
import unittest
import numpy as np
import pandas as pd
import os
import logging

from bigdl.ppml.fl.nn.fl_context import init_fl_context
from bigdl.ppml.fl.nn.tensorflow.utils import set_one_like_parameter
from bigdl.ppml.fl.nn.fl_server import FLServer
from bigdl.ppml.fl.estimator import Estimator
from bigdl.ppml.fl.utils import FLTest
from typing import List

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input

resource_path = os.path.join(os.path.dirname(__file__), "../../resources")


def mock_process(data_train, target, client_id, upload_server_model):
    init_fl_context(client_id, target)
    # set new_fl_client to True will create a FLClient with new ID for multi-party test
    df_train = pd.read_csv(os.path.join(resource_path, data_train))
    if 'Outcome' in df_train:
        df_x = df_train.drop('Outcome', 1)
        df_y = df_train['Outcome']
    else:
        df_x = df_train
        df_y = None
    df_x = df_x.drop('ID', 1)
    x = df_x.to_numpy(dtype="float32")
    y = np.expand_dims(df_y.to_numpy(dtype="float32"), axis=1) if df_y is not None else None

    model = build_client_model(4)
    set_one_like_parameter(model)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    server_model = build_server_model() if upload_server_model else None
    logging.info("Creating FL Tensorflow Estimator")
    ppl = Estimator.from_keras(client_model=model,
                               loss_fn=loss_fn,
                               optimizer_cls=tf.keras.optimizers.SGD,
                               optimizer_args={'learning_rate': 1e-3},
                               server_model=server_model)
    logging.info("Starting training")
    response = ppl.fit(x, y)
    result = ppl.predict(x)
    logging.info(response)
    return ppl


class TestLogisticRegression(FLTest):
    fmt = '%(asctime)s %(levelname)s {%(module)s:%(lineno)d} - %(message)s'
    logging.basicConfig(format=fmt, level=logging.DEBUG)

    @classmethod
    def setUpClass(cls) -> None:
        multiprocessing.set_start_method('spawn')

    def setUp(self) -> None:
        self.fl_server = FLServer(client_num=2)
        self.fl_server.set_port(self.port)
        self.fl_server.build()
        self.fl_server.start()

    def tearDown(self) -> None:
        self.fl_server.stop()

    def test_two_party_logistic_regression(self) -> None:
        df_train = pd.read_csv(
            os.path.join(resource_path, 'pima-indians-diabetes.csv'))

        df_x1 = df_train[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness']]
        df_x2 = df_train[['Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        df_y = df_train['Outcome']
        model = build_whole_model(feature_num1=4, feature_num2=4)
        set_one_like_parameter(model)
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        x1 = tf.convert_to_tensor(df_x1.to_numpy(dtype="float32"))
        x2 = tf.convert_to_tensor(df_x2.to_numpy(dtype="float32"))
        y = tf.convert_to_tensor(np.expand_dims(df_y.to_numpy(dtype="float32"), axis=1))
        assert len(x1) == len(x2) == len(y)
        batch_size = 4

        tensorflow_loss_list = []

        i, size = 0, len(x1)
        # just abandon last batch to reduce code
        while i + batch_size < len(x1):
            X1, X2, Y = x1[i:i + batch_size], x2[i:i + batch_size], y[i:i + batch_size]
            with tf.GradientTape() as tape:
                pred = model([X1, X2])
                loss = loss_fn(pred, Y)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            i += batch_size
            if i % 100 == 0:
                loss, current = loss, i
                print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")                
                tensorflow_loss_list.append(np.array(loss))


        mock_party2 = Process(target=mock_process,
            args=('diabetes-vfl-2.csv', self.target, 2, False))
        mock_party2.start()
        ppl = mock_process(data_train='diabetes-vfl-1.csv',
                           target=self.target,
                           client_id=1,
                           upload_server_model=True)
        mock_party2.join()
        assert np.allclose(tensorflow_loss_list, ppl.loss_history, 1e-1, 1e-1), \
            "Validation failed, correctness of PPML and native Tensorflow not the same"

def build_client_model(feature_num):
    inputs = Input(shape=(feature_num))
    outputs = Dense(1)(inputs)
    return Model(inputs=inputs, outputs=outputs, name="vfl_client_model")

def build_server_model():
    input1 = Input(shape=(1))
    input2 = Input(shape=(1))
    x = tf.stack([input1, input2])
    x = tf.reduce_sum(x, 0)  # above two act as interactive layer, CAddTable
    outputs = tf.sigmoid(x)
    return Model(inputs=[input1, input2], outputs=outputs, name="vfl_server_model")

def build_whole_model(feature_num1, feature_num2):
    input1 = Input(shape=(feature_num1))
    input2 = Input(shape=(feature_num2))
    dense1 = Dense(1)(input1)
    dense2 = Dense(1)(input2)
    x = tf.stack([dense1, dense2])
    x = tf.reduce_sum(x, 0)  # above two act as interactive layer, CAddTable
    outputs = tf.sigmoid(x)
    return Model(inputs=[input1, input2], outputs=outputs, name="tf_model")

class LogisticRegressionNetwork1(Model):
    def __init__(self) -> None:
        super().__init__()
        self.dense = Dense(1)

    def call(self, x):
        x = self.dense(x)
        return x


class LogisticRegressionNetwork2(Model):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = tf.nn.sigmoid
        self.build((1,))

        self.summary()

    def call(self, x: List[Tensor]):
        x = tf.stack(x)
        x = tf.reduce_sum(x, 0)  # above two act as interactive layer, CAddTable
        x = self.sigmoid(x)
        return x


class LogisticRegressionNetwork(Model):
    def __init__(self) -> None:
        super().__init__()
        self.dense1 = Dense(1)
        self.dense2 = Dense(1)
        self.sigmoid = tf.nn.sigmoid

    def call(self, x1, x2):
        x1 = self.dense1(x1)
        x2 = self.dense2(x2)
        x = tf.stack([x1, x2])
        x = tf.reduce_sum(x, 0)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    unittest.main()
