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
import os
import pytest
from unittest import TestCase

import numpy as np
import tensorflow as tf

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator

NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 400

def linear_dataset(a=2, size=1000):
    x = np.random.rand(size)
    y = x / 2

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    return x, y

def create_train_datasets(config, batch_size):
    x_train, y_train = linear_dataset(size=NUM_TRAIN_SAMPLES)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).batch(
        batch_size)
    
    return train_dataset

def create_test_dataset(config, batch_size):
    x_test, y_test = linear_dataset(size=NUM_TEST_SAMPLES)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    return test_dataset

def simple_model(config):
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=(1,)),
                                        tf.keras.layers.Dense(1)])
    return model

def compile_args(config):
    if "lr" in config:
        lr = config["lr"]
    else:
        lr = 1e-3
    args = {
        "optimizer": tf.keras.optimizers.SGD(lr),
        "loss": "mean_squared_error",
        "metrics": ["mean_squared_error"]
    }
    return args

def model_creator(config):
    model = simple_model(config)
    model.compile(**compile_args(config))
    return model

class TestPytorchEstimator(TestCase):
    def setUp(self):
        init_orca_context(runtime="ray", address="localhost:6379")

    def tearDown(self):
        stop_orca_context()
    
    def test_train(self):
        estimator = Estimator.from_keras(model_creator=model_creator,
                                        verbose=True,
                                        config=None,
                                        backend="tf2",
                                        workers_per_node=2)

        start_stats = estimator.evaluate(create_test_dataset, batch_size=32)
        print(start_stats)

        train_stats = estimator.fit(create_train_datasets, epochs=1, batch_size=32)
        print("This is Train Results:", train_stats)

        end_stats = estimator.evaluate(create_test_dataset, batch_size=32)
        print("This is Val Results:", end_stats)
        assert estimator.get_model()

        dloss = end_stats["validation_loss"] - start_stats["validation_loss"]
        dmse = (end_stats["validation_mean_squared_error"] -
                start_stats["validation_mean_squared_error"])
        print(f"dLoss: {dloss}, dMSE: {dmse}")

        assert dloss < 0 and dmse < 0, "training sanity check failed. loss increased!"


if __name__ == "__main__":
    pytest.main([__file__])
