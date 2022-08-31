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
from unittest import TestCase

import ray
from ray.data import Dataset
import tensorflow as tf
import numpy as np

from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca import init_orca_context, stop_orca_context

def train_data_creator(a=5, b=10, size=1000):
    def get_dataset(a, b, size) -> Dataset:
        items = [i / size for i in range(size)]
        dataset = ray.data.from_items([{
            "x": x,
            "y": a * x + b
        } for x in items])
        return dataset
    train_dataset = get_dataset(a, b, size)
    return train_dataset

def val_data_creator(a=5, b=10, size=100):
    def get_dataset(a, b, size) -> Dataset:
        items = [i / size for i in range(size)]
        dataset = ray.data.from_items([{
            "x": x,
            "y": a * x + b
        } for x in items])
        return dataset
    val_dataset = get_dataset(a, b, size)
    return val_dataset

def simple_model(config):
    model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1, )),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(1)])
    return model

def compile_args(config):
    if config is None:
        lr = 1e-3
    else:
        lr = config["lr"]
    args = {
        "optimizer": tf.keras.optimizers.SGD(lr),
        "loss": "mean_squared_error",
        "metrics": ["mean_squared_error"]}
    return args

def model_creator(config):
    model = simple_model(config)
    model.compile(**compile_args(config))
    return model

def get_estimator(workers_per_node=2, model_fn=model_creator):
    estimator = Estimator.from_keras(model_creator=model_fn,
                                     config={"lr": 1e-3},
                                     workers_per_node=workers_per_node,
                                     backend="ray")
    return estimator

class TestTF2Estimator(TestCase):
    def setUp(self):
        init_orca_context(runtime="ray", address="localhost:6379")

    def tearDown(self):
        stop_orca_context()

    def test_train_and_evaluate(self):
        orca_estimator = get_estimator(workers_per_node=2)
        train_dataset = train_data_creator()
        validation_dataset = val_data_creator()

        data_config_args = {"output_signature":
                            (tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                             tf.TensorSpec(shape=(None), dtype=tf.float32))
                            }

        start_eval_stats = orca_estimator.evaluate(data=validation_dataset,
                                                   num_steps=2, batch_size=32,
                                                   label_cols="x", feature_cols=["y"],
                                                   data_config=data_config_args)

        train_stats = orca_estimator.fit(data=train_dataset,
                                         epochs=2, batch_size=32,
                                         label_cols="x", feature_cols=["y"],
                                         data_config=data_config_args)
        print(train_stats)

        end_eval_stats = orca_estimator.evaluate(data=validation_dataset,
                                                 num_steps=2, batch_size=32,
                                                 label_cols="x", feature_cols=["y"],
                                                 data_config=data_config_args)

        assert orca_estimator.get_model()

        dloss = end_eval_stats["validation_loss"] - start_eval_stats["validation_loss"]
        dmse = (end_eval_stats["validation_mean_squared_error"] -
                start_eval_stats["validation_mean_squared_error"])
        print(f"dLoss: {dloss}, dMSE: {dmse}")
        assert dloss < 0 and dmse < 0, "training sanity check failed. loss increased!"

if __name__ == "__main__":
    pytest.main([__file__])
