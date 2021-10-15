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
from unittest import TestCase
from bigdl.orca.automl.model.base_keras_model import KerasBaseModel, KerasModelBuilder
import numpy as np
import tensorflow as tf
import pytest


def get_data():
    def get_linear_data(a, b, size):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        y = a*x + b
        return x, y
    train_x, train_y = get_linear_data(2, 5, 1000)
    val_x, val_y = get_linear_data(2, 5, 400)
    data = {'x': train_x, 'y': train_y, 'val_x': val_x, 'val_y': val_y}
    return data


def model_creator_keras(config):
    """Returns a tf.keras model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss="mse",
                  optimizer='sgd',
                  metrics=["mse"])
    return model


def model_creator_multiple_metrics(config):
    """Returns a tf.keras model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss="mse",
                  optimizer='sgd',
                  metrics=["mse", "mae"])
    return model


class TestBaseKerasModel(TestCase):
    data = get_data()

    def test_fit_evaluate(self):
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        val_result = model.fit_eval(data=(self.data["x"], self.data["y"]),
                                    validation_data=(self.data["val_x"], self.data["val_y"]),
                                    metric="mse",
                                    epochs=20)
        assert val_result.get("mse")

    def test_fit_eval_default_metric(self):
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        val_result = model.fit_eval(data=(self.data["x"], self.data["y"]),
                                    validation_data=(self.data["val_x"], self.data["val_y"]),
                                    epochs=20)
        hist_metric_name = tf.keras.metrics.get("mse").__name__
        assert val_result.get(hist_metric_name)

    def test_multiple_metrics_default(self):
        modelBuilder_keras = KerasModelBuilder(model_creator_multiple_metrics)
        model = modelBuilder_keras.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        with pytest.raises(ValueError):
            model.fit_eval(data=(self.data["x"], self.data["y"]),
                           validation_data=(self.data["val_x"], self.data["val_y"]),
                           epochs=20)

    def test_uncompiled_model(self):
        def model_creator(config):
            """Returns a tf.keras model"""
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(1)
            ])
            return model

        modelBuilder_keras = KerasModelBuilder(model_creator)
        with pytest.raises(ValueError):
            model = modelBuilder_keras.build(config={
                "lr": 1e-2,
                "batch_size": 32,
            })
            model.fit_eval(data=(self.data["x"], self.data["y"]),
                           validation_data=(self.data["val_x"], self.data["val_y"]),
                           metric="mse",
                           epochs=20)

    def test_unaligned_metric_value(self):
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        with pytest.raises(ValueError):
            model.fit_eval(data=(self.data["x"], self.data["y"]),
                           validation_data=(self.data["val_x"], self.data["val_y"]),
                           metric='mae',
                           epochs=20)


if __name__ == "__main__":
    pytest.main([__file__])
