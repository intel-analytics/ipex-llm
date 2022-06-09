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


def get_linear_data(a=2, b=5, size=None):
    x = np.arange(0, 10, 10 / size, dtype=np.float32)
    y = a*x + b
    return x, y


def get_dataset(size, config):
    data = get_linear_data(size=size)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(config["batch_size"])
    return dataset


def get_train_val_data():
    data = get_linear_data(size=1000)
    validation_data = get_linear_data(size=400)
    return data, validation_data


def get_train_data_creator():
    def train_data_creator(config):
        return get_dataset(size=1000, config=config)
    return train_data_creator


def get_val_data_creator():
    def val_data_creator(config):
        return get_dataset(size=400, config=config)
    return val_data_creator


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
    data, validation_data = get_train_val_data()

    def test_fit_evaluate(self):
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        val_result = model.fit_eval(data=self.data,
                                    validation_data=self.validation_data,
                                    metric="mse",
                                    epochs=20)
        assert val_result.get("mse")

    def test_fit_eval_creator(self):
        data_creator = get_train_data_creator()
        validation_data_creator = get_val_data_creator()
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        val_result = model.fit_eval(data=data_creator,
                                    validation_data=validation_data_creator,
                                    metric="mse",
                                    epochs=20)
        assert val_result.get("mse")

    def test_fit_eval_default_metric(self):
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        val_result = model.fit_eval(data=self.data,
                                    validation_data=self.validation_data,
                                    epochs=20)
        hist_metric_name = tf.keras.metrics.get("mse").__name__
        assert val_result.get(hist_metric_name)

    def test_multiple_metrics_default(self):
        modelBuilder_keras = KerasModelBuilder(model_creator_multiple_metrics)
        model = modelBuilder_keras.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        with pytest.raises(RuntimeError):
            model.fit_eval(data=self.data,
                           validation_data=self.validation_data,
                           epochs=20)

    def test_uncompiled_model(self):
        def model_creator(config):
            """Returns a tf.keras model"""
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(1)
            ])
            return model

        modelBuilder_keras = KerasModelBuilder(model_creator)
        with pytest.raises(RuntimeError):
            model = modelBuilder_keras.build(config={
                "lr": 1e-2,
                "batch_size": 32,
            })
            model.fit_eval(data=self.data,
                           validation_data=self.validation_data,
                           metric="mse",
                           epochs=20)

    def test_unaligned_metric_value(self):
        modelBuilder_keras = KerasModelBuilder(model_creator_keras)
        model = modelBuilder_keras.build(config={
            "lr": 1e-2,
            "batch_size": 32,
        })
        with pytest.raises(RuntimeError):
            model.fit_eval(data=self.data,
                           validation_data=self.validation_data,
                           metric='mae',
                           epochs=20)


if __name__ == "__main__":
    pytest.main([__file__])
