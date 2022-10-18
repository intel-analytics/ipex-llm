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
import math
import os

import pytest
from unittest import TestCase

import tensorflow as tf
import numpy as np

from bigdl.orca.data import SparkXShards
from bigdl.orca.data.tf.data import Dataset
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext


class SimpleModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.simple_model = tf.keras.Sequential([
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(1)])

    def call(self, inputs):
        return self.simple_model(inputs["item"])

    def train_step(self, data):
        y = data["label"]

        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        y = data["label"]

        y_pred = self(data, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


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


def simple_model(config):
    import tensorflow as tf
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=(1,)),
                                        tf.keras.layers.Dense(1)])
    return model


def model_creator(config):
    model = simple_model(config)
    model.compile(**compile_args(config))
    return model


def model_creator_for_orca_dataset(config):
    model = SimpleModel()
    model.compile(**compile_args(config))
    return model


def get_estimator(workers_per_node=2, model_fn=model_creator):
    estimator = Estimator.from_keras(model_creator=model_fn,
                                     config={"lr": 1e-3},
                                     workers_per_node=workers_per_node,
                                     backend="ray")
    return estimator


class TestTF2EstimatorTF2Backend(TestCase):
    def setUp(self):
        self.sc = init_orca_context(init_ray_on_spark=True)

    def tearDown(self):
        stop_orca_context()

    def test_orca_tf_dataset(self):
        rdd = self.sc.parallelize(range(5))
        shard = SparkXShards(rdd)

        def generate_dataset(df, a, b, size):
            items = np.array([[i / size] for i in range(size)], dtype=np.float32)
            labels = np.array([a * x[0] + b for x in items], dtype=np.float32)
            return {"item": items, "label": labels}

        train_shard = shard.transform_shard(generate_dataset, 5, 10, 1000)
        test_shard = shard.transform_shard(generate_dataset, 5, 10, 100)

        train_dataset = Dataset.from_tensor_slices(train_shard)
        test_dataset = Dataset.from_tensor_slices(test_shard)
        train_step = math.ceil(5 * 1000 / 32)
        test_step = math.ceil(5 * 100 / 32)

        orca_estimator = get_estimator(workers_per_node=2, model_fn=model_creator_for_orca_dataset)

        start_eval_stats = orca_estimator.evaluate(data=test_dataset,
                                                   num_steps=test_step, batch_size=32)

        train_stats = orca_estimator.fit(data=train_dataset,
                                         epochs=2, batch_size=32, steps_per_epoch=train_step)
        print(train_stats)

        end_eval_stats = orca_estimator.evaluate(data=test_dataset,
                                                 num_steps=test_step, batch_size=32)

        assert orca_estimator.get_model(sample_input={"item": np.array([[1]], dtype=np.float32)})

        dloss = end_eval_stats[0]["validation_loss"] - start_eval_stats[0]["validation_loss"]
        dmse = (end_eval_stats[0]["validation_mean_squared_error"] -
                start_eval_stats[0]["validation_mean_squared_error"])
        print(f"dLoss: {dloss}, dMSE: {dmse}")
        assert dloss < 0 and dmse < 0, "training sanity check failed. loss increased!"

        pred_shards = orca_estimator.predict(test_dataset)
        pred1 = pred_shards.collect()

        path = "/tmp/model_test_orca_dataset.ckpt"
        try:
            orca_estimator.save_checkpoint(path)
            orca_estimator.shutdown()
            est = get_estimator(model_fn=model_creator_for_orca_dataset)
            with self.assertRaises(Exception) as context:
                est.load_checkpoint(path)
            self.assertTrue('Failed to set model weights, please provide real tensor data'
                            in str(context.exception))
            est.load_checkpoint(path, sample_input={"item": np.array([[1]], dtype=np.float32)})
            result_shards = est.predict(test_dataset)
            pred2 = result_shards.collect()
        finally:
            os.remove(path)

        assert np.allclose(pred1[0]["prediction"], pred2[0]["prediction"])


if __name__ == "__main__":
    pytest.main([__file__])