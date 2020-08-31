#
# Copyright 2018 Analytics Zoo Authors.
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

import numpy as np

from zoo.orca.learn.tf2 import Estimator
from zoo.ray import RayContext

NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 400

import os

resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../../resources")


def linear_dataset(a=2, size=1000):
    x = np.random.rand(size)
    y = x / 2

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    return x, y


def create_train_datasets(config):
    import tensorflow as tf
    batch_size = config["batch_size"]
    x_train, y_train = linear_dataset(size=NUM_TRAIN_SAMPLES)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).batch(
        batch_size)

    return train_dataset


def create_test_dataset(config):
    import tensorflow as tf
    batch_size = config["batch_size"]
    x_test, y_test = linear_dataset(size=NUM_TEST_SAMPLES)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    return test_dataset


def create_auto_shard_datasets(config):
    import tensorflow as tf
    data_path = os.path.join(resource_path, "orca/learn/test_auto_shard/*.csv")
    dataset = tf.data.Dataset.list_files(data_path)
    dataset = dataset.interleave(lambda x: tf.data.TextLineDataset(x))
    dataset = dataset.map(lambda x: tf.strings.to_number(x))
    dataset = dataset.map(lambda x: (x, x))
    dataset = dataset.batch(2)
    return dataset


def create_auto_shard_model(config):
    import tensorflow as tf
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.identity(x))
    ])

    return model


def create_auto_shard_compile_args(config):
    import tensorflow as tf

    def loss_func(y1, y2):
        return tf.abs(y1[0] - y1[1]) + tf.abs(y2[0] - y2[1])

    args = {
        "optimizer": tf.keras.optimizers.SGD(lr=0.0),
        "loss": loss_func,
    }
    return args


def simple_model(config):
    import tensorflow as tf
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=(1,)),
                                        tf.keras.layers.Dense(1)])
    return model


def compile_args(config):
    import tensorflow as tf
    args = {
        "optimizer": tf.keras.optimizers.Adam(),
        "loss": "mean_squared_error",
        "metrics": ["mean_squared_error"]
    }
    return args


class TestTFRayEstimator(TestCase):
    def impl_test_fit_and_evaluate(self, backend):
        import tensorflow as tf
        ray_ctx = RayContext.get()
        batch_size = 32
        global_batch_size = batch_size * ray_ctx.num_ray_nodes
        config = {
            "batch_size": batch_size
        }
        trainer = Estimator(
            model_creator=simple_model,
            compile_args_creator=compile_args,
            verbose=True,
            config=config,
            backend=backend)

        # model baseline performance
        start_stats = trainer.evaluate(create_test_dataset,
                                       steps=NUM_TEST_SAMPLES // global_batch_size)
        print(start_stats)

        def scheduler(epoch):
            if epoch < 2:
                return 0.001
            else:
                return 0.001 * tf.math.exp(0.1 * (2 - epoch))

        scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        # train for 2 epochs
        trainer.fit(create_train_datasets, epochs=2, callbacks=[scheduler])
        trainer.fit(create_train_datasets, epochs=2, callbacks=[scheduler])

        # model performance after training (should improve)
        end_stats = trainer.evaluate(create_test_dataset,
                                     steps=NUM_TEST_SAMPLES // global_batch_size)
        print(end_stats)

        # sanity check that training worked
        dloss = end_stats["validation_loss"] - start_stats["validation_loss"]
        dmse = (end_stats["validation_mean_squared_error"] -
                start_stats["validation_mean_squared_error"])
        print(f"dLoss: {dloss}, dMSE: {dmse}")

        assert dloss < 0 and dmse < 0, "training sanity check failed. loss increased!"

    def test_fit_and_evaluate_tf(self):
        self.impl_test_fit_and_evaluate(backend="tf")

    def test_fit_and_evaluate_horovod(self):
        self.impl_test_fit_and_evaluate(backend="horovod")

    def impl_test_auto_shard(self, backend):

        # file 1 contains all 0s, file 2 contains all 1s
        # If shard by files, then each model will
        # see the same records in the same batch.
        # If shard by records, then each batch
        # will have different records.
        # The loss func is constructed such that
        # the former case will return 0, and the latter
        # case will return non-zero.

        ray_ctx = RayContext.get()
        trainer = Estimator(
            model_creator=create_auto_shard_model,
            compile_args_creator=create_auto_shard_compile_args,
            verbose=True,
            config={},
            backend=backend, workers_per_node=2)
        stats = trainer.fit(create_auto_shard_datasets, epochs=1, steps_per_epoch=2)
        assert stats["train_loss"] == 0.0

    # this test has known issue and will be fixed by following pr
    # def test_auto_shard_tf(self):
    #     self.impl_test_auto_shard("tf")

    def test_auto_shard_horovod(self):
        self.impl_test_auto_shard("horovod")
