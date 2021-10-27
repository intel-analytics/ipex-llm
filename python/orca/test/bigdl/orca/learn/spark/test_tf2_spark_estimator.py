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
import shutil
from unittest import TestCase

import numpy as np
import pytest
import tensorflow as tf

from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca import OrcaContext

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


def create_train_datasets(config, batch_size):
    import tensorflow as tf
    x_train, y_train = linear_dataset(size=NUM_TRAIN_SAMPLES)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLES).batch(
        batch_size)

    return train_dataset


def create_test_dataset(config, batch_size):
    import tensorflow as tf
    x_test, y_test = linear_dataset(size=NUM_TEST_SAMPLES)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    return test_dataset


def simple_model(config):
    import tensorflow as tf
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=(1,)),
                                        tf.keras.layers.Dense(1)])
    return model


def compile_args(config):
    import tensorflow as tf
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


def identity_model_creator(config):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1)),
        tf.keras.layers.Lambda(lambda x: tf.identity(x))
    ])
    model.compile()
    return model


def create_auto_shard_datasets(config, batch_size):
    import tensorflow as tf
    data_path = os.path.join(resource_path, "orca/learn/test_auto_shard/*.csv")
    dataset = tf.data.Dataset.list_files(data_path)
    dataset = dataset.interleave(lambda x: tf.data.TextLineDataset(x))
    dataset = dataset.map(lambda x: tf.strings.to_number(x))
    dataset = dataset.map(lambda x: (x, x))
    dataset = dataset.batch(batch_size)
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


def auto_shard_model_creator(config):
    model = create_auto_shard_model(config)
    model.compile(**create_auto_shard_compile_args(config))
    return model


class LRChecker(tf.keras.callbacks.Callback):
    def __init__(self, *args):
        super(LRChecker, self).__init__(*args)

        self.warmup_lr = [0.16, 0.22, 0.28, 0.34, 0.4]

    def on_epoch_end(self, epoch, logs=None):
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        print("epoch {} current lr is {}".format(epoch, current_lr))
        if epoch < 5:
            assert abs(current_lr - self.warmup_lr[epoch]) < 1e-5
        elif 5 <= epoch < 10:
            assert abs(current_lr - 0.4) < 1e-5
        elif 10 <= epoch < 15:
            assert abs(current_lr - 0.04) < 1e-5
        elif 15 <= epoch < 20:
            assert abs(current_lr - 0.004) < 1e-5
        else:
            assert abs(current_lr - 0.0004) < 1e-5


class TestTFEstimator(TestCase):

    def test_dataframe(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.2
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2,
            backend="spark")

        res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                          feature_cols=["feature"],
                          label_cols=["label"],
                          validation_data=df,
                          validation_steps=1)

        print("start saving")
        try:
            trainer.save_weights("/tmp/cifar10_keras.h5")
            trainer.load_weights("/tmp/cifar10_keras.h5")
            trainer.save("/tmp/a.model")
            trainer.load("/tmp/a.model")


            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                         label_cols=["label"])
            trainer.predict(df, feature_cols=["feature"]).collect()
        finally:
            os.remove("/tmp/cifar10_keras.h5")
            shutil.rmtree("/tmp/a.model")


if __name__ == "__main__":
    pytest.main([__file__])
