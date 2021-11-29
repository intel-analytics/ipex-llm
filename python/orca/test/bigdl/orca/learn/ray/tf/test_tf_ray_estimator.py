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

from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.data import XShards

import bigdl.orca.data.pandas
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca.ray import RayContext
import ray

NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 400

import os

resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../resources")


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


class TestTFRayEstimator(TestCase):
    def impl_test_fit_and_evaluate(self, backend):
        import tensorflow as tf
        ray_ctx = RayContext.get()
        batch_size = 32
        global_batch_size = batch_size * ray_ctx.num_ray_nodes

        if backend == "horovod":
            trainer = Estimator.from_keras(
                model_creator=simple_model,
                compile_args_creator=compile_args,
                verbose=True,
                config=None,
                backend=backend)
        else:

            trainer = Estimator.from_keras(model_creator=model_creator,
                                           verbose=True,
                                           config=None,
                                           backend=backend,
                                           workers_per_node=2)

        # model baseline performance
        start_stats = trainer.evaluate(create_test_dataset, batch_size=global_batch_size,
                                       num_steps=NUM_TEST_SAMPLES // global_batch_size)
        print(start_stats)

        def scheduler(epoch):
            if epoch < 2:
                return 0.001
            else:
                return 0.001 * tf.math.exp(0.1 * (2 - epoch))

        scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        # train for 2 epochs
        trainer.fit(create_train_datasets, epochs=2, batch_size=global_batch_size,
                    steps_per_epoch=10, callbacks=[scheduler])
        trainer.fit(create_train_datasets, epochs=2, batch_size=global_batch_size,
                    steps_per_epoch=10, callbacks=[scheduler])

        # model performance after training (should improve)
        end_stats = trainer.evaluate(create_test_dataset, batch_size=global_batch_size,
                                     num_steps=NUM_TEST_SAMPLES // global_batch_size)
        print(end_stats)

        # sanity check that training worked
        dloss = end_stats["validation_loss"] - start_stats["validation_loss"]
        dmse = (end_stats["validation_mean_squared_error"] -
                start_stats["validation_mean_squared_error"])
        print(f"dLoss: {dloss}, dMSE: {dmse}")

        assert dloss < 0 and dmse < 0, "training sanity check failed. loss increased!"

    def test_fit_and_evaluate_tf(self):
        self.impl_test_fit_and_evaluate(backend="tf2")

    def test_fit_and_evaluate_horovod(self):
        self.impl_test_fit_and_evaluate(backend="horovod")

    def test_auto_shard_tf(self):
        # file 1 contains all 0s, file 2 contains all 1s
        # If shard by files, then each model will
        # see the same records in the same batch.
        # If shard by records, then each batch
        # will have different records.
        # The loss func is constructed such that
        # the former case will return 0, and the latter
        # case will return non-zero.

        ray_ctx = RayContext.get()
        trainer = Estimator.from_keras(
            model_creator=auto_shard_model_creator,
            verbose=True,
            backend="tf2", workers_per_node=2)
        stats = trainer.fit(create_auto_shard_datasets, epochs=1, batch_size=4, steps_per_epoch=2)
        assert stats["train_loss"] == 0.0

    def test_auto_shard_horovod(self):
        # file 1 contains all 0s, file 2 contains all 1s
        # If shard by files, then each model will
        # see the same records in the same batch.
        # If shard by records, then each batch
        # will have different records.
        # The loss func is constructed such that
        # the former case will return 0, and the latter
        # case will return non-zero.

        ray_ctx = RayContext.get()
        trainer = Estimator.from_keras(
            model_creator=create_auto_shard_model,
            compile_args_creator=create_auto_shard_compile_args,
            verbose=True,
            backend="horovod", workers_per_node=2)
        stats = trainer.fit(create_auto_shard_datasets, epochs=1, batch_size=4, steps_per_epoch=2)
        assert stats["train_loss"] == 0.0

    # this needs horovod >= 0.19.2
    def test_horovod_learning_rate_schedule(self):
        import horovod
        major, minor, patch = horovod.__version__.split(".")

        larger_major = int(major) > 0
        larger_minor = int(major) == 0 and int(minor) > 19
        larger_patch = int(major) == 0 and int(minor) == 19 and int(patch) >= 2

        if larger_major or larger_minor or larger_patch:
            ray_ctx = RayContext.get()
            batch_size = 32
            workers_per_node = 4
            global_batch_size = batch_size * workers_per_node
            config = {
                "lr": 0.8
            }
            trainer = Estimator.from_keras(
                model_creator=simple_model,
                compile_args_creator=compile_args,
                verbose=True,
                config=config,
                backend="horovod", workers_per_node=workers_per_node)
            import horovod.tensorflow.keras as hvd
            callbacks = [
                hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, initial_lr=0.4,
                                                         verbose=True),
                hvd.callbacks.LearningRateScheduleCallback(start_epoch=5, end_epoch=10,
                                                           multiplier=1., initial_lr=0.4),
                hvd.callbacks.LearningRateScheduleCallback(start_epoch=10, end_epoch=15,
                                                           multiplier=1e-1, initial_lr=0.4),
                hvd.callbacks.LearningRateScheduleCallback(start_epoch=15, end_epoch=20,
                                                           multiplier=1e-2, initial_lr=0.4),
                hvd.callbacks.LearningRateScheduleCallback(start_epoch=20, multiplier=1e-3,
                                                           initial_lr=0.4),
                LRChecker()
            ]
            for i in range(30):
                trainer.fit(create_train_datasets, epochs=1, batch_size=global_batch_size,
                            callbacks=callbacks)
        else:
            # skip tests in horovod lower version
            pass

    def test_sparkxshards(self):

        train_data_shard = XShards.partition({"x": np.random.randn(100, 1),
                                              "y": np.random.randint(0, 1, size=(100))})

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)

        trainer.fit(train_data_shard, epochs=1, batch_size=4, steps_per_epoch=25)
        trainer.evaluate(train_data_shard, batch_size=4, num_steps=25)

    def test_dataframe(self):

        sc = init_nncontext()
        rdd = sc.range(0, 10)
        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float)),
                                int(np.random.randint(0, 1, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)

        trainer.fit(df, epochs=1, batch_size=4, steps_per_epoch=25,
                    feature_cols=["feature"],
                    label_cols=["label"])
        trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                         label_cols=["label"])
        trainer.predict(df, feature_cols=["feature"]).collect()

    def test_dataframe_with_empty_partition(self):
        from bigdl.orca import OrcaContext
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 10)

        rdd_with_empty = rdd.repartition(4).\
            mapPartitionsWithIndex(lambda idx, part: [] if idx == 0 else part)

        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        from pyspark.ml.linalg import DenseVector
        df = rdd_with_empty.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float)),
                                           int(np.random.randint(0, 1, size=()))))\
            .toDF(["feature", "label"])

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)

        trainer.fit(df, epochs=1, batch_size=4, steps_per_epoch=25,
                    feature_cols=["feature"],
                    label_cols=["label"])
        trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                         label_cols=["label"])
        trainer.predict(df, feature_cols=["feature"]).collect()

    def test_pandas_dataframe(self):
        def model_creator(config):
            import tensorflow as tf
            input1 = tf.keras.layers.Input(shape=(1,))
            input2 = tf.keras.layers.Input(shape=(1,))
            concatenation = tf.concat([input1, input2], axis=-1)
            outputs = tf.keras.layers.Dense(units=1, activation='softmax')(concatenation)
            model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
            model.compile(**compile_args(config))
            return model

        file_path = os.path.join(resource_path, "orca/learn/ncf2.csv")
        train_data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        config = {
            "lr": 0.8
        }

        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=1)

        trainer.fit(train_data_shard, epochs=1, batch_size=4, steps_per_epoch=25,
                    feature_cols=["user", "item"],
                    label_cols=["label"])
        trainer.evaluate(train_data_shard, batch_size=4, num_steps=25,
                         feature_cols=["user", "item"], label_cols=["label"])
        trainer.predict(train_data_shard, feature_cols=["user", "item"]).collect()

    def test_dataframe_shard_size(self):
        from bigdl.orca import OrcaContext
        OrcaContext._shard_size = 3
        sc = init_nncontext()
        rdd = sc.range(0, 10)
        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float)),
                                int(np.random.randint(0, 1, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)

        trainer.fit(df, epochs=1, batch_size=4, steps_per_epoch=25,
                    feature_cols=["feature"],
                    label_cols=["label"])
        trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                         label_cols=["label"])
        trainer.predict(df, feature_cols=["feature"]).collect()
        OrcaContext._shard_size = None

    def test_partition_num_less_than_workers(self):
        sc = init_nncontext()
        rdd = sc.range(200, numSlices=1)
        assert rdd.getNumPartitions() == 1
        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float)),
                                int(np.random.randint(0, 1, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)
        assert df.rdd.getNumPartitions() < trainer.num_workers

        trainer.fit(df, epochs=1, batch_size=4, steps_per_epoch=25,
                    validation_data=df, validation_steps=1,
                    feature_cols=["feature"],
                    label_cols=["label"])
        trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                         label_cols=["label"])
        trainer.predict(df, feature_cols=["feature"]).collect()

    def test_num_part_data_diff_val_data(self):
        sc = init_nncontext()
        rdd = sc.range(200, numSlices=10)
        val_rdd = sc.range(60, numSlices=8)
        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float)),
                                int(np.random.randint(0, 1, size=())))).toDF(["feature", "label"])
        val_df = val_rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float)),
                                        int(np.random.randint(0, 1, size=()))))\
            .toDF(["feature", "label"])

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)
        assert df.rdd.getNumPartitions() > trainer.num_workers
        assert df.rdd.getNumPartitions() != val_df.rdd.getNumPartitions()

        trainer.fit(df, epochs=1, batch_size=4, steps_per_epoch=25,
                    validation_data=val_df, validation_steps=1,
                    feature_cols=["feature"],
                    label_cols=["label"])

    def test_dataframe_predict(self):
        sc = init_nncontext()
        rdd = sc.parallelize(range(20))
        df = rdd.map(lambda x: ([float(x)] * 5,
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])

        estimator = Estimator.from_keras(
            model_creator=identity_model_creator,
            verbose=True,
            config={},
            workers_per_node=2)
        result = estimator.predict(df, batch_size=4,
                                   feature_cols=["feature"])
        expr = "sum(cast(feature <> to_array(prediction) as int)) as error"
        assert result.selectExpr(expr).first()["error"] == 0

    def test_sparkxshards_with_inbalanced_data(self):

        train_data_shard = XShards.partition({"x": np.random.randn(100, 1),
                                              "y": np.random.randint(0, 1, size=(100))})

        def random_pad(data):
            import numpy as np
            import random
            times = random.randint(1, 10)
            data["x"] = np.concatenate([data["x"]] * times)
            data["y"] = np.concatenate([data["y"]] * times)
            return data

        train_data_shard = train_data_shard.transform_shard(random_pad)

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)

        trainer.fit(train_data_shard, epochs=1, batch_size=4, steps_per_epoch=25)
        trainer.evaluate(train_data_shard, batch_size=4, num_steps=25)

    def test_predict_xshards(self):
        train_data_shard = XShards.partition({"x": np.random.randn(100, 1),
                                              "y": np.random.randint(0, 1, size=(100,))})
        expected = train_data_shard.collect()

        expected = [shard["x"] for shard in expected]

        for x in expected:
            print(x.shape)

        expected = np.concatenate(expected)

        config = {
        }
        trainer = Estimator.from_keras(
            model_creator=identity_model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)

        result_shards = trainer.predict(train_data_shard, batch_size=10).collect()

        result = [shard["prediction"] for shard in result_shards]
        expected_result = [shard["x"] for shard in result_shards]

        result = np.concatenate(result)

        assert np.allclose(expected, result)

    def test_save_and_load(self):
        def model_creator(config):
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                       padding='valid'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                       padding='valid'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation='softmax')]
            )
            model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            return model

        def train_data_creator(config, batch_size):
            dataset = tf.data.Dataset.from_tensor_slices((np.random.randn(100, 28, 28, 3),
                                                          np.random.randint(0, 10, (100, 1))))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(1000)
            dataset = dataset.batch(batch_size)
            return dataset

        batch_size = 320
        try:
            est = Estimator.from_keras(model_creator=model_creator, workers_per_node=2)

            history = est.fit(train_data_creator,
                              epochs=1,
                              batch_size=batch_size,
                              steps_per_epoch=5)
            print("start saving")
            est.save("/tmp/cifar10_keras.ckpt")
            est.load("/tmp/cifar10_keras.ckpt")
            print("save success")
        finally:
            os.remove("/tmp/cifar10_keras.ckpt")

    def test_string_input(self):

        def model_creator(config):
            import tensorflow as tf
            vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
                max_tokens=10, output_mode='int', output_sequence_length=4)
            model = tf.keras.models.Sequential()
            model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
            model.add(vectorize_layer)
            return model

        from bigdl.orca import OrcaContext
        from pyspark.sql.types import StructType, StructField, StringType
        spark = OrcaContext.get_spark_session()
        schema = StructType([StructField("input", StringType(), True)])
        input_data = [["foo qux bar"], ["qux baz"]]
        input_df = spark.createDataFrame(input_data, schema)
        estimator = Estimator.from_keras(model_creator=model_creator)
        output_df = estimator.predict(input_df, batch_size=1, feature_cols=["input"])
        output = output_df.collect()
        print(output)


if __name__ == "__main__":
    pytest.main([__file__])
