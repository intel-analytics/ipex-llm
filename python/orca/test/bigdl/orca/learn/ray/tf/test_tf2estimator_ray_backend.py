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
import shutil
import tempfile

import pytest
from unittest import TestCase

import tensorflow as tf
import numpy as np

from bigdl.orca.data.pandas import read_csv
from bigdl.orca.data import SparkXShards, XShards
from bigdl.orca.data.tf.data import Dataset
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca import OrcaContext

NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 400

resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../resources")


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
        "optimizer": tf.keras.optimizers.Adam(lr),
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
    data_path = os.path.join(resource_path, "orca/learn/test_auto_shard/*.csv")
    dataset = tf.data.Dataset.list_files(data_path)
    dataset = dataset.interleave(lambda x: tf.data.TextLineDataset(x))
    dataset = dataset.map(lambda x: tf.strings.to_number(x))
    dataset = dataset.map(lambda x: (x, x))
    dataset = dataset.batch(batch_size)
    return dataset


def create_auto_shard_model(config):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.identity(x))
    ])

    return model


def create_auto_shard_compile_args(config):

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


def multi_output_model(config):
    image_input_1 = tf.keras.Input(shape=(32, 32, 3), name="input_1")
    image_input_2 = tf.keras.Input(shape=(32, 32, 3), name="input_2")

    x1 = tf.keras.layers.Conv2D(3, 3)(image_input_1)
    x1 = tf.keras.layers.GlobalMaxPooling2D()(x1)
    x2 = tf.keras.layers.Conv2D(3, 3)(image_input_2)
    x2 = tf.keras.layers.GlobalMaxPooling2D()(x2)
    x = tf.keras.layers.concatenate([x1, x2])

    score_output = tf.keras.layers.Dense(5, name="score_output")(x)
    class_output = tf.keras.layers.Dense(5, name="class_output")(x)

    model = tf.keras.Model(
        inputs=[image_input_1, image_input_2], outputs=[score_output, class_output]
    )
    return model

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


class TestTF2EstimatorRayBackend(TestCase):

    def test_orca_tf_dataset(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.parallelize(range(5))
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

        assert isinstance(train_stats, dict), "fit should return a dict"
        assert isinstance(end_eval_stats, dict), "evaluate should return a dict"
        assert orca_estimator.get_model(sample_input={"item": np.array([[1]], dtype=np.float32)})

        dloss = end_eval_stats["validation_loss"] - start_eval_stats["validation_loss"]
        dmse = (end_eval_stats["validation_mean_squared_error"] -
                start_eval_stats["validation_mean_squared_error"])
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

    def test_fit_and_evaluate_tf(self):
        ray_ctx = OrcaRayContext.get()
        batch_size = 32
        global_batch_size = batch_size * ray_ctx.num_ray_nodes

        trainer = Estimator.from_keras(model_creator=model_creator,
                                       verbose=True,
                                       config=None,
                                       backend="ray",
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

    def test_auto_shard_tf(self):
        # file 1 contains all 0s, file 2 contains all 1s
        # If shard by files, then each model will
        # see the same records in the same batch.
        # If shard by records, then each batch
        # will have different records.
        # The loss func is constructed such that
        # the former case will return 0, and the latter
        # case will return non-zero.

        ray_ctx = OrcaRayContext.get()
        trainer = Estimator.from_keras(
            model_creator=auto_shard_model_creator,
            verbose=True,
            backend="ray", workers_per_node=2)
        stats = trainer.fit(create_auto_shard_datasets, epochs=1, batch_size=4, steps_per_epoch=2)
        assert stats["loss"] == [0.0]

    def test_sparkxshards(self):

        train_data_shard = XShards.partition({"x": np.random.randn(100, 1),
                                              "y": np.random.randint(0, 1, size=(100))}).repartition(8)

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)

        stats = trainer.fit(train_data_shard, epochs=1, batch_size=4, steps_per_epoch=25)
        assert isinstance(stats, dict), "fit should return a dict"
        stats = trainer.evaluate(train_data_shard, batch_size=4, num_steps=25)
        assert isinstance(stats, dict), "evaluate should return a dict"

    def test_less_partitition_than_workers(self):

        train_data_shard = XShards.partition({"x": np.random.randn(100, 1),
                                              "y": np.random.randint(0, 1, size=(100))}).repartition(1)

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=4)

        trainer.fit(train_data_shard, epochs=1, batch_size=4, steps_per_epoch=25)
        trainer.evaluate(train_data_shard, batch_size=4, num_steps=25)
        trainer.predict(train_data_shard, batch_size=4).rdd.collect()
        trainer.shutdown()

        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100).repartition(1)
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                int(np.random.randint(0, 1, size=())))).toDF(["feature", "label"])
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
        trainer.shutdown()

    def test_dataframe(self):

        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100).repartition(9)
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float32)),
                                int(np.random.randint(0, 1, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.8
        }
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2)

        stats = trainer.fit(df, epochs=1, batch_size=4, steps_per_epoch=25,
                            feature_cols=["feature"],
                            label_cols=["label"])
        assert isinstance(stats, dict), "fit should return a dict"
        stats = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                 label_cols=["label"])
        assert isinstance(stats, dict), "evaluate should return a dict"
        trainer.predict(df, feature_cols=["feature"]).collect()

    def test_dataframe_decimal_input(self):

        from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
        from pyspark.sql.functions import col

        spark = OrcaContext.get_spark_session()
        schema = StructType([
            StructField("feature", FloatType(), True),
            StructField("label", IntegerType(), True)
        ])
        data = [(30.2222, 1), (40.0, 0), (15.1, 1),
                (-2.456, 1), (3.21, 0), (11.28, 1)]
        df = spark.createDataFrame(data=data, schema=schema)
        df = df.withColumn("feature", col("feature").cast("decimal(38,2)"))

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

    def test_dataframe_with_empty_partition(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 10)

        rdd_with_empty = rdd.repartition(4).\
            mapPartitionsWithIndex(lambda idx, part: [] if idx == 0 else part)

        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        from pyspark.ml.linalg import DenseVector
        df = rdd_with_empty.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float32)),
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
            input1 = tf.keras.layers.Input(shape=(1,))
            input2 = tf.keras.layers.Input(shape=(1,))
            concatenation = tf.concat([input1, input2], axis=-1)
            outputs = tf.keras.layers.Dense(units=1, activation='softmax')(concatenation)
            model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
            model.compile(**compile_args(config))
            return model

        file_path = os.path.join(resource_path, "orca/learn/ncf2.csv")
        train_data_shard = read_csv(file_path)

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
        
        result = trainer.predict(train_data_shard, feature_cols=["user", "item"])
        predictions = result.collect()[0]
        import pandas as pd
        assert isinstance(predictions, pd.DataFrame), "predict should return a pandas dataframe"
        assert isinstance(predictions["prediction"], pd.Series), \
               "predict dataframe should have a column named prediction"

    def test_dataframe_shard_size(self):
        OrcaContext._shard_size = 3
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 10)
        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float32)),
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
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(200, numSlices=1)
        assert rdd.getNumPartitions() == 1
        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float32)),
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
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(200, numSlices=10)
        val_rdd = sc.range(60, numSlices=8)
        from pyspark.sql import SparkSession
        spark = SparkSession(sc)
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float32)),
                                int(np.random.randint(0, 1, size=())))).toDF(["feature", "label"])
        val_df = val_rdd.map(lambda x: (DenseVector(np.random.randn(1,).astype(np.float32)),
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
        sc = OrcaContext.get_spark_context()
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

    def test_save_and_load_checkpoint(self):
        def model_creator(config):
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
            est.save_checkpoint("/tmp/cifar10_keras.ckpt")
            est.load_checkpoint("/tmp/cifar10_keras.ckpt")
            print("save success")
        finally:
            os.remove("/tmp/cifar10_keras.ckpt")
    
    def test_save_load_model_h5(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.2
        }

        try:
            temp_dir = tempfile.mkdtemp()

            trainer = Estimator.from_keras(
                model_creator=model_creator,
                verbose=True,
                config=config,
                workers_per_node=3,
                backend="ray")

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            print("start saving")
            trainer.save(os.path.join(temp_dir, "cifar10.h5"))

            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            before_res = trainer.predict(df, feature_cols=["feature"]).collect()
            expect_res = np.concatenate([part["prediction"] for part in before_res])

            trainer.load(os.path.join(temp_dir, "cifar10.h5"))
            
            # continous predicting
            after_res = trainer.predict(df, feature_cols=["feature"]).collect()
            pred_res = np.concatenate([part["prediction"] for part in after_res])

            assert np.array_equal(expect_res, pred_res)

            # continous training
            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_load_model_savemodel(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.2
        }

        try:
            temp_dir = tempfile.mkdtemp()

            trainer = Estimator.from_keras(
                model_creator=model_creator,
                verbose=True,
                config=config,
                workers_per_node=3,
                backend="ray")

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            print("start saving")
            trainer.save(os.path.join(temp_dir, "cifar10_savemodel"))

            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            before_res = trainer.predict(df, feature_cols=["feature"]).collect()
            expect_res = np.concatenate([part["prediction"] for part in before_res])

            trainer.load(os.path.join(temp_dir, "cifar10_savemodel"))
            
            # continous predicting
            after_res = trainer.predict(df, feature_cols=["feature"]).collect()
            pred_res = np.concatenate([part["prediction"] for part in after_res])

            assert np.array_equal(expect_res, pred_res)

            # continous training
            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_optional_model_creator(self):   
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.2
        }

        try:
            temp_dir = tempfile.mkdtemp()

            trainer = Estimator.from_keras(
                model_creator=model_creator,
                verbose=True,
                config=config,
                workers_per_node=3,
                backend="ray")

            trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                        feature_cols=["feature"],
                        label_cols=["label"],
                        validation_data=df,
                        validation_steps=1)

            trainer.save(os.path.join(temp_dir, "cifar10.h5"))

            before_res = trainer.predict(df, feature_cols=["feature"]).collect()
            expect_res = np.concatenate([part["prediction"] for part in before_res])
            trainer.shutdown()

            est = Estimator.from_keras(
                verbose=True,
                config=config,
                workers_per_node=3,
                backend="ray")

            est.load(os.path.join(temp_dir, "cifar10.h5"))
            est.save(os.path.join(temp_dir, "cifar10_option.h5"))

            # continous predicting
            after_res = est.predict(df, feature_cols=["feature"]).collect()
            pred_res = np.concatenate([part["prediction"] for part in after_res])

            assert np.array_equal(expect_res, pred_res)

            # continous training
            est.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                    feature_cols=["feature"],
                    label_cols=["label"],
                    validation_data=df,
                    validation_steps=1)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_load_model_weights_h5(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.2
        }

        try:
            temp_dir = tempfile.mkdtemp()

            trainer = Estimator.from_keras(
                model_creator=model_creator,
                verbose=True,
                config=config,
                workers_per_node=3,
                backend="ray")

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            print("start saving")
            trainer.save_weights(os.path.join(temp_dir,"model_weights.h5"))
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            before_res = trainer.predict(df, feature_cols=["feature"]).collect()
            expect_res = np.concatenate([part["prediction"] for part in before_res])
            
            trainer.load_weights(os.path.join(temp_dir,"model_weights.h5"))
            # continous predicting
            after_res = trainer.predict(df, feature_cols=["feature"]).collect()
            pred_res = np.concatenate([part["prediction"] for part in after_res])

            assert np.array_equal(expect_res, pred_res)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_load_model_weights_tf(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.2
        }

        try:
            temp_dir = tempfile.mkdtemp()

            trainer = Estimator.from_keras(
                model_creator=model_creator,
                verbose=True,
                config=config,
                workers_per_node=3,
                backend="ray")

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            print("start saving")
            trainer.save_weights(os.path.join(temp_dir,"model_weights"))
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            before_res = trainer.predict(df, feature_cols=["feature"]).collect()
            expect_res = np.concatenate([part["prediction"] for part in before_res])
            
            trainer.load_weights(os.path.join(temp_dir,"model_weights"))
            # continous predicting
            after_res = trainer.predict(df, feature_cols=["feature"]).collect()
            pred_res = np.concatenate([part["prediction"] for part in after_res])

            assert np.array_equal(expect_res, pred_res)
        finally:
            shutil.rmtree(temp_dir)

    def test_string_input(self):

        def model_creator(config):
            vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
                max_tokens=10, output_mode='int', output_sequence_length=4,
                vocabulary=["foo", "bar", "baz"])
            model = tf.keras.models.Sequential()
            model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
            model.add(vectorize_layer)
            return model

        from pyspark.sql.types import StructType, StructField, StringType
        spark = OrcaContext.get_spark_session()
        schema = StructType([StructField("input", StringType(), True)])
        input_data = [["foo qux bar"], ["qux baz"]]
        input_df = spark.createDataFrame(input_data, schema)
        estimator = Estimator.from_keras(model_creator=model_creator)
        output_df = estimator.predict(input_df, batch_size=1, feature_cols=["input"])
        output = output_df.collect()
        print(output)

    def test_array_string_input(self):

        def model_creator(config):
            model = tf.keras.models.Sequential([
                tf.keras.Input(shape=(None,), dtype=tf.string),
                tf.keras.layers.StringLookup(
                    vocabulary=config["vocabulary"]
                )
            ])
            return model

        import itertools
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
        spark = OrcaContext.get_spark_session()
        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("input", ArrayType(StringType(), True), True)
        ])
        input_data = [(0, ["foo", "qux", "bar"]), (1, ["qux", "baz", "baz"])]
        input_df = spark.createDataFrame(input_data, schema)
        string_data = [row["input"] for row in input_df.select("input").distinct().collect()]
        vocabulary = list(set(itertools.chain(*string_data)))
        config = {"vocabulary": vocabulary}
        estimator = Estimator.from_keras(model_creator=model_creator, config=config)
        output_df = estimator.predict(input_df, batch_size=1, feature_cols=["input"])
        output = output_df.collect()
        print(output)
        
    def test_tensorboard(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.2
        }

        try:
            temp_dir = tempfile.mkdtemp()

            trainer = Estimator.from_keras(
                model_creator=model_creator,
                verbose=True,
                config=config,
                workers_per_node=2,
                backend="ray")

            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=os.path.join(temp_dir, "train_log"),
                                               update_freq='epoch')
            ]
            res = trainer.fit(df, epochs=3, batch_size=4, steps_per_epoch=25,
                              callbacks=callbacks,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)
            assert len(os.listdir(os.path.join(temp_dir, "train_log"))) > 0

            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=os.path.join(temp_dir, "train_log_2"),
                                               update_freq='batch')
            ]
            res = trainer.fit(df, epochs=3, batch_size=4, steps_per_epoch=25,
                              callbacks=callbacks,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=11)
            assert len(os.listdir(os.path.join(temp_dir, "train_log_2"))) > 0

            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=os.path.join(temp_dir, "val_log"),
                                               update_freq='batch')
            ]
            res = trainer.evaluate(df, batch_size=4, num_steps=25,
                                   callbacks=callbacks,
                                   feature_cols=["feature"],
                                   label_cols=["label"])
            assert len(os.listdir(os.path.join(temp_dir, "val_log"))) > 0

        finally:
            shutil.rmtree(temp_dir)

    def test_get_model(self):   
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        config = {
            "lr": 0.2
        }

        try:
            temp_dir = tempfile.mkdtemp()

            trainer = Estimator.from_keras(
                model_creator=model_creator,
                verbose=True,
                config=config,
                workers_per_node=3,
                backend="ray")

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            trainer.save(os.path.join(temp_dir, "cifar10.h5"))
            pre_model_weights = trainer.get_model().get_weights()

            trainer.shutdown()

            est = Estimator.from_keras(
                verbose=True,
                config=config,
                workers_per_node=3,
                backend="ray")

            est.load(os.path.join(temp_dir, "cifar10.h5"))
            after_model_weights = est.get_model().get_weights()

            for pre_tensor, after_tensor in list(zip(pre_model_weights, after_model_weights)):
                assert np.allclose(pre_tensor, after_tensor)
        finally:
            shutil.rmtree(temp_dir)

    def test_multi_output_predict(self):
        from pyspark.sql.types import FloatType, ArrayType
        from pyspark.sql.functions import udf

        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        df = rdd.map(lambda x:[x,
                               np.random.rand(3072).tolist(),
                               np.random.rand(3072).tolist()]).toDF(["index",
                                                                     "input_1",
                                                                     "input_2"])

        def reshape(x):
            return np.array(x).reshape([32, 32, 3]).tolist()

        reshape_udf = udf(reshape, ArrayType(ArrayType(ArrayType(FloatType()))))

        df = df.withColumn("input_1", reshape_udf(df.input_1))
        df = df.withColumn("input_2", reshape_udf(df.input_2))

        def model_creator(config):
            model = multi_output_model(config)
            model.compile(
                optimizer=tf.keras.optimizers.RMSprop(config["lr"]),
                loss=[tf.keras.losses.MeanSquaredError(),
                      tf.keras.losses.CategoricalCrossentropy()],
            )
            return model

        estimator = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config={"lr": 0.2},
            workers_per_node=2,
            backend="ray")

        pred_res = estimator.predict(df,
                                     feature_cols=["input_1", "input_2"],
                                     output_cols=["score_output", "class_output"])
        pred_res.collect()
        assert "score_output" and "class_output" in pred_res.columns

        # output_cols is None
        pred_df = estimator.predict(df,
                                    feature_cols=["input_1", "input_2"])
        pred_df.collect()
        assert "prediction" in pred_df.columns


if __name__ == "__main__":
    pytest.main([__file__])
