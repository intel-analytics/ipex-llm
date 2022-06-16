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
import shutil
import tempfile
from unittest import TestCase
import time

import time
import numpy as np
import pytest
import tensorflow as tf

from bigdl.orca.data import SparkXShards
from bigdl.orca.data.tf.data import Dataset
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca import OrcaContext

import os

resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../../resources")


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


def simple_model(config):
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


class TestTFEstimator(TestCase):
    # todo need more test cases
    def test_dataframe(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
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
                backend="spark",
                model_dir=temp_dir)

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            print("start saving")
            trainer.save_weights(os.path.join(temp_dir, "cifar10_keras.h5"))
            trainer.load_weights(os.path.join(temp_dir, "cifar10_keras.h5"))
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            res = trainer.predict(df, feature_cols=["feature"]).collect()
            print("predict result: ", res)
        finally:
            shutil.rmtree(temp_dir)

    def test_dataframe_with_empty_partition(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)

        rdd_with_empty = rdd.repartition(4). \
            mapPartitionsWithIndex(lambda idx, part: [] if idx == 0 else part)

        spark = OrcaContext.get_spark_session()
        from pyspark.ml.linalg import DenseVector
        df = rdd_with_empty.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
                                           int(np.random.randint(0, 2, size=())))) \
            .toDF(["feature", "label"])

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
                backend="spark",
                model_dir=temp_dir)

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"]
                              )

            print("start saving")
            trainer.save_weights(os.path.join(temp_dir, "cifar10_keras.h5"))
            trainer.load_weights(os.path.join(temp_dir, "cifar10_keras.h5"))
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            res = trainer.predict(df, feature_cols=["feature"]).collect()
            print("predict result: ", res)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_xshards_pandas_dataframe(self):
        from bigdl.orca.data.pandas import read_csv
        sc = OrcaContext.get_spark_context()

        def model_creator(config):
            import tensorflow as tf
            input1 = tf.keras.layers.Input(shape=(1,))
            input2 = tf.keras.layers.Input(shape=(1,))
            concatenation = tf.concat([input1, input2], axis=-1)
            outputs = tf.keras.layers.Dense(units=1, activation='softmax')(concatenation)
            model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
            model.compile(**compile_args(config))
            return model
        
        resource_path = os.path.join(os.path.split(__file__)[0], "../../../resources")
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")

        xshards = read_csv(file_path, usecols=[0, 1, 2], dtype={0: np.float32, 1: np.float32,
                                                                2: np.float32})
        
        config = {
            "lr": 0.2
        }
        
        try:
            temp_dir = tempfile.mkdtemp()

            trainer = Estimator.from_keras(
                model_creator=model_creator,
                verbose=True,
                config=config,
                workers_per_node=1,
                backend="spark",
                model_dir=temp_dir)

            res = trainer.fit(data=xshards, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["user", "item"], label_cols=["label"])

            print("start saving")
            trainer.save_weights(os.path.join(temp_dir, "cifar10_keras.h5"))
            trainer.load_weights(os.path.join(temp_dir, "cifar10_keras.h5"))
            res = trainer.evaluate(data=xshards, num_steps=25, batch_size=4,
                                   feature_cols=["user", "item"], label_cols=["label"])
            print("validation result: ", res)
        finally:
            shutil.rmtree(temp_dir)

    def test_checkpoint_weights(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
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
                backend="spark",
                model_dir=temp_dir)

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(temp_dir, "ckpt_{epoch}"),
                                                   save_weights_only=True)
            ]

            res = trainer.fit(df, epochs=3, batch_size=4, steps_per_epoch=25,
                              callbacks=callbacks,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)
            latest_checkpoint = Estimator.latest_checkpoint(temp_dir)
            trainer.load_weights(latest_checkpoint)
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            res = trainer.predict(df, feature_cols=["feature"]).collect()
            print("predict result: ", res)
        finally:
            shutil.rmtree(temp_dir)

    def test_checkpoint_weights_h5(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
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
                backend="spark",
                model_dir=temp_dir)

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(temp_dir, "ckpt_weights.h5"),
                                                   save_weights_only=True)
            ]

            res = trainer.fit(df, epochs=3, batch_size=4, steps_per_epoch=25,
                              callbacks=callbacks,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)
            trainer.load_weights(os.path.join(temp_dir, "ckpt_weights.h5"))
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            res = trainer.predict(df, feature_cols=["feature"]).collect()
            print("predict result: ", res)
        finally:
            shutil.rmtree(temp_dir)


    def test_dataframe_shard_size(self):
        sc = OrcaContext.get_spark_context()
        OrcaContext._shard_size = 3
        rdd = sc.range(0, 100, numSlices=10)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        val_rdd = sc.range(0, 20, numSlices=6)
        val_df = val_rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
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
                backend="spark",
                model_dir=temp_dir)

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              validation_data=val_df,
                              validation_steps=2,
                              feature_cols=["feature"],
                              label_cols=["label"])

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"])

            res = trainer.evaluate(val_df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            res = trainer.predict(df, feature_cols=["feature"]).collect()
            print("predict result: ", res)
        finally:
            shutil.rmtree(temp_dir)
        OrcaContext._shard_size = None

    def test_dataframe_different_train_val(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100, numSlices=10)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        val_rdd = sc.range(0, 20, numSlices=6)
        val_df = val_rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
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
                backend="spark",
                model_dir=temp_dir)

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              validation_data=val_df,
                              validation_steps=2,
                              feature_cols=["feature"],
                              label_cols=["label"])

            res = trainer.evaluate(val_df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            res = trainer.predict(df, feature_cols=["feature"]).collect()
            print("predict result: ", res)
        finally:
            shutil.rmtree(temp_dir)

    def test_tensorboard(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
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
                backend="spark",
                model_dir=temp_dir)

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

    def test_checkpoint_model(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
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
                backend="spark",
                model_dir=temp_dir)

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(temp_dir, "ckpt_{epoch}"),
                                                   save_weights_only=False
                                                   )
            ]

            res = trainer.fit(df, epochs=3, batch_size=4, steps_per_epoch=25,
                              callbacks=callbacks,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1
                              )
            assert len(os.listdir(os.path.join(temp_dir, "ckpt_3"))) > 0

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(temp_dir, "best"),
                                                   save_weights_only=False,
                                                   save_best_only=True
                                                   )
            ]

            res = trainer.fit(df, epochs=3, batch_size=4, steps_per_epoch=25,
                              callbacks=callbacks,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1
                              )
            assert len(os.listdir(os.path.join(temp_dir, "best"))) > 0
        finally:
            shutil.rmtree(temp_dir)

    def test_save_load_model(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
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
                backend="spark",
                model_dir=temp_dir)

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            print("start saving")
            trainer.save(os.path.join(temp_dir, "a.h5"))
            trainer.load(os.path.join(temp_dir, "a.h5"))
            trainer.save(os.path.join(temp_dir, "saved_model"))
            trainer.load(os.path.join(temp_dir, "saved_model"))
            # continous training
            res = trainer.fit(df, epochs=10, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1,
                              initial_epoch=5)
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            res = trainer.predict(df, feature_cols=["feature"]).collect()
            print("predict result: ", res)
        finally:
            shutil.rmtree(temp_dir)

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

        def compile_args(config):
            if config is None:
                lr = 1e-3
            else:
                lr = config["lr"]
            args = {
                "optimizer": tf.keras.optimizers.SGD(1e-3),
                "loss": "mean_squared_error",
                "metrics": ["mean_squared_error"]}
            return args

        def model_creator(config):
            model = SimpleModel()
            model.compile(**compile_args(config))
            return model

        orca_estimator = Estimator.from_keras(model_creator=model_creator,
                                              config={"lr": 1e-3},
                                              workers_per_node=2,
                                              backend="tf2")

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

        path = "/tmp/model.ckpt"
        try:
            orca_estimator.save(path)
            orca_estimator.shutdown()
            est = Estimator.from_keras(model_creator=model_creator,
                                       config={"lr": 1e-3},
                                       workers_per_node=2,
                                       backend="tf2")
            with self.assertRaises(Exception) as context:
                est.load(path)
            self.assertTrue('Failed to set model weights, please provide real tensor data'
                            in str(context.exception))
            est.load(path, sample_input={"item": np.array([[1]], dtype=np.float32)})
            result_shards = est.predict(test_dataset)
            pred2 = result_shards.collect()
        finally:
            os.remove(path)

        assert np.allclose(pred1[0]["prediction"], pred2[0]["prediction"])


if __name__ == "__main__":
    pytest.main([__file__])
