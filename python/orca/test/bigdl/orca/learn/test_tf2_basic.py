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
import shutil
import tempfile
from unittest import TestCase

import numpy as np
import tensorflow as tf

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca import OrcaContext


def simple_model(config):
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=(1,)),
                                        tf.keras.layers.Dense(1)])
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


class TestTFEstimatorBasic(TestCase):
    
    def setUp(self):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        conf = {"spark.python.worker.reuse": "false"}
        sc = init_orca_context(cores=8)

    def tearDown(self):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        stop_orca_context()

    def test_dataframe(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        data = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                  int(np.random.randint(0, 2, size=())))).collect()
        df = spark.createDataFrame(data=data, schema=["feature", "label"])

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
                backend="spark")

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            assert isinstance(res, dict), "fit should return a dict"
            print("start saving")
            trainer.save_weights(os.path.join(temp_dir, "cifar10_keras.h5"))
            trainer.load_weights(os.path.join(temp_dir, "cifar10_keras.h5"))
            df = spark.createDataFrame(data=data, schema=["feature", "label"])
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            assert isinstance(res, dict), "evaluate should return a dict"
            print("validation result: ", res)

            res = trainer.predict(df, feature_cols=["feature"]).collect()
            print("predict result: ", res)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_checkpoint_weights(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        data = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                  int(np.random.randint(0, 2, size=())))).collect()
        df = spark.createDataFrame(data=data, schema=["feature", "label"])

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
                backend="spark")

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

    def test_dataframe_shard_size(self):
        sc = OrcaContext.get_spark_context()
        OrcaContext._shard_size = 3
        rdd = sc.range(0, 100, numSlices=10)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        data = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                  int(np.random.randint(0, 2, size=())))).collect()
        df = spark.createDataFrame(data=data, schema=["feature", "label"])

        val_rdd = sc.range(0, 20, numSlices=6)
        val_data = val_rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                               int(np.random.randint(0, 2, size=())))).collect()
        val_df = spark.createDataFrame(data=val_data, schema=["feature", "label"])

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
                            validation_data=val_df,
                            validation_steps=2,
                            feature_cols=["feature"],
                            label_cols=["label"])
        
        trainer.shutdown()
        trainer = Estimator.from_keras(
            model_creator=model_creator,
            verbose=True,
            config=config,
            workers_per_node=2,
            backend="spark")

        df = spark.createDataFrame(data=data, schema=["feature", "label"])
        val_df = spark.createDataFrame(data=val_data, schema=["feature", "label"])
        res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                            feature_cols=["feature"],
                            label_cols=["label"])

        res = trainer.evaluate(val_df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                label_cols=["label"])
        print("validation result: ", res)

        res = trainer.predict(df, feature_cols=["feature"]).collect()
        print("predict result: ", res)
        OrcaContext._shard_size = None

    def test_dataframe_different_train_val(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100, numSlices=10)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        val_rdd = sc.range(0, 20, numSlices=6)
        val_df = val_rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
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
                            validation_data=val_df,
                            validation_steps=2,
                            feature_cols=["feature"],
                            label_cols=["label"])

        res = trainer.evaluate(val_df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                label_cols=["label"])
        print("validation result: ", res)

        res = trainer.predict(df, feature_cols=["feature"]).collect()
        print("predict result: ", res)
        trainer.shutdown()

    def test_tensorboard(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()

        from pyspark.ml.linalg import DenseVector
        data = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                  int(np.random.randint(0, 2, size=())))).collect()
        df = spark.createDataFrame(data=data, schema=["feature", "label"])

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
                backend="spark")

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
        data = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float32)),
                                  int(np.random.randint(0, 2, size=())))).collect()
        df = spark.createDataFrame(data=data, schema=["feature", "label"])

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
                backend="spark")

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

            trainer.shutdown()

            est = Estimator.from_keras(
                model_creator=model_creator,
                verbose=True,
                config=config,
                workers_per_node=2,
                backend="spark")

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(temp_dir, "best"),
                                                   save_weights_only=False,
                                                   save_best_only=True
                                                   )
            ]

            res = est.fit(df, epochs=3, batch_size=4, steps_per_epoch=25,
                          callbacks=callbacks,
                          feature_cols=["feature"],
                          label_cols=["label"],
                          validation_data=df,
                          validation_steps=1
                          )
            assert len(os.listdir(os.path.join(temp_dir, "best"))) > 0

            est.shutdown()
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

        df = spark.createDataFrame(data=df.collect(), schema=["index", "input_1", "input_2"])

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
            backend="spark")

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
