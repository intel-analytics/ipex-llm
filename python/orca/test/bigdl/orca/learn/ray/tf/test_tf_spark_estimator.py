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

from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca import OrcaContext


resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../../resources")


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
                backend="spark")

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
                workers_per_node=3,
                backend="spark")

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
                backend="spark")

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
                backend="spark")

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
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
                                int(np.random.randint(0, 2, size=())))).toDF(["feature", "label"])

        val_rdd = sc.range(0, 20, numSlices=6)
        val_df = val_rdd.map(lambda x: (DenseVector(np.random.randn(1, ).astype(np.float)),
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

    def test_save_load_model_h5(self):
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
                backend="spark")

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            print("start saving")
            trainer.save(os.path.join(temp_dir, "a.h5"))

            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            before_res = trainer.predict(df, feature_cols=["feature"]).collect()
            expect_res = np.concatenate([part["prediction"] for part in before_res])

            trainer.load(os.path.join(temp_dir, "a.h5"))
            
            # continous predicting
            after_res = trainer.predict(df, feature_cols=["feature"]).collect()
            pred_res = np.concatenate([part["prediction"] for part in after_res])

            assert np.array_equal(expect_res, pred_res)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_load_model_savemodel(self):
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
                backend="spark")

            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              validation_data=df,
                              validation_steps=1)

            print("start saving")
            trainer.save(os.path.join(temp_dir, "saved_model"))

            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=["feature"],
                                   label_cols=["label"])
            print("validation result: ", res)

            before_res = trainer.predict(df, feature_cols=["feature"]).collect()
            expect_res = np.concatenate([part["prediction"] for part in before_res])

            trainer.load(os.path.join(temp_dir, "saved_model"))
            
            # continous predicting
            after_res = trainer.predict(df, feature_cols=["feature"]).collect()
            pred_res = np.concatenate([part["prediction"] for part in after_res])

            assert np.array_equal(expect_res, pred_res)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_load_model_architecture(self):
        config = {
            "lr": 0.2
        }
        import uuid
        model_path = os.path.join(tempfile.gettempdir(), str(uuid.uuid1()) + ".json")
        try:
            model = simple_model(config)
            with open(model_path, "w") as f:
                f.write(model.to_json())
        
            from bigdl.dllib.utils.file_utils import enable_hdfs_load

            @enable_hdfs_load
            def load_model_architecture(path):
                with open(path, "rb") as f:
                    model = tf.keras.models.model_from_json(f.read())
                return model

            model_load = load_model_architecture(model_path)
            assert model.summary() == model_load.summary()
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)



if __name__ == "__main__":
    pytest.main([__file__])
