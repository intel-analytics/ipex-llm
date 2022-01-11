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
import tensorflow as tf
import numpy as np
from unittest import TestCase
from bigdl.orca.automl.auto_estimator import AutoEstimator
import pytest


def model_creator(config):
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(config["hidden_size"],
                                                              input_shape=(1,)),
                                        tf.keras.layers.Dense(1)])
    model.compile(loss="mse",
                  optimizer=tf.keras.optimizers.SGD(config["lr"]),
                  metrics=["mse"])
    return model


def model_creator_multi_inputs(config):
    from tensorflow.keras.layers import Input, Dense, Concatenate
    from tensorflow.keras.models import Model
    input_1 = Input(shape=(32,))
    input_2 = Input(shape=(64,))

    x = Dense(config["dense_1"], activation="relu")(input_1)
    x = Model(inputs=input_1, outputs=x)

    y = Dense(config["dense_1"], activation="relu")(input_2)
    y = Model(inputs=input_2, outputs=y)

    combined = Concatenate(axis=1)([x.output, y.output])

    z = Dense(config["dense_2"], activation="relu")(combined)
    z = Dense(1, activation="linear")(z)

    model = Model(inputs=[x.input, y.input], outputs=z)

    model.compile(loss="mse",
                 optimizer=tf.keras.optimizers.Adam(config["lr"]),
                 metrics=["mse"])

    return model


def get_search_space_multi_inputs():
    from bigdl.orca.automl import hp
    return {
        "dense_1": hp.choice([8, 16]),
        "dense_2": hp.choice([2, 4]),
        "lr": hp.choice([0.001, 0.003, 0.01]),
        "batch_size": hp.choice([32, 64])
    }


def get_multi_inputs_data():
    def get_df(size):
        rdd = sc.parallelize(range(size))
        df = rdd.map(lambda x: ([float(x)] * 32, [float(x)] * 64,
                        [int(np.random.randint(0, 2, size=()))])
                ).toDF(["f1", "f2", "label"])
        return df

    from pyspark.sql import SparkSession
    from bigdl.orca import OrcaContext
    sc = OrcaContext.get_spark_context()
    spark = SparkSession(sc)
    feature_cols = ["f1", "f2"]
    label_cols = ["label"]
    train_df = get_df(size=100)
    val_df = get_df(size=30)
    return train_df, val_df, feature_cols, label_cols


def get_train_val_data():
    def get_x_y(size):
        x = np.random.rand(size)
        y = x / 2

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        return x, y
    data = get_x_y(size=1000)
    validation_data = get_x_y(size=400)
    return data, validation_data


def create_linear_search_space():
    from bigdl.orca.automl import hp
    return {
        "hidden_size": hp.choice([5, 10]),
        "lr": hp.choice([0.001, 0.003, 0.01]),
        "batch_size": hp.choice([32, 64])
    }


class TestTFKerasAutoEstimator(TestCase):
    def setUp(self) -> None:
        from bigdl.orca import init_orca_context
        init_orca_context(cores=4, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from bigdl.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        auto_est = AutoEstimator.from_keras(model_creator=model_creator,
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")

        data, validation_data = get_train_val_data()
        auto_est.fit(data=data,
                     validation_data=validation_data,
                     search_space=create_linear_search_space(),
                     n_sampling=2,
                     epochs=1,
                     metric="mse")
        assert auto_est.get_best_model()
        best_config = auto_est.get_best_config()
        assert "hidden_size" in best_config
        assert all(k in best_config.keys() for k in create_linear_search_space().keys())

    def test_fit_multiple_times(self):
        auto_est = AutoEstimator.from_keras(model_creator=model_creator,
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")

        data, validation_data = get_train_val_data()
        auto_est.fit(data=data,
                     validation_data=validation_data,
                     search_space=create_linear_search_space(),
                     n_sampling=2,
                     epochs=1,
                     metric="mse")
        with pytest.raises(RuntimeError):
            auto_est.fit(data=data,
                         validation_data=validation_data,
                         search_space=create_linear_search_space(),
                         n_sampling=2,
                         epochs=1,
                         metric="mse")

    def test_fit_metric_func(self):
        auto_est = AutoEstimator.from_keras(model_creator=model_creator,
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")

        data, validation_data = get_train_val_data()

        def pyrmsle(y_true, y_pred):
            y_pred[y_pred < -1] = -1 + 1e-6
            elements = np.power(np.log1p(y_true) - np.log1p(y_pred), 2)
            return float(np.sqrt(np.sum(elements) / len(y_true)))

        with pytest.raises(ValueError) as exeinfo:
            auto_est.fit(data=data,
                         validation_data=validation_data,
                         search_space=create_linear_search_space(),
                         n_sampling=2,
                         epochs=1,
                         metric=pyrmsle)
        assert "metric_mode" in str(exeinfo)

        auto_est.fit(data=data,
                     validation_data=validation_data,
                     search_space=create_linear_search_space(),
                     n_sampling=2,
                     epochs=1,
                     metric=pyrmsle,
                     metric_mode="min")

    def test_multiple_inputs_model(self):
        auto_est = AutoEstimator.from_keras(model_creator=model_creator_multi_inputs,
                                            logs_dir="/tmp/zoo_automl_logs",
                                            resources_per_trial={"cpu": 2},
                                            name="test_fit")

        data, validation_data, feature_cols, label_cols = get_multi_inputs_data()
        auto_est.fit(data=data,
                     validation_data=validation_data,
                     search_space=get_search_space_multi_inputs(),
                     n_sampling=2,
                     epochs=1,
                     metric="mse",
                     feature_cols=feature_cols,
                     label_cols=label_cols,
                     )
        assert auto_est.get_best_model()
        best_config = auto_est.get_best_config()
        assert "lr" in best_config
        assert all(k in best_config.keys() for k in get_search_space_multi_inputs().keys())


if __name__ == "__main__":
    pytest.main([__file__])
