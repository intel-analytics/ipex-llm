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
# ==============================================================================
#
# Copyright 2020 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This example is based on Tensorflow Recommenders example
# [basic ranking](https://www.tensorflow.org/recommenders/examples/basic_ranking).
#

import os
import sys
import math
from optparse import OptionParser
import tensorflow as tf
import tensorflow_recommenders as tfrs

from bigdl.friesian.models import TFRSModel
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca.data.tf.data import Dataset


class SampleRankingModel(tfrs.models.Model):
    def __init__(self, user_id_num, movie_title_num):
        super().__init__()
        embedding_dim = 32
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.user_embedding = tf.keras.layers.Embedding(user_id_num + 1, embedding_dim)
        self.movie_embedding = tf.keras.layers.Embedding(movie_title_num + 1, embedding_dim)
        self.ratings = tf.keras.Sequential([
              # Learn multiple dense layers.
              tf.keras.layers.Dense(256, activation="relu"),
              tf.keras.layers.Dense(64, activation="relu"),
              # Make rating predictions in the final layer.
              tf.keras.layers.Dense(1)
          ])

    def call(self, features):
        embeddings = tf.concat([self.user_embedding(features["user_id"]),
                                self.movie_embedding(features["movie_title"]),
                                tf.reshape(features["timestamp"], (-1, 1))], axis=1)
        return self.ratings(embeddings)

    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        labels = inputs["user_rating"]
        rating_predictions = self(inputs)
        return self.task(labels=labels, predictions=rating_predictions)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('--cluster_mode', type=str, default="local",
                      help='The cluster mode, such as local or yarn.')
    parser.add_option('--executor_cores', type=int, default=18,
                      help='The executor core number.')
    parser.add_option('--executor_memory', type=str, default="10g",
                      help='The executor memory.')
    parser.add_option('--num_executor', type=int, default=1,
                      help='The number of executor.')
    parser.add_option('--driver_cores', type=int, default=4,
                      help='The driver core number.')
    parser.add_option('--driver_memory', type=str, default="36g",
                      help='The driver memory.')
    parser.add_option("--data_dir", dest="data_dir", default="./ml-1m",
                      help='The directory to ml-1m .dat files')

    (options, args) = parser.parse_args(sys.argv)
    data_dir = options.data_dir

    if options.cluster_mode == "local":
        init_orca_context("local", cores=options.executor_cores, memory=options.executor_memory,
                          init_ray_on_spark=True)
    elif options.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=options.executor_cores,
                          num_nodes=options.num_executor, memory=options.executor_memory,
                          driver_cores=options.driver_cores, driver_memory=options.driver_memory,
                          init_ray_on_spark=True)
    else:
        raise ValueError("cluster_mode should be 'local' or 'yarn', but got " + args.cluster_mode)

    (options, args) = parser.parse_args(sys.argv)
    data_dir = options.data_dir

    # UserID::MovieID::Rating::Timestamp
    # UserID::Gender::Age::Occupation::Zip-code
    # MovieID::Title::Genres
    dataset = {
        "ratings": ['userid', 'movieid', 'rating', 'timestamp'],
        "movies": ["movieid", "title", "genres"]
    }

    # Use Friesian FeatureTable to read the .dat files.
    tbl_dict = dict()
    for data, cols in dataset.items():
        tbl = FeatureTable.read_csv(os.path.join(data_dir, data + ".dat"),
                                    delimiter=":", header=False)
        tmp_cols = tbl.columns[::2]
        tbl = tbl.select(tmp_cols)
        col_dict = {c[0]: c[1] for c in zip(tmp_cols, cols)}
        tbl = tbl.rename(col_dict)
        tbl_dict[data] = tbl

    full_tbl = tbl_dict["ratings"].join(tbl_dict["movies"], "movieid") \
        .dropna(columns=None)
    full_tbl = full_tbl.cast(["rating"], "int")
    full_tbl = full_tbl.cast(["userid"], "string")
    full_tbl.show(5, False)

    # Generate unique index value of categorical features and encode these columns with generated
    # string indices.
    str_idx = full_tbl.gen_string_idx(["userid", "title"])
    user_id_size = str_idx[0].size()
    title_size = str_idx[1].size()
    full_tbl = full_tbl.encode_string(["userid", "title"], str_idx)
    full_tbl.show(5, False)

    # Calculate mean and standard deviation for normalization
    avg, stddev = full_tbl.get_stats(["timestamp"], ["avg", "stddev"])["timestamp"]

    # Sample 10% data and split it into a training and test set.
    part_tbl = full_tbl.sample(0.1, seed=42)
    train_tbl, test_tbl = part_tbl.random_split([0.8, 0.2])

    train_count = train_tbl.size()
    steps = math.ceil(train_count / 8192)
    print("train size: ", train_count, ", steps: ", steps)

    test_count = test_tbl.size()
    test_steps = math.ceil(test_count / 4096)
    print("test size: ", test_count, ", steps: ", test_steps)

    # Create Orca TF Datasets from a Friesian FeatureTables.
    train_ds = Dataset.from_feature_table(train_tbl)
    test_ds = Dataset.from_feature_table(test_tbl)

    # Once the Orca TF Dataset is created, we can perform some data preprocessing using the map
    # function. Since the model use input["movie_title"], input["user_id"] and input["user_rating"]
    # in the model call and compute_loss function, we should change the key name of the Dataset.
    # Also, we normalize the continuous feature timestamp here.
    def preprocess(x):
        return {
            "movie_title": x["title"],
            "user_id": x["userid"],
            "user_rating": x["rating"],
            # Normalize continuous timestamp
            "timestamp": (tf.cast(x["timestamp"], tf.float32) - avg) / stddev
        }

    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    # Create an Orca Estimator using the SampleRankingModel.
    def model_creator(config):
        model = SampleRankingModel(user_id_num=user_id_size, movie_title_num=title_size)
        model = TFRSModel(model)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(config["lr"]))
        return model


    config = {
        "lr": 0.1
    }

    est = Estimator.from_keras(model_creator=model_creator,
                               verbose=True,
                               config=config, backend="tf2")

    # Train the model using Orca TF Dataset.
    est.fit(train_ds, 3, batch_size=8192, steps_per_epoch=steps)
    # Evaluate the model on the test set.
    est.evaluate(test_ds, 4096, num_steps=test_steps)

    # Shutdown the Estimator and stop the orca context.
    est.shutdown()
    stop_orca_context()
