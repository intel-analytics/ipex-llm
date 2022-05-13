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

import sys
import numpy as np
from optparse import OptionParser
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs

from bigdl.orca.learn.tf2.estimator import Estimator
from bigdl.orca.common import init_orca_context


class RankingModel(tfrs.Model):

    def __init__(self, loss):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids),
            tf.keras.layers.Embedding(len(unique_user_ids) + 2, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 2, embedding_dimension)
        ])

        # Compute predictions.
        self.score_model = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

        self.task = tfrs.tasks.Ranking(
            loss=loss,
            metrics=[
                tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
                tf.keras.metrics.RootMeanSquaredError()
            ]
        )

    def call(self, features):
        # We first convert the id features into embeddings.
        # User embeddings are a [batch_size, embedding_dim] tensor.
        user_embeddings = self.user_embeddings(features["user_id"])

        # Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
        # tensor.
        movie_embeddings = self.movie_embeddings(features["movie_title"])

        # We want to concatenate user embeddings with movie emebeddings to pass
        # them into the ranking model. To do so, we need to reshape the user
        # embeddings to match the shape of movie embeddings.
        list_length = features["movie_title"].shape[1]
        user_embedding_repeated = tf.repeat(
            tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

        # Once reshaped, we concatenate and pass into the dense layers to generate
        # predictions.
        concatenated_embeddings = tf.concat(
            [user_embedding_repeated, movie_embeddings], 2)

        return self.score_model(concatenated_embeddings)

    def compute_loss(self, features, training=False):
        labels = features.pop("user_rating")

        scores = self(features)

        return self.task(
            labels=labels,
            predictions=tf.squeeze(scores, axis=-1),
        )


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('--cluster_mode', type=str, default="local",
                      help='The cluster mode, such as local, yarn or standalone.')
    parser.add_option('--master', type=str, default=None,
                      help='The master url, only used when cluster mode is standalone.')
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
    parser.add_option("--data_dir", dest="data_dir", default="./total.parquet",
                      help='The directory to read/write tensorflow datasets')

    (options, args) = parser.parse_args(sys.argv)
    data_dir = options.data_dir

    if options.cluster_mode == "local":
        sc = init_orca_context("local", cores=options.executor_cores,
                               memory=options.executor_memory,
                               init_ray_on_spark=True)
    elif options.cluster_mode == "yarn":
        if options.data_tar_gz == "":
            raise ValueError("data_tar_gz should be provided if cluster_mode = yarn")
        executor_data = "data/" + options.data_tar_gz.split(".")[0]
        sc = init_orca_context("yarn-client", cores=options.executor_cores,
                               num_nodes=options.num_executor, memory=options.executor_memory,
                               driver_cores=options.driver_cores,
                               driver_memory=options.driver_memory,
                               init_ray_on_spark=True,
                               additional_archive=options.data_tar_gz + "#data")
    else:
        raise ValueError("cluster_mode should be 'local' or 'yarn', but got " + args.cluster_mode)

    ratings = tfds.load("movielens/100k-ratings", split="train", data_dir=options.data_dir)
    movies = tfds.load("movielens/100k-movies", split="train", data_dir=options.data_dir)

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"],
    })
    movies = movies.map(lambda x: x["movie_title"])

    unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
    unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
        lambda x: x["user_id"]))))

    def model_creator(config):
        from bigdl.friesian.learn.friesianTFRSModel import FriesianTFRSModel
        model = RankingModel(tf.keras.losses.MeanSquaredError())
        model = FriesianTFRSModel(model)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(config['lr']))
        return model

    def train_data_creator(config, batch_size):
        ratings = tfds.load("movielens/100k-ratings", split="train", data_dir=executor_data)

        ratings = ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"],
        })
        tf.random.set_seed(42)

        # Split between train and tests sets, as before.
        shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
        train = shuffled.take(80_000)
        train = tfrs.examples.movielens.sample_listwise(
            train,
            num_list_per_user=50,
            num_examples_per_list=5,
            seed=42
        )
        train = train.shuffle(100_000).batch(batch_size)
        return train

    def val_data_creator(config, batch_size):
        ratings = tfds.load("movielens/100k-ratings", split="train", data_dir=executor_data)

        ratings = ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"],
        })
        tf.random.set_seed(42)

        # Split between train and tests sets, as before.
        shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
        test = shuffled.skip(80_000).take(20_000)
        test = tfrs.examples.movielens.sample_listwise(
            test,
            num_list_per_user=1,
            num_examples_per_list=5,
            seed=42
        )
        test = test.batch(batch_size)
        return test


    epochs = 30
    config = {"lr": 0.1}

    est = Estimator.from_keras(model_creator=model_creator, config=config, workers_per_node=3)
    est.fit(data=train_data_creator, epochs=epochs, batch_size=8192, verbose=1,
            validation_data=val_data_creator)

