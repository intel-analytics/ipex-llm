import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2.estimator import Estimator
from optparse import OptionParser


class RankingModel(tfrs.Model):
    def __init__(self, loss):
        super().__init__()
        embedding_dimension = 32
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids),
            tf.keras.layers.Embedding(len(unique_user_ids) + 2, embedding_dimension)
        ])
        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 2, embedding_dimension)
        ])
        self.score_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        self.task = tfrs.tasks.Ranking(
            loss=loss,
            metrics=[
                tfr.keras.metrics.MRRMetric(name="mrr"),
                tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
                tf.keras.metrics.RootMeanSquaredError("RMSE")]
        )

    def call(self, features):
        user_embeddings = self.user_embeddings(features["user_id"])
        movie_embeddings = self.movie_embeddings(features["movie_title"])
        list_length = features["movie_title"].shape[1]
        user_embedding_repeated = tf.repeat(
            tf.expand_dims(user_embeddings, 1), [list_length], axis=1)
        concatenated_embeddings = tf.concat(
            [user_embedding_repeated, movie_embeddings], 2)
        return self.score_model(concatenated_embeddings)

    def compute_loss(self, features, training=False):
        labels = features.pop("user_rating")
        scores = self(features)
        return self.task(labels=labels, predictions=tf.squeeze(scores, axis=-1), )


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('--cluster_mode', type=str, default="local",
                      help='The cluster mode, such as local, yarn or standalone.')
    parser.add_option('--master', type=str, default=None,
                      help='The master url, only used when cluster mode is standalone.')
    parser.add_option('--executor_cores', type=int, default=4,
                      help='The executor core number.')
    parser.add_option('--executor_memory', type=str, default="10g",
                      help='The executor memory.')
    parser.add_option("--data_dir", dest="data_dir", default="~/tensorflow_datasets")


    (options, args) = parser.parse_args(sys.argv)
    executor_data = options.data_dir
    init_orca_context("local", cores=options.executor_cores, memory=options.executor_memory,
                     init_ray_on_spark=True)

    ratings = tfds.load("movielens/100k-ratings", split="train", data_dir=executor_data)
    movies = tfds.load("movielens/100k-movies", split="train", data_dir=executor_data)
    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"],
    })
    movies = movies.map(lambda x: x["movie_title"])
    unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
    unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
        lambda x: x["user_id"]))))

    config = {"lr": 0.01}
    batch_size = 1024

    def model_creator(config):
        model = RankingModel(tf.keras.losses.MeanSquaredError())
        from bigdl.friesian.feature.utils import distribute_tfrs_model
        model = distribute_tfrs_model(model)
        model.compile(optimizer=tf.keras.optimizers.Adam(config['lr']))
        return model

    def train_data_creator(config, batch_size):
        ratings = tfds.load("movielens/100k-ratings", split="train", data_dir=executor_data)

        ratings = ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"],
        })

        train = ratings.take(80_000)
        train = tfrs.examples.movielens.sample_listwise(
            train,
            num_list_per_user=50,
            num_examples_per_list=5,
            seed=42
        )
        train = train.batch(batch_size)
        return train

    est = Estimator.from_keras(model_creator=model_creator, config=config)
    est.fit(data=train_data_creator, epochs=1, batch_size=batch_size, verbose=1)
    stop_orca_context()
