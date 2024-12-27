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
                tfr.keras.metrics.MRRMetric(name="mrr"),
                tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
                tf.keras.metrics.RootMeanSquaredError("RMSE")]
        )

    def call(self, features):
        # We first convert the id features into embeddings.
        # User embeddings are a [batch_size, embedding_dim] tensor.
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
    parser.add_option('--executor_cores', type=int, default=18,
                      help='The executor core number.')
    parser.add_option('--executor_memory', type=str, default="10g",
                      help='The executor memory.')
    parser.add_option('--num_executor', type=int, default=8,
                      help='The number of executor.')
    parser.add_option('--driver_cores', type=int, default=4,
                      help='The driver core number.')
    parser.add_option('--driver_memory', type=str, default="36g",
                      help='The driver memory.')
    parser.add_option("--data_dir", dest="data_dir", default="~/tensorflow_datasets")
    parser.add_option("--data_tar_gz", dest="data_tar_gz", default="") \

    (options, args) = parser.parse_args(sys.argv)
    executor_data = options.data_dir
    if options.cluster_mode == "local":
        init_orca_context("local", cores=options.executor_cores, memory=options.executor_memory,
                          init_ray_on_spark=True)
    elif options.cluster_mode == "yarn":
        if options.data_tar_gz == "":
            raise ValueError("data_tar_gz should be provided if cluster_mode = yarn")
        executor_data = "data/" + options.data_tar_gz.split(".")[0]
        init_orca_context("yarn-client", cores=options.executor_cores,
                          num_nodes=options.num_executor, memory=options.executor_memory,
                          driver_cores=options.driver_cores, driver_memory=options.driver_memory,
                          init_ray_on_spark=True, additional_archive=options.data_tar_gz + "#data", extra_python_lib="utils.py")
    elif options.cluster_mode == "spark-submit":
        init_orca_context("spark-submit")
    else:
        raise ValueError(
            "cluster_mode should be one of 'local', 'yarn', 'standalone' and 'spark-submit'"
            ", but got " + args.cluster_mode)

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

    config = {
        "lr": 0.01
    }

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
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ratings = ratings.with_options(options)

        train = ratings.take(80_000)
        print(train.take(1))
        train = tfrs.examples.movielens.sample_listwise(
            train,
            num_list_per_user=50,
            num_examples_per_list=5,
            seed=42
        )
        print(train.take(1))
        train = train.batch(batch_size)
        return train

    def val_data_creator(config, batch_size):
        ratings = tfds.load("movielens/100k-ratings", split="train", data_dir=executor_data)
        ratings = ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"],
        })
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        ratings = ratings.with_options(options)

        test = ratings.skip(80_000).take(20_000)
        test = tfrs.examples.movielens.sample_listwise(
            test,
            num_list_per_user=1,
            num_examples_per_list=5,
            seed=42
        )

        test = test.batch(batch_size)
        return test

    est = Estimator.from_keras(model_creator=model_creator, config=config)
    est.fit(data=train_data_creator, epochs=1, batch_size=batch_size, verbose=0)
    stats = est.evaluate(data=val_data_creator, batch_size=batch_size)
    print(stats)
    stop_orca_context()
