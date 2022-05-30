import os
import sys
import tensorflow as tf
import tensorflow_ranking as tfr
from optparse import OptionParser
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca import OrcaContext
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.learn.tf2 import Estimator


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
        user_embeddings = self.user_embeddings(features["userid"])

        # Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
        # tensor.
        movie_embeddings = self.movie_embeddings(features["titles"])

        # We want to concatenate user embeddings with movie emebeddings to pass
        # them into the ranking model. To do so, we need to reshape the user
        # embeddings to match the shape of movie embeddings.
        list_length = features["titles"].shape[1]
        user_embedding_repeated = tf.repeat(
            tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

        # Once reshaped, we concatenate and pass into the dense layers to generate
        # predictions.
        concatenated_embeddings = tf.concat(
            [user_embedding_repeated, movie_embeddings], 2)

        return self.score_model(concatenated_embeddings)

    def compute_loss(self, features, training=False):
        labels = features.pop("ratings")

        scores = self(features)

        return self.task(
            labels=labels,
            predictions=tf.squeeze(scores, axis=-1),
        )


if __name__ == "__main__":
    # data_dir = "/home/yina/Documents/data/movielen/ml-1m"
    data_dir = "/Users/yita/Documents/intel/data"
    init_orca_context(memory="20g", cores=18, init_ray_on_spark=False)
    dataset = {
        "ratings": ['userid', 'movieid', 'rating', 'timestamp'],
        "users": ["userid", "gender", "age", "occupation", "zip-code"],
        "movies": ["movieid", "title", "genres"]
    }

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
        .dropna(columns=None).select(["userid", "title", "rating"])
    # cast
    full_tbl = full_tbl.cast(["rating"], "int")
    full_tbl = full_tbl.cast(["userid"], "string")
    train_tbl, test_tbl = full_tbl.random_split([0.8, 0.2], seed=1)

    num_list_per_user = 50
    num_example_per_list = 5

    # def sample_list(movie_list, rating_list):
    #     random_state = np.random.RandomState(42)
    #     length = len(movie_list)
    #     sample_lists = []
    #     for _ in range(num_list_per_user):
    #         sampled_indices = random_state.choice(length, size=num_example_per_list, replace=False)
    #         sampled_movies = [movie_list[idx] for idx in sampled_indices]
    #         sampled_ratings = [rating_list[idx] for idx in sampled_indices]
    #         sample_lists.append([sampled_movies, sampled_ratings])
    #     return sample_lists


    def sample(num_list_per_user, num_example_per_list):
        def sample_idx(length):
            random_state = np.random.RandomState(42)
            sampled_indices_list = []
            for _ in range(num_list_per_user):
                sampled_indices = random_state.choice(range(length), size=num_example_per_list,
                                                      replace=False)
                sampled_indices_list.append(sampled_indices.tolist())
            return sampled_indices_list
        return sample_idx

    def sample_feature(feature_idx_list):
        feature_list, indices = feature_idx_list
        return [feature_list[int(idx)] for idx in indices]


    def preprocess(tbl, feature_cols, num_list_per_user, num_examples_per_list):
        col_dict = {"collect_list(" + c + ")": c + "s" for c in feature_cols}
        tbl = tbl.group_by("userid", agg="collect_list")
        tbl = tbl.rename(col_dict)

        # TODO: Drop the user if they don't have enough ratings
        tbl = tbl.sample_listwise(["ratings", "titles"], num_list_per_user,
                                  num_examples_per_list, 42)
        return tbl

    import math

    train_tbl = preprocess(train_tbl, ["title", "rating"], 50, 5)
    test_tbl = preprocess(test_tbl, ["title", "rating"], 1, 5)
    train_tbl.show(truncate=False)
    print(train_tbl.schema)

    train_count = train_tbl.size()
    steps = math.ceil(train_count / 8192)
    print("train size: ", train_count, ", steps: ", steps)

    from bigdl.orca.data.tf.data import Dataset
    train_dataset = Dataset.from_feature_table(train_tbl)

    unique_movie_titles = tbl_dict["movies"].get_vocabularies(["title"])["title"]
    tbl_dict["users"] = tbl_dict["users"].cast("userid", "string")
    unique_user_ids = tbl_dict["users"].get_vocabularies(["userid"])["userid"]
    print(len(unique_movie_titles), len(unique_user_ids))
    print(unique_movie_titles[0:2])
    print(unique_user_ids[0:2])

    def model_creator(config):
        model = RankingModel(tfr.keras.losses.ListMLELoss())
        from bigdl.friesian.models import TFRSModel
        model = TFRSModel(model)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
        return model

    config = {
        "lr": 0.1
    }

    est = Estimator.from_keras(model_creator=model_creator,
                               verbose=True,
                               config=config, backend="tf2")
    est.fit(train_dataset, 30, batch_size=8192, steps_per_epoch=steps)
