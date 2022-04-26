import math
import sys
from optparse import OptionParser

import tensorflow as tf
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.data import SparkXShards

import bigdl.orca.data.pandas
import numpy as np
import tensorflow_datasets as tfds
from bigdl.orca import init_orca_context
import tensorflow_recommenders as tfrs
from bigdl.orca.learn.tf2.estimator import Estimator
from bigdl.orca.common import OrcaContextMeta
from bigdl.orca.data.tf.tf2_data import Dataset

import pickle

with open('/Users/yita/Documents/intel/data/ml-1m-vocab.pkl', 'rb') as f:
    vocabularies = pickle.load(f)


class DCN(tfrs.Model):

    def __init__(self, use_cross_layer, deep_layer_sizes, projection_dim=None):
        super().__init__()

        self.embedding_dimension = 32

        # str_features = ["movie_id", "user_id", "user_zip_code",
        #                 "user_occupation_text"]
        # int_features = ["user_gender", "bucketized_user_age"]

        str_features = ["movie_id", "user_id", "zip_code",
                        "occupation"]
        int_features = ["gender", "age"]

        self._all_features = str_features + int_features
        self._embeddings = {}

        # Compute embeddings for string features.
        for feature_name in str_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.StringLookup(
                    vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1,
                                              self.embedding_dimension)
                ])

        # Compute embeddings for int features.
        for feature_name in int_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.IntegerLookup(
                    vocabulary=vocabulary, mask_value=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1,
                                              self.embedding_dimension)
                ])

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim,
                kernel_initializer="glorot_uniform")
        else:
            self._cross_layer = None

        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
                             for layer_size in deep_layer_sizes]

        self._logit_layer = tf.keras.layers.Dense(1)

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
            metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
        )

    def call(self, features):
        # Concatenate embeddings
        embeddings = []
        for feature_name in self._all_features:
            embedding_fn = self._embeddings[feature_name]
            embeddings.append(embedding_fn(features[feature_name]))

        x = tf.concat(embeddings, axis=1)

        # Build Cross Network
        if self._cross_layer is not None:
            x = self._cross_layer(x)

        # Build Deep Network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)

        return self._logit_layer(x)

    def compute_loss(self, features, training=False):
        labels = features.pop("rating")
        scores = self(features)
        return self.task(
            labels=labels,
            predictions=scores,
        )

# OrcaContextMeta._eager_mode = False

sc = init_orca_context("local", cores=4, memory="2g", init_ray_on_spark=True)

cols = ["movie_id", "user_id", "gender", "age", "occupation", "zip_code", "rating"]

ratings = tfds.load("movielens/100k-ratings", split="train")
ratings = ratings.map(lambda x: {
    "movie_id": x["movie_id"],
    "user_id": x["user_id"],
    "rating": x["user_rating"],
    "gender": int(x["user_gender"]),
    "zip_code": x["user_zip_code"],
    "occupation": x["user_occupation_text"],
    "age": int(x["bucketized_user_age"]),
})

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)


def dict_list_to_dict(dict_list):
    data_dict = dict()
    for col in cols:
        data_dict[col] = list()
    for i in dict_list:
        for col in cols:
            data_dict[col].append(i[col])
    for col in cols:
        data_dict[col] = np.asarray(data_dict[col])
    return data_dict


np_train = list(tfds.as_numpy(train))
train_dict = dict_list_to_dict(np_train)
np_val = list(tfds.as_numpy(test))
val_dict = dict_list_to_dict(np_val)


dataset_shard = SparkXShards(sc.parallelize([train_dict]))
val_shard = SparkXShards(sc.parallelize([val_dict]))


train_dataset = Dataset.from_tensor_slices(dataset_shard)
val_dataset = Dataset.from_tensor_slices(val_shard)

config = {
    "lr": 0.01
}


def model_creator(config):
    model = DCN(use_cross_layer=True, deep_layer_sizes=[192, 192])
    model.compile(optimizer=tf.keras.optimizers.Adam(config['lr']))
    return model


estimator = Estimator.from_keras(model_creator=model_creator,
                                 verbose=True,
                                 config=config,
                                 backend="tf2")

# estimator = Estimator.from_keras(model_creator=model_creator,
#                                  verbose=True,
#                                  config=config,
#                                  backend="spark",
#                                  model_dir="/Users/yita/Documents/intel/data")

estimator.fit(train_dataset, 8, batch_size=8192, steps_per_epoch=10, validation_data=val_dataset, validation_steps=3)
estimator.evaluate(val_dataset, batch_size=4096, num_steps=5)
