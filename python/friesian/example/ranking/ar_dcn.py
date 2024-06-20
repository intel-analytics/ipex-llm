import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2.estimator import Estimator
from optparse import OptionParser


class DCN(tfrs.Model):
	def __init__(self, use_cross_layer, deep_layer_sizes, projection_dim=None):
		super().__init__()

		self.embedding_dimension = 32

		str_features = ["movie_id", "user_id", "user_zip_code",
		                "user_occupation_text"]
		int_features = ["user_gender", "bucketized_user_age"]

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
			loss=tf.keras.losses.MeanSquaredError(),#reduction=tf.keras.losses.Reduction.NONE),
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
		labels = features.pop("user_rating")
		scores = self(features)
		return self.task(
			labels=labels,
			predictions=scores,
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
	parser.add_option('--num_executor', type=int, default=8,
	                  help='The number of executor.')
	parser.add_option('--driver_cores', type=int, default=4,
	                  help='The driver core number.')
	parser.add_option('--driver_memory', type=str, default="36g",
	                  help='The driver memory.')
	parser.add_option("--data_dir", dest="data_dir", default="~/tensorflow_datasets")
	parser.add_option("--data_tar_gz", dest="data_tar_gz", default="")

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

	ratings = tfds.load("movielens/100k-ratings", split="train", data_dir=options.data_dir)
	ratings = ratings.map(lambda x: {
		"movie_id": x["movie_id"],
		"user_id": x["user_id"],
		"user_rating": x["user_rating"],
		"user_gender": int(x["user_gender"]),
		"user_zip_code": x["user_zip_code"],
		"user_occupation_text": x["user_occupation_text"],
		"bucketized_user_age": int(x["bucketized_user_age"]),
	})

	feature_names = ["movie_id", "user_id", "user_gender", "user_zip_code", "user_occupation_text",
	                 "bucketized_user_age"]

	vocabularies = {}

	for feature_name in feature_names:
		vocab = ratings.batch(1_000_000).map(lambda x: x[feature_name])
		vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))

	config = {
		"lr": 0.01
	}


	def model_creator(config):
		model = DCN(use_cross_layer=True, deep_layer_sizes=[192, 192])
		from bigdl.friesian.feature.utils import distribute_tfrs_model
		model = distribute_tfrs_model(model)
		model.compile(optimizer=tf.keras.optimizers.Adam(config['lr']))
		return model


	def train_data_creator(config, batch_size):
		ratings = tfds.load("movielens/100k-ratings", split="train", data_dir=executor_data)
		ratings = ratings.map(lambda x: {
			"movie_id": x["movie_id"],
			"user_id": x["user_id"],
			"user_rating": x["user_rating"],
			"user_gender": tf.cast(x["user_gender"], tf.int32),
			"user_zip_code": x["user_zip_code"],
			"user_occupation_text": x["user_occupation_text"],
			"bucketized_user_age": tf.cast(x["bucketized_user_age"], tf.int32),
		})
		# options = tf.data.Options()
		# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		# ratings = ratings.with_options(options)
		tf.random.set_seed(42)
		shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

		train = shuffled.take(80_000).shuffle(100_000).batch(8192)
		return train


	def val_data_creator(config, batch_size):
		ratings = tfds.load("movielens/100k-ratings", split="train", data_dir=executor_data)
		ratings = ratings.map(lambda x: {
			"movie_id": x["movie_id"],
			"user_id": x["user_id"],
			"user_rating": x["user_rating"],
			"user_gender": tf.cast(x["user_gender"], tf.int32),
			"user_zip_code": x["user_zip_code"],
			"user_occupation_text": x["user_occupation_text"],
			"bucketized_user_age": tf.cast(x["bucketized_user_age"], tf.int32),
		})
		# options = tf.data.Options()
		# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
		# ratings = ratings.with_options(options)
		tf.random.set_seed(42)
		shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

		test = shuffled.skip(80_000).take(20_000).batch(4096)
		return test


	est = Estimator.from_keras(model_creator=model_creator, config=config)
	est.fit(data=train_data_creator, epochs=1, batch_size=8192, validation_data=val_data_creator, verbose=1)
	stop_orca_context()

