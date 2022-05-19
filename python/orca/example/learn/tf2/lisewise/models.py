import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_ranking as tfr
import numpy as np


class Padded2RaggedModel(tfrs.Model):

    def __init__(self, unique_userids, unique_movie_titles, max_len, loss):
        super().__init__()
        embedding_dimension = 32
        self.max_len = max_len  # for output padding
        self._loss = loss

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.Input(shape=(), dtype=tf.string),
            tf.keras.layers.StringLookup(vocabulary=unique_userids),
            tf.keras.layers.Embedding(len(unique_userids) + 2, embedding_dimension)
        ], name="user_emb")

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.Input(shape=(None,), dtype=tf.string, ragged=True),
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 2, embedding_dimension)
        ], name="movie_emb")

        # Compute predictions.
        self.score_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ], name="MLP")

        self._metrics = [
            tfr.keras.metrics.NDCGMetric(name="ndcg_metric", ragged=True),
            tf.keras.metrics.RootMeanSquaredError()
        ]

        self.task = tfrs.tasks.Ranking(
            loss=self._loss,
            metrics=self._metrics
        )

    @classmethod
    def _expand(cls, x):
        return tf.expand_dims(x, axis=-1)

    @classmethod
    def _repeat(cls, src, length, axis):
        return tf.repeat(
            tf.expand_dims(src, axis=axis),

            [length],
            axis=axis
        )

    def call(self, features, padding_output=True):
        userid = self._expand(features[0])
        titles = self._expand(features[1])
        lengths = features[-1]

        max_length = features[1].shape[1]  # padded

        user_emb = self.user_embeddings(userid)
        user_emb_repeated = self._repeat(user_emb, max_length, -2)
        movie_emb = self.movie_embeddings(titles)

        concatenated_emb = tf.concat(
            [user_emb_repeated, movie_emb], axis=-1
        )

        scores = self.score_model(concatenated_emb)
        scores = tf.squeeze(scores, axis=-1)
        ragged_scores = tf.RaggedTensor.from_tensor(scores, lengths)
        # score
        if padding_output:
            ragged_shape = ragged_scores.shape
            return ragged_scores.to_tensor(np.inf, shape=(ragged_shape[0], self.max_len))
        return ragged_scores

    def compute_loss(self, inputs, training=False):
        features, labels = inputs
        lengths = features[-1]

        labels = tf.RaggedTensor.from_tensor(labels, lengths)
        # print(labels)
        labels = tf.cast(labels, dtype=tf.float32)
        scores = self.call(features, padding_output=False)
        return self.task(
            labels=labels,
            predictions=scores,
        )
