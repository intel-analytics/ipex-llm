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
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This example is based on Tensorflow Recommenders example
# [basic ranking](https://www.tensorflow.org/recommenders/examples/basic_ranking).


import os
import sys
import math
from optparse import OptionParser
import tensorflow as tf

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca.data.tf.data import Dataset
from pyspark.sql.functions import col, mean, stddev


class SampleRankingModel(tf.keras.Model):
    def __init__(self, unique_user_ids, unique_movie_titles):
        super().__init__()
        embedding_dim = 32

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim)])
        self.movie_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dim)])
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

    def train_step(self, data):
        y = data["user_rating"]

        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        y = data["user_rating"]

        y_pred = self(data, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


def parse_args():
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
    parser.add_option("--batch_size", dest="batch_size", default=8192,
                      help="The batch size used for training and evaluate.")

    (options, args) = parser.parse_args(sys.argv)
    return (options, args)


if __name__ == "__main__":
    (options, args) = parse_args()
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

    spark = OrcaContext.get_spark_session()

    # UserID::MovieID::Rating::Timestamp
    # UserID::Gender::Age::Occupation::Zip-code
    # MovieID::Title::Genres
    dataset = {
        "ratings": ['userid', 'movieid', 'rating', 'timestamp'],
        "movies": ["movieid", "title", "genres"]
    }

    # Use spark dataframe to read the .dat files.
    df_dict = dict()
    for data, cols in dataset.items():
        df = spark.read.options(header=False, inferSchema=True, delimiter=":")\
            .csv(os.path.join(data_dir, data + ".dat"))
        tmp_cols = df.columns[::2]
        df = df.select(tmp_cols)
        for old_name, new_name in zip(tmp_cols, cols):
            df = df.withColumnRenamed(old_name, new_name)
        df_dict[data] = df

    full_df = df_dict["ratings"].join(df_dict["movies"], "movieid").dropna()
    full_df = full_df.withColumn("rating", col("rating").cast("float"))
    full_df = full_df.withColumn("userid", col("userid").cast("string"))
    full_df.show(5, False)

    # Generate vocabularies for the StringLookup layers
    user_id_vocab = full_df.select("userid").distinct().rdd.map(lambda row: row["userid"]).collect()
    movie_title_vocab = full_df.select("title").distinct().rdd\
        .map(lambda row: row["title"]).collect()

    # Calculate mean and standard deviation for normalization
    df_stats = full_df.select(
        mean(col('timestamp')).alias('mean'),
        stddev(col('timestamp')).alias('std')
    ).collect()
    mean = df_stats[0]['mean']
    stddev = df_stats[0]['std']

    # Sample 10% data and split it into a training and test set.
    part_df = full_df.sample(fraction=0.1, seed=42)
    train_df, test_df = part_df.randomSplit([0.8, 0.2])

    train_count = train_df.count()
    steps = math.ceil(train_count / options.batch_size)
    print("train size: ", train_count, ", steps: ", steps)

    test_count = test_df.count()
    test_steps = math.ceil(test_count / options.batch_size)
    print("test size: ", test_count, ", steps: ", test_steps)

    # Create Orca TF Datasets from a Spark DataFrame
    train_ds = Dataset.from_spark_df(train_df)
    test_ds = Dataset.from_spark_df(test_df)

    # Once the Orca TF Dataset is created, we can perform some data preprocessing using the map
    # function. Since the model use input["movie_title"], input["user_id"] and input["user_rating"]
    # in the model call, train_step and test_step function, we should change the key name of the
    # Dataset. Also, we normalize the continuous feature timestamp here.
    def preprocess(x):
        return {
            "movie_title": x["title"],
            "user_id": x["userid"],
            "user_rating": x["rating"],
            # Normalize continuous timestamp
            "timestamp": (tf.cast(x["timestamp"], tf.float32) - mean) / stddev
        }

    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)

    # Create an Orca Estimator using the SampleRankingModel.
    def model_creator(config):
        model = SampleRankingModel(unique_user_ids=user_id_vocab,
                                   unique_movie_titles=movie_title_vocab)
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.RootMeanSquaredError()],
                      optimizer=tf.keras.optimizers.Adagrad(config["lr"]))
        return model

    config = {
        "lr": 0.1
    }

    est = Estimator.from_keras(model_creator=model_creator,
                               verbose=True,
                               config=config, backend="ray")

    # Train the model using Orca TF Dataset.
    est.fit(train_ds, 3, batch_size=options.batch_size, steps_per_epoch=steps)
    # Evaluate the model on the test set.
    est.evaluate(test_ds, options.batch_size, num_steps=test_steps)

    # Shutdown the Estimator and stop the orca context.
    est.shutdown()
    stop_orca_context()
