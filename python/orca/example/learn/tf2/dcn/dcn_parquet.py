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
from optparse import OptionParser
import math
import tensorflow as tf
import tensorflow_recommenders as tfrs

from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca import init_orca_context
from bigdl.orca.learn.tf2.estimator import Estimator
from bigdl.orca.data.tf.data import Dataset
from bigdl.friesian.feature import FeatureTable


class DCN(tfrs.Model):

    def __init__(self, use_cross_layer, deep_layer_sizes, projection_dim=None):
        super().__init__()

        self.embedding_dimension = 32

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
                                              self.embedding_dimension)])

        # Compute embeddings for int features.
        for feature_name in int_features:
            vocabulary = vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.IntegerLookup(
                    vocabulary=vocabulary, mask_value=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1,
                                              self.embedding_dimension)])

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
            loss=tf.keras.losses.MeanSquaredError(),
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
                      help='The directory to read/write data')

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
        invalidInputError(False,
                          "cluster_mode should be 'local' or 'yarn', but got " + args.cluster_mode)

    cols = ["movie_id", "user_id", "gender", "age", "occupation", "zip_code", "rating"]
    str_features = ["movie_id", "user_id", "zip_code", "occupation"]
    int_features = ["gender", "age"]

    table = FeatureTable.read_parquet(data_dir)
    gender_dict = {'M': 0, 'F': 1}
    gender_to_int = lambda x: gender_dict[x]
    table = table.apply("gender", "gender", gender_to_int, dtype="int")
    table = table.ordinal_shuffle_partition()
    print(table.schema)
    # convert dtype
    table = table.cast(str_features, "string")
    table = table.cast(int_features, "int")
    print(table.schema)
    # generate vocab
    vocabularies = table.get_vocabularies(cols)
    table = table.limit(100_000).repartition(options.num_executor * 2)
    # show row number in each partition
    table.get_partition_row_number().show()
    train_tbl, test_tbl = table.random_split([0.8, 0.2])
    train_count = train_tbl.size()
    steps = math.ceil(train_count / 8192)
    print("train size: ", train_count, ", steps: ", steps)
    test_count = test_tbl.size()
    val_steps = math.ceil(test_count / 8192)
    print("test size: ", test_count, ", steps: ", val_steps)

    train_dataset = Dataset.from_feature_table(train_tbl)
    val_dataset = Dataset.from_feature_table(test_tbl)

    config = {
        "lr": 0.01
    }

    def model_creator(config):
        model = DCN(use_cross_layer=True, deep_layer_sizes=[192, 192])
        from bigdl.friesian.models import TFRSModel
        model = TFRSModel(model)
        model.compile(optimizer=tf.keras.optimizers.Adam(config['lr']))
        return model

    estimator = Estimator.from_keras(model_creator=model_creator,
                                     verbose=True,
                                     config=config,
                                     backend="ray")

    estimator.fit(train_dataset, 8, batch_size=8192, steps_per_epoch=steps,
                  validation_data=val_dataset, validation_steps=val_steps)
    estimator.evaluate(val_dataset, batch_size=8192, num_steps=val_steps)
