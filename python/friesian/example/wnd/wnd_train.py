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

from __future__ import division, print_function, unicode_literals

import os
import math
import tempfile
from bigdl.friesian.utils import SafePickle
from time import time
from argparse import ArgumentParser

from bigdl.dllib.utils.file_utils import get_remote_file_to_local
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.file import exists, makedirs
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.learn.tf2.estimator import Estimator


INT_COLS = ["c{}".format(i) for i in list(range(1, 14))]
CAT_COLS = ["c{}".format(i) for i in list(range(14, 40))]


def get_data(data_dir):
    if not exists(os.path.join(data_dir, "train_parquet")) or \
            not exists(os.path.join(data_dir, "test_parquet")):
        invalidInputError(False,
                          "Not train and test data parquet specified")
    else:
        train_tbl = FeatureTable.read_parquet(os.path.join(data_dir, "train_parquet"))
        test_tbl = FeatureTable.read_parquet(os.path.join(data_dir, "test_parquet"))

    # get cat sizes
    with tempfile.TemporaryDirectory() as local_path:
        get_remote_file_to_local(os.path.join(data_dir, "meta/categorical_sizes.pkl"),
                                 os.path.join(local_path, "categorical_sizes.pkl"))
        with open(os.path.join(local_path, "categorical_sizes.pkl"), 'rb') as f:
            cat_sizes_dict = SafePickle.load(f)
        get_remote_file_to_local(os.path.join(data_dir, "meta/cross_sizes.pkl"),
                                 os.path.join(local_path, "cross_sizes.pkl"))
        with open(os.path.join(local_path, "cross_sizes.pkl"), 'rb') as f:
            cross_sizes_dict = SafePickle.load(f)

    wide_cols = [col for col in CAT_COLS if cat_sizes_dict[col] <= 10]
    wide_dims = [cat_sizes_dict[col] for col in wide_cols]
    cross_cols = list(cross_sizes_dict.keys())
    cross_dims = [cross_sizes_dict[col] for col in cross_cols]
    embedding_cols = [col for col in CAT_COLS if cat_sizes_dict[col] >= 100]
    embedding_dims = [cat_sizes_dict[col] for col in embedding_cols]
    indicator_cols = list(set(CAT_COLS) - set(wide_cols) - set(embedding_cols))
    indicator_dims = [cat_sizes_dict[col] for col in indicator_cols]

    column_info = {"wide_base_cols": wide_cols,
                   "wide_base_dims": wide_dims,
                   "wide_cross_cols": cross_cols,
                   "wide_cross_dims": cross_dims,
                   "indicator_cols": indicator_cols,
                   "indicator_dims": indicator_dims,
                   "continuous_cols": INT_COLS,
                   "embed_cols": embedding_cols,
                   "embed_in_dims": embedding_dims,
                   "embed_out_dims": [16] * len(embedding_cols),
                   "label": "c0"}

    return train_tbl, test_tbl, column_info


def build_model(column_info, hidden_units=[100, 50, 25]):
    """Build an estimator appropriate for the given model type."""
    import tensorflow as tf

    wide_base_input_layers = []
    wide_base_layers = []
    for i in range(len(column_info["wide_base_cols"])):
        wide_base_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        wide_base_layers.append(tf.keras.backend.one_hot(
            wide_base_input_layers[i], column_info["wide_base_dims"][i] + 1))

    wide_cross_input_layers = []
    wide_cross_layers = []
    for i in range(len(column_info["wide_cross_cols"])):
        wide_cross_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        wide_cross_layers.append(tf.keras.backend.one_hot(
            wide_cross_input_layers[i], column_info["wide_cross_dims"][i]))

    indicator_input_layers = []
    indicator_layers = []
    for i in range(len(column_info["indicator_cols"])):
        indicator_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        indicator_layers.append(tf.keras.backend.one_hot(
            indicator_input_layers[i], column_info["indicator_dims"][i] + 1))

    embed_input_layers = []
    embed_layers = []
    for i in range(len(column_info["embed_in_dims"])):
        embed_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        iembed = tf.keras.layers.Embedding(
            column_info["embed_in_dims"][i] + 1,
            output_dim=column_info["embed_out_dims"][i])(embed_input_layers[i])
        flat_embed = tf.keras.layers.Flatten()(iembed)
        embed_layers.append(flat_embed)

    continuous_input_layers = []
    continuous_layers = []
    for i in range(len(column_info["continuous_cols"])):
        continuous_input_layers.append(tf.keras.layers.Input(shape=[]))
        continuous_layers.append(
            tf.keras.layers.Reshape(target_shape=(1,))(continuous_input_layers[i]))

    if len(wide_base_layers + wide_cross_layers) > 1:
        wide_input = tf.keras.layers.concatenate(
            wide_base_layers + wide_cross_layers, axis=1)
    else:
        wide_input = (wide_base_layers + wide_cross_layers)[0]
    wide_out = tf.keras.layers.Dense(1)(wide_input)
    if len(indicator_layers + embed_layers + continuous_layers) > 1:
        deep_concat = tf.keras.layers.concatenate(indicator_layers +
                                                  embed_layers +
                                                  continuous_layers, axis=1)
    else:
        deep_concat = (indicator_layers + embed_layers + continuous_layers)[0]
    linear = deep_concat
    for ilayer in range(0, len(hidden_units)):
        linear_mid = tf.keras.layers.Dense(hidden_units[ilayer])(linear)
        bn = tf.keras.layers.BatchNormalization()(linear_mid)
        relu = tf.keras.layers.ReLU()(bn)
        dropout = tf.keras.layers.Dropout(0.1)(relu)
        linear = dropout
    deep_out = tf.keras.layers.Dense(1)(linear)
    added = tf.keras.layers.add([wide_out, deep_out])
    out = tf.keras.layers.Activation("sigmoid")(added)
    model = tf.keras.models.Model(wide_base_input_layers +
                                  wide_cross_input_layers +
                                  indicator_input_layers +
                                  embed_input_layers +
                                  continuous_input_layers,
                                  out)

    return model


def model_creator(config):
    import tensorflow as tf

    model = build_model(column_info=config["column_info"],
                        hidden_units=config['hidden_units'])
    optimizer = tf.keras.optimizers.Adam(config["lr"])
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'binary_crossentropy', 'AUC'])
    return model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=48,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="30g",
                        help='The executor memory.')
    parser.add_argument('--num_executors', type=int, default=8,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument("--data_dir", type=str, required=True, dest="data_dir",
                        help="The path to the folder of preprocessed parquet files and meta data")
    parser.add_argument("--model_dir", type=str, default="./wnd_model",
                        help="The path to saved the trained model")
    parser.add_argument("--batch_size", "-b", type=int, default=1024,
                        help="The batch size to train the model.")
    parser.add_argument("--epochs", "-e", type=int, default=2,
                        help="The number of epochs to train the model.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="The learning rate to train the model.")

    args = parser.parse_args()

    if args.cluster_mode == "local":
        init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory,
                          init_ray_on_spark=True)
    elif args.cluster_mode == "standalone":
        init_orca_context("standalone", master=args.master,
                          cores=args.executor_cores, num_nodes=args.num_executors,
                          memory=args.executor_memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          init_ray_on_spark=True)
    elif args.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=args.executor_cores,
                          num_nodes=args.num_executors, memory=args.executor_memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          init_ray_on_spark=True)
    elif args.cluster_mode == "spark-submit":
        init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)

    train_tbl, test_tbl, column_info = get_data(args.data_dir)
    feature_cols = \
        column_info["wide_base_cols"] + column_info["wide_cross_cols"] + \
        column_info["indicator_cols"] + column_info["embed_cols"] + column_info["continuous_cols"]
    label_cols = [column_info["label"]]

    config = {
        "lr": args.lr,
        "hidden_units": [1024, 512],
        "column_info": column_info,
    }

    est = Estimator.from_keras(
        model_creator=model_creator,
        verbose=True,
        config=config,
        backend="ray")

    train_count = train_tbl.size()
    print("train size: ", train_count)
    steps = math.ceil(train_count / args.batch_size)
    test_count = test_tbl.size()
    print("test size: ", test_count)
    val_steps = math.ceil(test_count / args.batch_size)

    if not exists(args.model_dir):
        makedirs(args.model_dir)

    start = time()
    est.fit(data=train_tbl.df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=steps,
            validation_data=test_tbl.df,
            validation_steps=val_steps,
            feature_cols=feature_cols,
            label_cols=label_cols)
    end = time()
    print("Training time is: ", end - start)
    est.save(os.path.join(args.model_dir, "model-%d.ckpt" % args.epochs))
    est.save_weights(os.path.join(args.model_dir, "model.h5"))

    stop_orca_context()
