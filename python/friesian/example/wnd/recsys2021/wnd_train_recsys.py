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

import sys
import os
import math
from optparse import OptionParser
from time import time
import tempfile
import pickle

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.file import exists, makedirs
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.utils.file_utils import get_remote_file_to_local
from bigdl.orca.learn.tf2.estimator import Estimator

import tensorflow as tf
from bigdl.dllib.utils.log4Error import *


wide_cols = ['engaged_with_user_is_verified', 'enaging_user_is_verified']
wide_dims = [1, 1]
cross_cols = ['present_media_language']

cat_cols = ['present_media',
            'tweet_type',
            'language']

count_cols = ['engaged_with_user_follower_count',
              'engaged_with_user_following_count',
              'enaging_user_follower_count',
              'enaging_user_following_count']

indicator_cols = cat_cols + count_cols
# indicator_dims = cat_dims + count_dims

embedding_cols = [
    'engaged_with_user_id',
    'enaging_user_id',
    'hashtags',
    'present_domains',  #
]

len_cols = ['len_hashtags',
            'len_domains',
            'len_links']

conf = {"spark.network.timeout": "10000000",
        "spark.sql.broadcastTimeout": "7200",
        "spark.sql.shuffle.partitions": "2000",
        "spark.locality.wait": "0s",
        "spark.sql.hive.filesourcePartitionFileCacheSize": "4096000000",
        "spark.sql.crossJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryo.unsafe": "true",
        "spark.kryoserializer.buffer.max": "1024m",
        "spark.task.cpus": "1",
        "spark.executor.heartbeatInterval": "200s",
        "spark.driver.maxResultSize": "40G",
        "spark.app.name": "recsys-train",
        "spark.driver.memoryOverhead": "100G",
        "spark.executor.memoryOverhead": "100G"}


def get_size(data_dir):
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
            cat_sizes_dic = pickle.load(f)

    indicator_sizes = [cat_sizes_dic[c] for c in indicator_cols]
    print("indicator sizes: ", indicator_sizes)
    embedding_sizes = [cat_sizes_dic[c] for c in embedding_cols]
    print("embedding sizes: ", embedding_sizes)
    cross_sizes = [cat_sizes_dic[c] for c in cross_cols]

    return train_tbl, test_tbl, indicator_sizes, embedding_sizes, cross_sizes


def build_model(column_info, hidden_units=[100, 50, 25]):
    """Build an estimator appropriate for the given model type."""
    wide_base_input_layers = []
    wide_base_layers = []
    for i in range(len(column_info.wide_base_cols)):
        wide_base_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        wide_base_layers.append(tf.keras.backend.one_hot(wide_base_input_layers[i],
                                                         column_info.wide_base_dims[i] + 1))

    wide_cross_input_layers = []
    wide_cross_layers = []
    for i in range(len(column_info.wide_cross_cols)):
        wide_cross_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        wide_cross_layers.append(tf.keras.backend.one_hot(wide_cross_input_layers[i],
                                                          column_info.wide_cross_dims[i]))

    indicator_input_layers = []
    indicator_layers = []
    for i in range(len(column_info.indicator_cols)):
        indicator_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        indicator_layers.append(tf.keras.backend.one_hot(indicator_input_layers[i],
                                                         column_info.indicator_dims[i] + 1))

    embed_input_layers = []
    embed_layers = []
    for i in range(len(column_info.embed_in_dims)):
        embed_input_layers.append(tf.keras.layers.Input(shape=[], dtype="int32"))
        embedding_layer = tf.keras.layers.Embedding(column_info.embed_in_dims[i] + 1,
                                                    output_dim=column_info.embed_out_dims[i])
        iembed = embedding_layer(embed_input_layers[i])
        flat_embed = tf.keras.layers.Flatten()(iembed)
        embed_layers.append(flat_embed)

    continuous_input_layers = []
    continuous_layers = []
    for i in range(len(column_info.continuous_cols)):
        continuous_input_layers.append(tf.keras.layers.Input(shape=[]))
        continuous_layers.append(
            tf.keras.layers.Reshape(target_shape=(1,))(continuous_input_layers[i]))

    if len(wide_base_layers + wide_cross_layers) > 1:
        wide_input = tf.keras.layers.concatenate(wide_base_layers + wide_cross_layers, axis=1)
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
    model = build_model(column_info=config["column_info"],
                        hidden_units=config['hidden_units'])
    optimizer = tf.keras.optimizers.Adam(config["lr"])
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy', 'binary_crossentropy', 'AUC', 'Precision', 'Recall'])
    return model


class ColumnFeatureInfo(object):
    """
    The same data information shared by the WideAndDeep model and its feature generation part.

    Each instance could contain the following fields:
    wide_base_cols: Data of wide_base_cols together with wide_cross_cols will be fed
                    into the wide model. List of String. Default is an empty list.
    wide_base_dims: Dimensions of wide_base_cols. The dimensions of the data in
                    wide_base_cols should be within the range of wide_base_dims.
                    List of int. Default is an empty list.
    wide_cross_cols: Data of wide_cross_cols will be fed into the wide model.
                     List of String. Default is an empty list.
    wide_cross_dims: Dimensions of wide_cross_cols. The dimensions of the data in
                     wide_cross_cols should be within the range of wide_cross_dims.
                     List of int. Default is an empty list.
    indicator_cols: Data of indicator_cols will be fed into the deep model as multi-hot vectors.
                    List of String. Default is an empty list.
    indicator_dims: Dimensions of indicator_cols. The dimensions of the data in
                    indicator_cols should be within the range of indicator_dims.
                    List of int. Default is an empty list.
    embed_cols: Data of embed_cols will be fed into the deep model as embeddings.
                List of String. Default is an empty list.
    embed_in_dims: Input dimension of the data in embed_cols. The dimensions of the data in
                   embed_cols should be within the range of embed_in_dims.
                   List of int. Default is an empty list.
    embed_out_dims: The dimensions of embeddings. List of int. Default is an empty list.
    continuous_cols: Data of continuous_cols will be treated as continuous values for
                     the deep model. List of String. Default is an empty list.
    label: The name of the 'label' column. String. Default is 'label'.
    """

    def __init__(self, wide_base_cols=None, wide_base_dims=None, wide_cross_cols=None,
                 wide_cross_dims=None, indicator_cols=None, indicator_dims=None,
                 embed_cols=None, embed_in_dims=None, embed_out_dims=None,
                 continuous_cols=None, label="label", bigdl_type="float"):
        self.wide_base_cols = [] if not wide_base_cols else wide_base_cols
        self.wide_base_dims = [] if not wide_base_dims else [int(d) for d in wide_base_dims]
        self.wide_cross_cols = [] if not wide_cross_cols else wide_cross_cols
        self.wide_cross_dims = [] if not wide_cross_dims else [int(d) for d in wide_cross_dims]
        self.indicator_cols = [] if not indicator_cols else indicator_cols
        self.indicator_dims = [] if not indicator_dims else [int(d) for d in indicator_dims]
        self.embed_cols = [] if not embed_cols else embed_cols
        self.embed_in_dims = [] if not embed_in_dims else [int(d) for d in embed_in_dims]
        self.embed_out_dims = [] if not embed_out_dims else [int(d) for d in embed_out_dims]
        self.continuous_cols = [] if not continuous_cols else continuous_cols
        self.label = label
        self.bigdl_type = bigdl_type

    def __reduce__(self):
        return ColumnFeatureInfo, (self.wide_base_cols, self.wide_base_dims, self.wide_cross_cols,
                                   self.wide_cross_dims, self.indicator_cols, self.indicator_dims,
                                   self.embed_cols, self.embed_in_dims, self.embed_out_dims,
                                   self.continuous_cols, self.label)

    def __str__(self):
        return "ColumnFeatureInfo {wide_base_cols: %s, wide_base_dims: %s, wide_cross_cols: %s, " \
               "wide_cross_dims: %s, indicator_cols: %s, indicator_dims: %s, embed_cols: %s, " \
               "embed_cols: %s, embed_in_dims: %s, embed_out_dims: %s, continuous_cols: %s, " \
               "label: '%s'}" \
               % (self.wide_base_cols, self.wide_base_dims, self.wide_cross_cols,
                  self.wide_cross_dims, self.indicator_cols, self.indicator_dims,
                  self.embed_cols, self.embed_cols, self.embed_in_dims,
                  self.embed_out_dims, self.continuous_cols, self.label)

    @property
    def feature_cols(self):
        return self.wide_base_cols + self.wide_cross_cols +\
            self.indicator_cols + self.embed_cols + self.continuous_cols

    @property
    def label_cols(self):
        return [self.label]


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('--cluster_mode', type=str, default="local",
                      help='The cluster mode, such as local, yarn or standalone.')
    parser.add_option('--backend', type=str, default="ray",
                      help='The backend of Orca Estimator, either ray or spark.')
    parser.add_option('--master', type=str, default=None,
                      help='The master url, only used when cluster mode is standalone.')
    parser.add_option('--executor_cores', type=int, default=44,
                      help='The executor core number.')
    parser.add_option('--executor_memory', type=str, default="30g",
                      help='The executor memory.')
    parser.add_option('--num_executor', type=int, default=8,
                      help='The number of executor.')
    parser.add_option('--driver_cores', type=int, default=4,
                      help='The driver core number.')
    parser.add_option('--driver_memory', type=str, default="36g",
                      help='The driver memory.')
    parser.add_option("--data_dir", dest="data_dir")
    parser.add_option("--model_dir", dest="model_dir")
    parser.add_option("--batch_size", "-b", dest="batch_size", default=102400, type=int)
    parser.add_option("--epoch", "-e", dest="epochs", default=2, type=int)
    parser.add_option("--learning_rate", "-l", dest="learning_rate", default=1e-4, type=float)
    parser.add_option('--early_stopping', type=int, default=3, dest="early_stopping")
    parser.add_option('--hidden_units', dest="hidden_units", type=str,
                      help='hidden units for deep mlp', default="1024, 1024")

    (options, args) = parser.parse_args(sys.argv)
    options.hidden_units = [int(x) for x in options.hidden_units.split(',')]

    if options.cluster_mode == "local":
        init_orca_context("local", cores=options.executor_cores, memory=options.executor_memory)
    elif options.cluster_mode == "standalone":
        init_orca_context("standalone", master=options.master,
                          cores=options.executor_cores, num_nodes=options.num_executor,
                          memory=options.executor_memory,
                          driver_cores=options.driver_cores, driver_memory=options.driver_memory,
                          conf=conf)
    elif options.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=options.executor_cores,
                          num_nodes=options.num_executor, memory=options.executor_memory,
                          driver_cores=options.driver_cores, driver_memory=options.driver_memory,
                          conf=conf)
    elif options.cluster_mode == "spark-submit":
        init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)

    train_tbl, test_tbl, indicator_sizes, embedding_sizes, cross_sizes = get_size(options.data_dir)

    column_info = ColumnFeatureInfo(wide_base_cols=wide_cols,
                                    wide_base_dims=wide_dims,
                                    wide_cross_cols=cross_cols,
                                    wide_cross_dims=cross_sizes,
                                    indicator_cols=indicator_cols,
                                    indicator_dims=indicator_sizes,
                                    embed_cols=embedding_cols,
                                    embed_in_dims=embedding_sizes,
                                    embed_out_dims=[8] * len(embedding_cols),
                                    continuous_cols=len_cols,
                                    label="label"
                                    )

    config = {
        "lr": options.learning_rate,
        "hidden_units": options.hidden_units,
        "column_info": column_info,
        "inter_op_parallelism": 1,
        "intra_op_parallelism": 40
    }

    est = Estimator.from_keras(
        model_creator=model_creator,
        verbose=True,
        config=config,
        backend=options.backend)

    train_count = train_tbl.size()
    print("train size: ", train_count)
    steps = math.ceil(train_count / options.batch_size)
    test_count = test_tbl.size()
    print("test size: ", test_count)
    val_steps = math.ceil(test_count / options.batch_size)

    if not exists(options.model_dir):
        makedirs(options.model_dir)

    callbacks = []

    # early stopping
    earlystopping = options.early_stopping
    if earlystopping:
        from tensorflow.keras.callbacks import EarlyStopping

        callbacks.append(EarlyStopping(monitor='val_auc', mode='max',
                                       verbose=1, patience=earlystopping))

    start = time()
    est.fit(data=train_tbl.df,
            epochs=options.epochs,
            batch_size=options.batch_size,
            callbacks=callbacks,
            steps_per_epoch=steps,
            validation_data=test_tbl.df,
            validation_steps=val_steps,
            feature_cols=column_info.feature_cols,
            label_cols=column_info.label_cols)
    end = time()
    print("Training time is: ", end - start)
    est.save(os.path.join(options.model_dir, "model-%d.ckpt" % options.epochs))
    model = est.get_model()
    model.save_weights(os.path.join(options.model_dir, "model.h5"))

    stop_orca_context()
