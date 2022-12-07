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

import math
from time import time
from argparse import ArgumentParser

from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.tf2.estimator import Estimator
from bigdl.friesian.feature import FeatureTable

from deepctr.feature_column import SparseFeat, DenseFeat
from deepctr.models import MMOE, PLE

from keras.callbacks import EarlyStopping


def build_model(model_type, sparse_features, dense_features, feature_max_idx):
    sparse_feature_columns = [SparseFeat(feat, feature_max_idx[feat],
                                         embedding_dim='auto') for feat in sparse_features]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
    dnn_features_columns = sparse_feature_columns + dense_feature_columns
    if model_type == 'mmoe':
        model = MMOE(dnn_features_columns, tower_dnn_hidden_units=[],
                     task_types=['regression', 'binary'],
                     task_names=['duration', 'click'])
    elif model_type == 'ple':
        model = PLE(dnn_features_columns, shared_expert_num=1, specific_expert_num=1,
                    task_types=['regression', 'binary'],
                    num_levels=2, task_names=['duration', 'click'])
    else:
        invalidInputError(False, 'model_type should be one of "mmoe" and "ple", '
                                 'but got ' + model_type)
    return model


def model_creator(config):
    model = build_model(model_type=config['model_type'],
                        sparse_features=config['column_info']['cat_cols'],
                        dense_features=config['column_info']['continuous_cols'],
                        feature_max_idx=config['column_info']['feature_max_idx'])
    model.compile(optimizer='adam',
                  loss=["mean_squared_error", "binary_crossentropy"],
                  metrics=[['mae'], ["AUC", 'Precision', 'Recall']])
    return model


def label_cols(column_info):
    return column_info["label"]


def feature_cols(column_info):
    return column_info["cat_cols"] + column_info["embed_cols"] + column_info["continuous_cols"]


def train_multi_task(train_tbl_data, valid_tbl_data, save_path, model,
                     cat_cols, continuous_cols, feature_max_idx):
    column_info = {
        "cat_cols": cat_cols,
        "continuous_cols": continuous_cols,
        "feature_max_idx": feature_max_idx,
        "embed_cols": [],
        "embed_in_dims": [],
        "embed_out_dims": [],
        "label": ['duration', 'click']}

    config = {
        "column_info": column_info,
        "inter_op_parallelism": 4,
        "intra_op_parallelism": 8,
        "model_type": model  # mmoe or ple
    }

    batch_size = 256
    estimator = Estimator.from_keras(
        model_creator=model_creator,
        verbose=False,
        config=config)

    train_count = train_tbl_data.size()
    print("Total number of train records: {}".format(train_count))
    total_steps = math.ceil(train_count / batch_size)
    steps_per_epoch = 50
    # To train the full dataset for an entire epoch
    epochs = math.ceil(total_steps / steps_per_epoch)
    val_count = valid_tbl_data.size()
    print("Total number of val records: {}".format(val_count))
    val_steps = math.ceil(val_count / batch_size)
    callbacks = [EarlyStopping(monitor='val_duration_mae', mode='min', verbose=1, patience=3),
                 EarlyStopping(monitor='val_click_auc', mode='max', verbose=1, patience=3)]

    start = time()
    estimator.fit(data=train_tbl_data.df,
                  epochs=epochs,
                  batch_size=batch_size,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=valid_tbl_data.df,
                  validation_steps=val_steps,
                  callbacks=callbacks,
                  feature_cols=feature_cols(column_info),
                  label_cols=label_cols(column_info))
    end = time()
    print("Training time is: ", end - start)
    estimator.save(save_path)
    print('Save model to path: ', save_path)


def test_multi_task(valid_tbl_data, save_path, model, cat_cols, continuous_cols, feature_max_idx):
    column_info = {
        "cat_cols": cat_cols,
        "continuous_cols": continuous_cols,
        "feature_max_idx": feature_max_idx,
        "embed_cols": [],
        "embed_in_dims": [],
        "embed_out_dims": [],
        "label": ['duration', 'click']}
    config = {
        "column_info": column_info,
        "inter_op_parallelism": 4,
        "intra_op_parallelism": 8,
        "model_type": model  # mmoe or ple
    }
    estimator = Estimator.from_keras(
        model_creator=model_creator,
        verbose=False,
        config=config)
    estimator.load(save_path)

    batch_size = 256
    val_steps = math.ceil(valid_tbl_data.size() / batch_size)
    eval_results = estimator.evaluate(data=valid_tbl_data.df,
                                      num_steps=val_steps,
                                      batch_size=batch_size,
                                      feature_cols=feature_cols(column_info),
                                      label_cols=label_cols(column_info))
    for k, v in eval_results[0].items():
        print(k, v)


def _parse_args():
    parser = ArgumentParser(description="Set parameters for multi task demo")

    parser.add_argument('--model_type', type=str, default="mmoe",
                        help='The multi task model, mmoe or ple.')
    parser.add_argument('--train_data_path', type=str,
                        default='path/to/training/dataset',
                        help='The path for training dataset.')
    parser.add_argument('--test_data_path', type=str,
                        default='path/to/testing/dataset',
                        help='The path for testing dataset.')
    parser.add_argument('--model_save_path', type=str,
                        default='path/to/save/the/trained/model',
                        help='The path for saving the trained model.')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn, standalone or spark-submit.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="12g",
                        help='The executor memory.')
    parser.add_argument('--num_executors', type=int, default=4,
                        help='The number of executors.')
    parser.add_argument('--driver_cores', type=int, default=2,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="8g",
                        help='The driver memory.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    if args.cluster_mode == "local":  # For local machine
        sc = init_orca_context(cluster_mode="local",
                               cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executors,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores,
                               driver_memory=args.driver_memory)
    elif args.cluster_mode == "yarn":  # For Hadoop/YARN cluster
        sc = init_orca_context(cluster_mode="yarn", cores=args.executor_cores,
                               num_nodes=args.num_executors, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               object_store_memory="80g")
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)
    cat_cols = [
        'user_id',
        'article_id',
        'net_status',
        'exop_position',
        'device',
        'city',
        'age',
        'gender',
        'cat_1'
    ]
    continuous_cols = ['img_num', 'flush_nums']
    feature_max_idx = {'user_id': 40000, 'article_id': 200000, 'net_status': 1004,
                       'exop_position': 2000, 'device': 2000,
                       'city': 1379, 'age': 1005, 'gender': 1003, 'cat_1': 1038}

    # do train
    train_tbl = FeatureTable.read_parquet(args.train_data_path)
    valid_tbl = FeatureTable.read_parquet(args.test_data_path)
    train_multi_task(train_tbl, valid_tbl, args.model_save_path,
                     args.model_type, cat_cols, continuous_cols,
                     feature_max_idx)
    # do test
    test_multi_task(valid_tbl, args.model_save_path, args.model_type,
                    cat_cols, continuous_cols, feature_max_idx)

    stop_orca_context()
