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

import os

from argparse import ArgumentParser
import tempfile
import pickle
from time import time

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.file import makedirs, exists
from bigdl.dllib.utils.file_utils import put_local_file_to_remote
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.utils.log4Error import *


bool_cols = [
    'engaged_with_user_is_verified',
    'enaging_user_is_verified'
]

count_cols = [
    'engaged_with_user_follower_count',
    'engaged_with_user_following_count',
    'enaging_user_follower_count',
    'enaging_user_following_count'
]

timestamp_cols = [
    'reply_timestamp',
    'retweet_timestamp',
    'retweet_with_comment_timestamp',
    'like_timestamp'
]

cat_cols = [
    'engaged_with_user_id',
    'enaging_user_id',
    'present_media',
    'tweet_type',
    'language'
]

list_cols = [
    "hashtags",
    "present_domains",
    "present_links"
]

len_cols = ['len_hashtags',
            'len_domains',
            'len_links']


media_map = {
    '': 0,
    'GIF': 1,
    'GIF_GIF': 2,
    'GIF_Photo': 3,
    'GIF_Video': 4,
    'Photo': 5,
    'Photo_GIF': 6,
    'Photo_Photo': 7,
    'Photo_Video': 8,
    'Video': 9,
    'Video_GIF': 10,
    'Video_Photo': 11,
    'Video_Video': 12
}

type_map = {
    'Quote': 0,
    'Retweet': 1,
    'TopLevel': 2,
}

cross_cols = [['present_media', 'language']]

conf = {"spark.network.timeout": "10000000",
        "spark.sql.broadcastTimeout": "7200",
        "spark.sql.shuffle.partitions": "2000",
        "spark.locality.wait": "0s",
        "spark.sql.crossJoin.enabled": "true",
        "spark.task.cpus": "1",
        "spark.executor.heartbeatInterval": "200s",
        "spark.driver.maxResultSize": "40G",
        "spark.eventLog.enabled": "true",
        "spark.eventLog.dir": "hdfs://172.16.0.105:8020/sparkHistoryLogs",
        "spark.app.name": "recsys-preprocess",
        "spark.executor.memoryOverhead": "20g"}


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=44,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="130",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=8,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--train_files', type=str, default="all",
                        help="range for preprocessing train files, such as 000-269, 000-001.")
    parser.add_argument('--input_train_folder', type=str, required=True,
                        help="Path to the folder of train parquet files.")
    parser.add_argument('--input_test_folder', type=str, required=True,
                        help="Path to the folder of test parquet files.")
    parser.add_argument('--output_folder', type=str, default=".",
                        help="The path to save the preprocessed data to parquet files. ")
    parser.add_argument('--cross_sizes', type=str,
                        help='bucket sizes for cross columns', default="600")

    args = parser.parse_args()
    if args.train_files != "all":
        start, end = args.train_files.split('-')
        args.train_files = list(range(int(start), int(end) + 1))

    args.cross_sizes = [int(x) for x in args.cross_sizes.split(',')]

    return args


def preprocess(tbl):
    tbl = tbl.cast(bool_cols + count_cols + timestamp_cols, "int")\
        .fillna(0, timestamp_cols)\
        .fillna("", ["present_media", "present_domains", "present_links", "hashtags"])

    process_media = lambda x: '_'.join(x.split('\t')[:2])
    tbl = tbl.apply("present_media", "present_media", process_media, "string")\
        .encode_string("present_media", media_map)\
        .encode_string("tweet_type", type_map)

    count_func = lambda x: str(x).count('\t') + 1 if x else 0
    tbl = tbl.apply("hashtags", "len_hashtags", count_func, "int") \
        .apply("present_domains", "len_domains", count_func, "int") \
        .apply("present_links", "len_links", count_func, "int")

    return tbl


def encode_user_id(tbl):
    tbl = tbl.rename({"engaged_with_user_id": "user_id"}) \
        .encode_string("user_id", user_index, broadcast=False) \
        .rename({"user_id": "engaged_with_user_id"})\
        .fillna(0, "engaged_with_user_id")
    tbl = tbl.rename({"enaging_user_id": "user_id"}) \
        .encode_string("user_id", user_index, broadcast=False) \
        .rename({"user_id": "enaging_user_id"})\
        .fillna(0, "enaging_user_id")
    return tbl


def generate_features(tbl, bins, cross_sizes):
    tbl = tbl.cut_bins(columns=count_cols,
                       bins=bins,
                       out_cols=count_cols)

    tbl = tbl.cross_columns(cross_cols, cross_sizes)
    return tbl


def transform_label(tbl):
    gen_label = lambda x: 1 if max(x) > 0 else 0
    tbl = tbl.apply(in_col=timestamp_cols, out_col="label", func=gen_label, dtype="int")
    return tbl


if __name__ == '__main__':
    args = _parse_args()
    if args.cluster_mode == "local":
        init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        init_orca_context("standalone", master=args.master,
                          cores=args.executor_cores, num_nodes=args.num_executor,
                          memory=args.executor_memory,
                          driver_cores=args.driver_cores,
                          driver_memory=args.driver_memory, conf=conf)
    elif args.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=args.executor_cores,
                          num_nodes=args.num_executor, memory=args.executor_memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          conf=conf)
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn' and 'standalone'"
                          ", but got " + args.cluster_mode)

    start = time()
    if args.train_files != "all":
        train_paths = [os.path.join(args.input_train_folder, 'part-%05d*.parquet' % i)
                       for i in args.train_files]
        train_tbl = FeatureTable.read_parquet(train_paths)
    else:
        train_tbl = FeatureTable.read_parquet(args.input_train_folder)
    train_tbl.df.printSchema()

    test_tbl = FeatureTable.read_parquet(args.input_test_folder)

    train_tbl = preprocess(train_tbl)
    test_tbl = preprocess(test_tbl)

    train_tbl, language_idx = train_tbl.category_encode("language")
    test_tbl = test_tbl.encode_string("language", language_idx)

    user_index = train_tbl.gen_string_idx({'src_cols': ['engaged_with_user_id', 'enaging_user_id'],
                                           'col_name': 'user_id'})
    train_tbl = encode_user_id(train_tbl)
    test_tbl = encode_user_id(test_tbl)
    test_tbl = test_tbl.fillna(0, ["engaged_with_user_id", "enaging_user_id"])

    item_index = train_tbl.concat(test_tbl).gen_string_idx("tweet_id")
    train_tbl = train_tbl.encode_string("tweet_id", item_index).fillna(0, "tweet_id")
    test_tbl = test_tbl.encode_string("tweet_id", item_index).fillna(0, "tweet_id")

    indexes = train_tbl.gen_string_idx(list_cols, do_split=True, sep='\t')
    train_tbl = train_tbl.encode_string(list_cols, indexes,
                                        do_split=True, sep='\t', keep_most_frequent=True)
    test_tbl = test_tbl.encode_string(list_cols, indexes,
                                      do_split=True, sep='\t', keep_most_frequent=True)
    test_tbl = test_tbl.fillna(0, list_cols)

    train_tbl.cache()

    test_tbl.cache()

    bins = [1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    train_tbl = generate_features(train_tbl, bins, args.cross_sizes)
    test_tbl = generate_features(test_tbl, bins, args.cross_sizes)

    train_tbl, min_max_dict = train_tbl.min_max_scale(len_cols)
    test_tbl = test_tbl.transform_min_max_scale(len_cols, min_max_dict)

    train_tbl = transform_label(train_tbl)
    test_tbl = transform_label(test_tbl)

    # save preprocessed data
    train_tbl.write_parquet(os.path.join(args.output_folder, "train_parquet"))
    test_tbl.write_parquet(os.path.join(args.output_folder, "test_parquet"))

    # save meta
    cat_sizes_dict = {}
    cat_sizes_dict['present_media'] = len(media_map)
    cat_sizes_dict['tweet_type'] = len(type_map)
    cat_sizes_dict['language'] = language_idx.size()
    for i in range(len(list_cols)):
        cat_sizes_dict[list_cols[i]] = indexes[i].size()
    cat_sizes_dict['engaged_with_user_id'] = user_index.size()
    cat_sizes_dict['enaging_user_id'] = user_index.size()

    cross_sizes_dict = dict(zip(["_".join(cross_names) for cross_names in cross_cols],
                                args.cross_sizes))

    cat_sizes_dict.update(cross_sizes_dict)

    count_sizes_dict = dict(zip(count_cols, [len(bins)] * len(count_cols)))
    cat_sizes_dict.update(count_sizes_dict)
    print("cat size dict: ", cat_sizes_dict)

    if not exists(os.path.join(args.output_folder, "meta")):
        makedirs(os.path.join(args.output_folder, "meta"))

    with tempfile.TemporaryDirectory() as local_path:
        with open(os.path.join(local_path, "categorical_sizes.pkl"), 'wb') as f:
            pickle.dump(cat_sizes_dict, f)
        put_local_file_to_remote(os.path.join(local_path, "categorical_sizes.pkl"),
                                 os.path.join(args.output_folder, "meta/categorical_sizes.pkl"),
                                 over_write=True)

    end = time()
    print("Preprocessing and save time: ", end - start)

    stop_orca_context()
