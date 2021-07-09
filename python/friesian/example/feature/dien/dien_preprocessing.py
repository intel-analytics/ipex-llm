#
# Copyright 2018 Analytics Zoo Authors.
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
import os
import time
from pyspark import StorageLevel
from zoo.orca import init_orca_context, stop_orca_context
from pyspark.sql.functions import udf, col
from zoo.friesian.feature import FeatureTable
from pyspark.sql.types import StringType, ArrayType, FloatType
from argparse import ArgumentParser

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
        "spark.driver.maxResultSize": "40G"}


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=48,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=8,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--input_transaction', type=str, required=True,
                        help="transaction files.")
    parser.add_argument('--input_meta', type=str, required=True,
                        help="item metadata file")
    parser.add_argument('--output')
    parser.add_argument(
        '--write_mode',
        choices=['overwrite', 'errorifexists'],
        default='overwrite')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
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

    begin = time.time()
    transaction_tbl = FeatureTable.read_json(args.input_transaction).select(
        ['reviewerID', 'asin', 'unixReviewTime']) \
        .rename({'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'time'}) \
        .dropna(columns=['user', 'item'])
    print("transaction_tbl, ", transaction_tbl.size())

    # read meta data
    def get_category(x):
        cat = x[0][-1] if x[0][-1] is not None else "default"
        return cat.strip().lower()
    trans_label = lambda x: [1 - float(x), float(x)]

    item_tbl = FeatureTable.read_json(args.input_meta).select(['asin', 'categories'])\
        .dropna(columns=['asin', 'categories']) \
        .apply("categories", "category", get_category, "string") \
        .rename({"asin": "item"}).drop("categories").distinct()
    print("item_tbl, ", item_tbl.size())

    user_index = transaction_tbl.gen_string_idx('user', 1)
    item_category_indices = item_tbl.gen_string_idx(["item", "category"], 1)
    item_size = item_category_indices[0].size()
    category_index = item_category_indices[1]

    item_tbl = item_tbl\
        .encode_string(["item", "category"], [item_category_indices[0], category_index])\
        .distinct()

    transaction_tbl = transaction_tbl\
        .encode_string(['user', 'item'], [user_index, item_category_indices[0]])\
        .dropna(columns="item")\
        .add_hist_seq(cols=['item'], user_col="user",
                      sort_col='time', min_len=1, max_len=100)\
        .add_neg_hist_seq(item_size, 'item_hist_seq', neg_num=5)\
        .add_negative_samples(item_size, item_col='item', neg_num=1)

    full_tbl = transaction_tbl.join(item_tbl, "item")\
        .add_value_features(key_cols=["item_hist_seq", "neg_item_hist_seq"],
                            tbl=item_tbl, key="item", value="category")\
        .pad(cols=['item_hist_seq', 'category_hist_seq',
             'neg_item_hist_seq', 'neg_category_hist_seq'],
             seq_len=100,
             mask_cols=['item_hist_seq']) \
        .apply("item_hist_seq", "item_hist_seq_len", len, "int") \
        .apply("label", "label", trans_label, "array<float>")

    # write out
    user_index.write_parquet(args.output)
    item_category_indices[0].write_parquet(args.output + "item_index")
    category_index.write_parquet(args.output + "category_index")
    item_tbl.write_parquet(args.output + "item2cat")
    full_tbl.write_parquet(args.output + "data")

    print("final output count, ", full_tbl.size())
    stop_orca_context()
    end = time.time()
    print(f"perf preprocessing time: {(end - begin):.2f}s")
