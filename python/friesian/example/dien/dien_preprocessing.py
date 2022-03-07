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

import time
from argparse import ArgumentParser

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable, StringIndex


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
                        help='The cluster mode, such as local, yarn, standalone or spark-submit.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=48,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executors', type=int, default=8,
                        help='The number of executors.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--input_transaction', type=str, required=True,
                        help="transaction files.")
    parser.add_argument('--input_meta', type=str, required=True,
                        help="item metadata file")
    parser.add_argument('--output', default="./")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    if args.cluster_mode == "local":
        init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        init_orca_context("standalone", master=args.master,
                          cores=args.executor_cores, num_nodes=args.num_executors,
                          memory=args.executor_memory,
                          driver_cores=args.driver_cores,
                          driver_memory=args.driver_memory, conf=conf)
    elif args.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=args.executor_cores,
                          num_nodes=args.num_executors, memory=args.executor_memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          conf=conf)
    elif args.cluster_mode == "spark-submit":
        init_orca_context("spark-submit")
    else:
        raise ValueError(
            "cluster_mode should be one of 'local', 'yarn', 'standalone' and 'spark-submit'"
            ", but got " + args.cluster_mode)

    begin = time.time()
    transaction_tbl = FeatureTable.read_json(args.input_transaction).select(
        ['reviewerID', 'asin', 'unixReviewTime']) \
        .rename({'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'time'}) \
        .dropna(columns=['user', 'item'])
    transaction_tbl.cache()

    item_tbl = FeatureTable.read_csv(args.input_meta, delimiter="\t", names=['item', 'category'])\
        .apply("category", "category", lambda x: x.lower() if x is not None else "default")
    item_tbl.cache()

    user_index = StringIndex.read_parquet(args.output + "user.parquet")
    item_index = StringIndex.read_parquet(args.output + "item.parquet")
    category_index = StringIndex.read_parquet(args.output + "category.parquet")
    item_size = item_index.size()

    item_tbl = item_tbl\
        .encode_string(["item", "category"], [item_index, category_index])

    full_tbl = transaction_tbl\
        .encode_string(['item'], [item_index])\
        .add_hist_seq(cols=['item'], user_col="user",
                      sort_col='time', min_len=2, max_len=100, num_seqs=1)\
        .add_value_features(columns=["item", "item_hist_seq"],
                            dict_tbl=item_tbl, key="item", value="category")\
        .encode_string(['user'], [user_index])\
        .add_negative_samples(item_size, item_col='item', neg_num=1)

    # write out
    item_tbl.write_parquet(args.output + "item2cat")
    full_tbl.write_parquet(args.output + "data")

    end = time.time()
    print(f"DIEN preprocessing time: {(end - begin):.2f}s")
    full_tbl.show(5)
    stop_orca_context()
