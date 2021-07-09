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

from zoo.friesian.feature import FeatureTable

import os

from argparse import ArgumentParser
from time import time
from functools import reduce

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.data.file import makedirs, write_text, exists
from zoo import init_nncontext

LABEL_COL = 0
INT_COLS = ["c{}".format(i) for i in list(range(1, 14))]
CAT_COLS = ["c{}".format(i) for i in list(range(14, 40))]

CROSS_COLS = [CAT_COLS[0:2], CAT_COLS[2:4]]

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
    parser.add_argument('--days', type=str, required=True,
                        help="Day range for preprocessing, such as 0-23, 0-1.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Path to the folder of parquet files.")
    parser.add_argument('--output_folder', type=str, default=".",
                        help="The path to save the preprocessed data to parquet files. ")
    parser.add_argument('--frequency_limit', type=int, default=15,
                        help="frequency below frequency_limit will be "
                             "omitted from the encoding.")
    parser.add_argument('--cross_sizes', type=str,
                        help='bucket sizes for cross columns', default="10000, 10000")

    args = parser.parse_args()
    start, end = args.days.split('-')
    args.day_range = list(range(int(start), int(end) + 1))
    args.days = len(args.day_range)

    args.cross_sizes = [int(x) for x in args.cross_sizes.split(',')]

    return args


def preprocess_and_save(data_tbl, models, save_path):
    columns = dict([("_c{}".format(i), "c{}".format(i)) for i in range(40)])
    data_tbl = data_tbl.rename(columns)

    data_tbl = data_tbl.encode_string(CAT_COLS, models) \
        .fillna(0, INT_COLS + CAT_COLS)
    data_tbl, min_max = data_tbl.min_max_scale(INT_COLS)

    data_tbl = data_tbl.cross_columns(crossed_columns=[CAT_COLS[0:2], CAT_COLS[2:4]],
                                      bucket_sizes=cross_sizes)

    data_tbl = data_tbl.ordinal_shuffle_partition()

    data_tbl.write_parquet(save_path)


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

    time_start = time()
    paths = [os.path.join(args.input_folder, 'day_%d.parquet' % i) for i in args.day_range]
    tbl = FeatureTable.read_parquet(paths)

    # change name for all columns
    columns = dict([("_c{}".format(i), "c{}".format(i)) for i in range(40)])
    tbl = tbl.rename(columns)
    idx_list = tbl.gen_string_idx(CAT_COLS, freq_limit=args.frequency_limit)
    cat_sizes = [idx.size() for idx in idx_list]

    cross_sizes = args.cross_sizes

    # save meta
    if not exists(os.path.join(args.output_folder, "meta")):
        makedirs(os.path.join(args.output_folder, "meta"))
    cate_sizes_text = "\n".join([str(s) for s in cat_sizes])
    write_text(os.path.join(args.output_folder, "meta/categorical_sizes.txt"), cate_sizes_text)

    cross_sizes_text = "\n".join([str(s) for s in cross_sizes])
    write_text(os.path.join(args.output_folder, "meta/cross_sizes.txt"), cross_sizes_text)

    if args.days == 24:  # Full Criteo dataset
        train_data = FeatureTable.read_parquet(paths[:-1])
        preprocess_and_save(train_data, idx_list, os.path.join(args.output_folder, "train_parquet"))

        test_data = FeatureTable.read_parquet(
            os.path.join(args.input_folder, "day_23_test.parquet"))
        preprocess_and_save(test_data, idx_list, os.path.join(args.output_folder, "test_parquet"))
    else:  # Sample data
        data = FeatureTable.read_parquet(paths)
        preprocess_and_save(data, idx_list, os.path.join(args.output_folder, "data_parquet"))

    time_end = time()
    print("Total data loading and preprocessing time: ", time_end - time_start)
    stop_orca_context()
