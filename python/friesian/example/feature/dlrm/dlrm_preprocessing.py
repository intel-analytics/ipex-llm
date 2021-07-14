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

import os
from time import time
from argparse import ArgumentParser

from zoo.orca import init_orca_context, stop_orca_context
from zoo.friesian.feature import FeatureTable


LABEL_COL = 0
INT_COLS = ["_c{}".format(i) for i in list(range(1, 14))]
CAT_COLS = ["_c{}".format(i) for i in list(range(14, 40))]

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
        # Default replication is 3. You may adjust according your cluster settings.
        "spark.hadoop.dfs.replication": "1"}


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cluster_mode", type=str, default="local",
                        help="The cluster mode, local, yarn or standalone.")
    parser.add_argument("--master", type=str, default=None,
                        help="The master url, only used when the cluster_mode is standalone.")
    parser.add_argument("--cores", type=int, default=48,
                        help="The number of cores to use on each node.")
    parser.add_argument("--memory", type=str, default="240g",
                        help="The amount of memory to allocate on each node.")
    parser.add_argument("--num_nodes", type=int, default=8,
                        help="The number of nodes to use in the cluster.")
    parser.add_argument("--driver_cores", type=int, default=4,
                        help="The number of cores to use for the driver.")
    parser.add_argument("--driver_memory", type=str, default="36g",
                        help="The amount of memory to allocate for the driver.")
    parser.add_argument("--days", type=str, default="0-23",
                        help="The day range for data preprocessing, such as 0-23, 0-1.")
    parser.add_argument("--frequency_limit", type=int, default=15,
                        help="Categories with frequency below this value will be "
                             "omitted from encoding.")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="The path to the folder of parquet files, "
                             "either a local path or an HDFS path.")
    parser.add_argument("--output_folder", type=str,
                        help="The path to save the preprocessed data and "
                             "the generated string indices to parquet files. "
                             "HDFS path is recommended.")

    args = parser.parse_args()
    start, end = args.days.split("-")
    args.day_range = list(range(int(start), int(end) + 1))
    args.days = len(args.day_range)
    return args


def preprocess_and_save(data_tbl, models, mode, save_path):
    data_tbl = data_tbl.encode_string(CAT_COLS, models) \
        .fillna(0, INT_COLS + CAT_COLS).log(INT_COLS)
    data_tbl = data_tbl.ordinal_shuffle_partition()
    if save_path:
        if mode == "train":
            save_path = os.path.join(save_path, "saved_data")
        elif mode == "test":
            save_path = os.path.join(save_path, "saved_data_test")
        else:
            raise ValueError("mode should be either train or test")
        print("Saving {} data files to {}".format(mode, save_path))
        data_tbl.write_parquet(save_path)
    else:
        data_tbl.compute()
    return data_tbl


if __name__ == "__main__":
    args = _parse_args()
    if args.cluster_mode == "local":
        init_orca_context("local", cores=args.cores, memory=args.memory)
    elif args.cluster_mode == "standalone":
        init_orca_context("standalone", master=args.master, cores=args.cores,
                          num_nodes=args.num_nodes, memory=args.memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          conf=conf)
    elif args.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=args.cores,
                          num_nodes=args.num_nodes, memory=args.memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          conf=conf)
    time_start = time()

    paths = [os.path.join(args.input_folder, "day_%d.parquet" % i) for i in args.day_range]
    tbl = FeatureTable.read_parquet(paths)
    idx_list = tbl.gen_string_idx(CAT_COLS, freq_limit=args.frequency_limit)

    train_data = FeatureTable.read_parquet(paths[:-1])
    train_preprocessed = preprocess_and_save(train_data, idx_list, "train", args.output_folder)

    if args.days == 24:  # Full Criteo dataset
        test_data = FeatureTable.read_parquet(
            os.path.join(args.input_folder, "day_23_test.parquet"))
        test_preprocessed = preprocess_and_save(test_data, idx_list, "test", args.output_folder)

    time_end = time()
    print("Total preprocessing time: ", time_end - time_start)
    train_preprocessed.show(5)

    if args.output_folder:
        for idx in idx_list:
            idx.write_parquet(args.output_folder)

    print("Finished")
    stop_orca_context()
