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

from zoo.orca import init_orca_context, stop_orca_context

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
    parser.add_argument('--output_folder')
    parser.add_argument(
        '--write_mode',
        choices=['overwrite', 'errorifexists'],
        default='errorifexists')

    parser.add_argument('--frequency_limit', type=str, default="15",
                        help="Categories with a count/frequency below frequency_limit will be "
                             "omitted from the encoding. For instance, '15', '_c14:15,_c15:16', "
                             "etc")

    args = parser.parse_args()
    start, end = args.days.split('-')
    args.day_range = list(range(int(start), int(end) + 1))
    args.days = len(args.day_range)

    frequency_limit_dict = {}
    default_limit = None
    if args.frequency_limit:
        frequency_limit = args.frequency_limit.split(",")
        for fl in frequency_limit:
            frequency_pair = fl.split(":")
            if len(frequency_pair) == 1:
                default_limit = int(frequency_pair[0])
            elif len(frequency_pair) == 2:
                frequency_limit_dict[frequency_pair[0]] = frequency_pair[1]
    if len(frequency_limit_dict) > 0:
        args.frequency_limit = frequency_limit_dict
    else:
        args.frequency_limit = default_limit

    return args


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
    idx_list = tbl.gen_string_idx(CAT_COLS, freq_limit=args.frequency_limit)
    tbl_all_data = FeatureTable.read_parquet(paths[:-1])
    tbl_all_data = tbl_all_data.encode_string(CAT_COLS, idx_list)\
        .fillna(0, INT_COLS + CAT_COLS).log(INT_COLS)
    tbl_all_data = tbl_all_data.merge_cols(INT_COLS, "X_int").merge_cols(CAT_COLS, "X_cat")
    tbl_all_data.compute()
    time_end = time()
    print("Train data loading and preprocessing time: ", time_end - time_start)
    tbl_all_data.show(5)
    print("Finished")
    stop_orca_context()
