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
import pickle
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
                        help="The path to the user transaction file.")
    parser.add_argument('--input_meta', type=str, required=True,
                        help="The path to the item metadata file.")
    parser.add_argument('--index_folder', type=str, default="./",
                        help="The folder for user, item and category string indices.")
    parser.add_argument('--output', type=str, default="./",
                        help="The path to save the preprocessed data.")
    parser.add_argument('--num_save_files', type=int, default=None,
                        help="The number of csv files to save for the processed output.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executors,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores,
                               driver_memory=args.driver_memory, conf=conf)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executors, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=conf)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
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

    # def process_single_meta(row):
    #     obj = eval(row)
    #     cat = obj['categories'][0][-1]
    #     return [obj['asin'], cat]
    #
    # item_tbl = FeatureTable.read_text(args.input_meta)\
    #     .apply("value", "value", process_single_meta, dtype="array<string>")\
    #     .apply("value", "item", lambda x: x[0])\
    #     .apply("value", "category", lambda x: x[1])\
    #     .drop("value")

    item_tbl = FeatureTable.read_csv(args.input_meta, delimiter="\t", names=['item', 'category'])

    # Currently long id is not supported for add_negative_samples and add_value_features, cast to int.
    with open(args.index_folder + "vocs/cat_voc.pkl", 'rb') as f:
        categories = FeatureTable(sc.parallelize(list(pickle.load(f).items())).toDF(["category", "id"]))
        categories = categories.cast("id", "int")
        category_index = StringIndex(categories.df, "category")
    with open(args.index_folder + "vocs/mid_voc.pkl", 'rb') as f:
        items = FeatureTable(sc.parallelize(list(pickle.load(f).items())).toDF(["item", "id"]))
        items = items.cast("id", "int")
        item_index = StringIndex(items.df, "item")
    with open(args.index_folder + "vocs/uid_voc.pkl", 'rb') as f:
        users = FeatureTable(sc.parallelize(list(pickle.load(f).items())).toDF(["user", "id"]))
        users = users.cast("id", "int")
        user_index = StringIndex(users.df, "user")
    # user_index = StringIndex.read_parquet(args.index_folder + "user.parquet")
    # item_index = StringIndex.read_parquet(args.index_folder + "item.parquet")
    # category_index = StringIndex.read_parquet(args.index_folder + "category.parquet")
    item_size = item_index.size()

    item_tbl = item_tbl\
        .encode_string(["item", "category"], [item_index, category_index])\
        .fillna(0, ["item", "category"])
    item_tbl.cache()

    full_tbl = transaction_tbl\
        .encode_string(['user', 'item'], [user_index, item_index]) \
        .fillna(0, ["user", "item"])\
        .add_hist_seq(cols=['item'], user_col="user",
                      sort_col='time', min_len=2, max_len=100, num_seqs=1)\
        .add_negative_samples(item_size, item_col='item', neg_num=1)\
        .add_value_features(columns=["item", "item_hist_seq"],
                            dict_tbl=item_tbl, key="item", value="category")

    def list_to_string(items):
        items = [str(item) for item in items]
        return "\x02".join(items)

    # write to csv doesn't support array of int, need to change to string
    full_tbl = full_tbl.apply("item_hist_seq", "item_history", list_to_string)\
        .apply("category_hist_seq", "cat_history", list_to_string) \
        .drop("item_hist_seq", "category_hist_seq")\
        .rename({"label": "pos", "category": "cat"})

    # write out
    meta_dict = item_tbl.to_pandas().to_dict(orient='list')
    pickle.dump(meta_dict,
                open(args.output + "encoded_item_tuple_list.pkl", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
    full_tbl.write_csv(args.output + "processed", delimiter="\t",
                       header=False, num_partitions=args.num_save_files)

    end = time.time()
    print(f"DIEN preprocessing time: {(end - begin):.2f}s")
    full_tbl.show(5)
    print("Total number of processed records: {}".format(full_tbl.size()))
    stop_orca_context()
