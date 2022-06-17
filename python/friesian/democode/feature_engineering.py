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

import pandas as pd
import argparse
from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.friesian.feature import FeatureTable
from bigdl.dllib.utils.log4Error import *
from bigdl.dllib.feature.dataset import movielens
from pyspark.sql.functions import col as pyspark_col

spark_conf = {"spark.network.timeout": "10000000",
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
              "spark.eventLog.enabled": "true",
              "spark.app.name": "recsys-2tower",
              "spark.executor.memoryOverhead": "120g"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature engineering')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn, standalone or spark-submit.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=8,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--data_dir', type=str, default="./movielens", help='data directory')

    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", init_ray_on_spark=True)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executor,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf,
                               init_ray_on_spark=True)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executor, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf, extra_python_lib="model.py",
                               object_store_memory="80g",
                               init_ray_on_spark=True)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)
    # download data
    # _ = movielens.get_id_ratings(args.data_dir)
    ratings = pd.read_csv(args.data_dir + "/ml-1m/ratings.dat", delimiter="::",
                          names=["user", "item", "rate", "time"])
    ratings_tbl = FeatureTable.from_pandas(ratings)\
        .cast(["user", "item", "rate"], "int").cast("time", "long")
    print("count: ", ratings_tbl.size())

    user_df = pd.read_csv(args.data_dir + "/ml-1m/users.dat", delimiter="::",
                          names=["user", "gender", "age", "occupation", "zipcode"])
    user_tbl = FeatureTable.from_pandas(user_df)\
        .cast(["user", "age", "occupation", "zipcode"], "int").fillna(0, 'zipcode')

    item_df = pd.read_csv(args.data_dir + "/ml-1m/movies.dat", encoding="ISO-8859-1",
                          delimiter="::", names=["item", "title", "genres"])
    item_tbl = FeatureTable.from_pandas(item_df).cast("item", "int")

    user_stats = ratings_tbl.group_by("user", agg={"item":"count", "rate":"mean"})\
                            .rename({"count(item)": "user_visits", "avg(rate)": "user_mean_rate"})
    user_stats, user_min_max = user_stats.min_max_scale(["user_visits", "user_mean_rate"])

    item_stats = ratings_tbl.group_by("item", agg={"user":"count", "rate":"mean"})\
                            .rename({"count(user)": "item_visits", "avg(rate)": "item_mean_rate"})
    item_stats, user_min_max = item_stats.min_max_scale(["item_visits", "item_mean_rate"])

    item_size = item_stats.select("item").distinct().size()
    ratings_tbl = ratings_tbl.add_negative_samples(item_size=item_size, item_col="item", label_col="label", neg_num=1)

    user_tbl, inx_list = user_tbl.category_encode(["gender", "age", "zipcode", "occupation"])
    item_tbl, item_list = item_tbl.category_encode(["genres"])

    user_tbl = user_tbl.cross_columns([["gender", "age"], ["age", "zipcode"]], [50, 200])
    user_tbl = user_tbl.join(user_stats, on="user")
    item_tbl = item_tbl.join(item_stats, on="item")
    full = ratings_tbl.join(user_tbl, on="user")\
        .join(item_tbl, on="item")

    full.df.printSchema()
    wide_cols = ["gender", "age", "occupation", "zipcode", "genres"]
    wide_cross_cols = ["gender_age", "age_zipcode"]
    indicator_cols = wide_cols + wide_cross_cols
    embed_cols = ["user", "item"]
    num_cols = ["user_visits", "user_mean_rate", "item_visits", "item_mean_rate"]
    cat_cols = wide_cols + wide_cross_cols + embed_cols
    all = cat_cols + num_cols
    print("count: ", full.size())
    for c in all:
        print(c, full.df.filter(pyspark_col(c).isNull()).count())


    user_tbl.write_parquet(args.data_dir + "/processed/user_features/")
    item_tbl.write_parquet(args.data_dir + "/processed/item_features/")
    full.write_parquet(args.data_dir + "/processed/full/")
    stop_orca_context()

