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

import pickle
import argparse
from pyspark.sql.functions import array
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.file import exists, makedirs
from bigdl.friesian.feature import FeatureTable
from friesian.example.two_tower.model import ColumnInfoTower
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark_hnsw.evaluation import KnnSimilarityEvaluator
from pyspark_hnsw.knn import *
from pyspark_hnsw.linalg import Normalizer
from pyspark.ml import Pipeline
import os

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

def prepare_features(train_tbl, test_tbl, reindex_tbls):

    def add_ratio_features(tbl):
        cal_ratio = (lambda x: x[1] / x[0] if x[0] > 0 else 0.0)
        tbl = tbl.apply(["engaged_with_user_follower_count", "engaged_with_user_following_count"],
                        "engaged_with_user_follower_following_ratio", cal_ratio, "float")\
            .apply(["enaging_user_follower_count", "enaging_user_following_count"],
                   "enaging_user_follower_following_ratio", cal_ratio, "float")
        return tbl

    def organize_cols(tbl):
        tbl = tbl.select(array("enaging_user_follower_count", "enaging_user_following_count",
                               "enaging_user_follower_following_ratio").alias("user_num"),
                         "len_hashtags", "len_domains", "len_links",
                               "engaged_with_user_follower_count",
                               "engaged_with_user_following_count",
                               "engaged_with_user_follower_following_ratio",
                         *cat_cols, *embed_cols, "tweet_id", "label")
        return tbl

    print("reindexing embedding cols")
    train_tbl = train_tbl.reindex(embed_cols, reindex_tbls)
    test_tbl = test_tbl.reindex(embed_cols, reindex_tbls)
    embed_in_dims = {}
    for i, c, in enumerate(embed_cols):
        embed_in_dims[c] = max(reindex_tbls[i].df.agg({c+"_new": "max"}).collect()[0])

    print("add ratio features")
    train_tbl = add_ratio_features(train_tbl)
    test_tbl = add_ratio_features(test_tbl)

    print("scale numerical features")
    train_tbl, min_max_dic = train_tbl.min_max_scale(num_cols + ratio_cols)
    test_tbl = test_tbl.transform_min_max_scale(num_cols + ratio_cols, min_max_dic)

    stats_dir = os.path.join(args.model_dir, 'stats')
    if not exists(stats_dir):
        makedirs(stats_dir)
    with open(os.path.join(stats_dir, "min_max.pkl"), 'wb') as f:
        pickle.dump(min_max_dic, f)

    user_col_info = ColumnInfoTower(indicator_cols=["enaging_user_is_verified"],
                                    indicator_dims=[2],
                                    embed_cols=["enaging_user_id"],
                                    embed_in_dims=[embed_in_dims["enaging_user_id"]],
                                    embed_out_dims=[16],
                                    numerical_cols=["user_num"],
                                    numerical_dims=[3],
                                    name="user")
    item_col_info = ColumnInfoTower(indicator_cols=["engaged_with_user_is_verified",
                                                    "present_media", "tweet_type", "language"],
                                    indicator_dims=[2, 13, 3, 67],  # max + 1
                                    embed_cols=["engaged_with_user_id", "hashtags",
                                                "present_links", "present_domains"],
                                    embed_in_dims=[embed_in_dims["engaged_with_user_id"],
                                                   embed_in_dims["hashtags"],
                                                   embed_in_dims["present_links"],
                                                   embed_in_dims["present_domains"]],
                                    embed_out_dims=[16, 16, 16, 16],
                                    numerical_cols=["item_num"],
                                    numerical_dims=[6],
                                    name="item")

    print("organize columns and specify user_col_info and item_col_info")
    train_tbl = organize_cols(train_tbl)
    test_tbl = organize_cols(test_tbl)

    return train_tbl, test_tbl, user_col_info, item_col_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two Tower Training/Inference')
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
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1, type=int, help='train epoch')
    parser.add_argument('--batch_size', default=8000, type=int, help='batch size')
    parser.add_argument('--model_dir', default='snapshot', type=str,
                        help='snapshot directory name (default: snapshot)')
    parser.add_argument('--data_dir', type=str, help='data directory')
    parser.add_argument('--frequency_limit', type=int, default=25, help='frequency limit')

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
        raise ValueError(
            "cluster_mode should be one of 'local', 'yarn', 'standalone' and 'spark-submit'"
            ", but got " + args.cluster_mode)

    num_cols = ["enaging_user_follower_count", 'enaging_user_following_count',
                "engaged_with_user_follower_count", "engaged_with_user_following_count",
                "len_hashtags", "len_domains", "len_links"]
    cat_cols = ["engaged_with_user_is_verified", "enaging_user_is_verified",
                "present_media", "tweet_type", "language"]
    ratio_cols = ["engaged_with_user_follower_following_ratio",
                  "enaging_user_follower_following_ratio"]
    embed_cols = ["enaging_user_id", "engaged_with_user_id", "hashtags", "present_links",
                  "present_domains"]
    useful_cols = num_cols + cat_cols + embed_cols
    train_tbl = FeatureTable.read_parquet(args.data_dir + "/train_parquet")
    train_tbl.show(10)
    test_tbl = FeatureTable.read_parquet(args.data_dir + "/test_parquet")
    full_tbl = train_tbl.concat(test_tbl, "outer")
    reindex_tbls = full_tbl.gen_reindex_mapping(embed_cols, freq_limit=args.frequency_limit)
    item_cat_cols =["engaged_with_user_is_verified", "present_media", "tweet_type", "language",
                    "engaged_with_user_id", "hashtags", "present_links", "present_domains"]
    full, target_codes = full_tbl.target_encode(cat_cols=item_cat_cols, target_cols=["label"])

    train_tbl, test_tbl, user_info, item_info = prepare_features(train_tbl, test_tbl, reindex_tbls)
    train_tbl.show(10)
    item_num_cols = ["len_hashtags", "len_domains", "len_links",
                               "engaged_with_user_follower_count",
                               "engaged_with_user_following_count",
                               "engaged_with_user_follower_following_ratio"]
    item_features = item_num_cols + item_cat_cols
    item_tbl = train_tbl.select(["tweet_id", "label"] + item_features)

    item_tbl.show(10)
    print(item_tbl.size())
    item_tbl = item_tbl.distinct()
    print(item_tbl.size())
    print(item_tbl.select("tweet_id").distinct().size())
    item_cat_cols = [c + "_te_label" for c in item_cat_cols]
    item_tbl = item_tbl.encode_target(target_cols="label", targets=target_codes)\
        .merge_cols(item_num_cols + item_cat_cols, "features") \
        .select("tweet_id", "features")\
        .apply("features", "features", lambda x: DenseVector(x), VectorUDT())

    normalizer = Normalizer(inputCol='features', outputCol='normalized_features')
    hnsw = HnswSimilarity(identifierCol='tweet_id', queryIdentifierCol='tweet_id',
                          featuresCol='features', distanceFunction='inner-product', m=48,
                          ef=5, k=10,
                          efConstruction=200, numPartitions=11, excludeSelf=True,
                          similarityThreshold=0.1, predictionCol='approximate')
    brute_force = BruteForceSimilarity(identifierCol='tweet_id', queryIdentifierCol='tweet_id',
                                       featuresCol='normalized_features',
                                       distanceFunction='inner-product',
                                       k=10, numPartitions=11, excludeSelf=True,
                                       similarityThreshold=0.1, predictionCol='exact')

    pipeline = Pipeline(stages=[normalizer, hnsw, brute_force])

    model = pipeline.fit(item_tbl.df)
    query_items = item_tbl
    output = model.transform(query_items.df)

    output.show()
    print(output.count())

    stop_orca_context()
