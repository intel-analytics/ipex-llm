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

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable
from pyspark.ml.linalg import DenseVector, VectorUDT
from sklearn.metrics import accuracy_score
from bigdl.dllib.nnframes.nn_classifier import *
import argparse
import time

spark_conf = {"spark.network.timeout": "10000000",
              "spark.sql.broadcastTimeout": "7200",
              "spark.sql.shuffle.partitions": "2000",
              "spark.locality.wait": "0s",
              "spark.sql.hive.filesourcePartitionFileCacheSize": "4096000000",
              "spark.sql.crossJoin.enabled": "true",
              "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
              "spark.kryo.unsafe": "true",
              "spark.kryoserializer.buffer.max": "1024m",
              "spark.task.cpus": "4",
              "spark.executor.heartbeatInterval": "200s",
              "spark.driver.maxResultSize": "40G",
              "spark.eventLog.enabled": "true",
              "spark.app.name": "recsys-xgb"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep FM Training')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=4,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--dricver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--model_dir', default='snapshot', type=str,
                        help='snapshot directory name (default: snapshot)')
    parser.add_argument('--data_dir', type=str, help='data directory')

    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=4, conf=spark_conf)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executor, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")

    num_cols = ["enaging_user_follower_count", 'enaging_user_following_count',
                "engaged_with_user_follower_count", "engaged_with_user_following_count",
                "len_hashtags", "len_domains", "len_links"]
    cat_cols = ["engaged_with_user_is_verified", "enaging_user_is_verified", "present_media",
                "tweet_type", "language", 'present_media_language']
    embed_cols = ["enaging_user_id", "engaged_with_user_id", "hashtags", "present_links",
                  "present_domains"]

    features = num_cols + [col + "_te_label" for col in cat_cols] +\
               [col + "_te_label" for col in embed_cols]
    begin = time.time()
    train_tbl = FeatureTable.read_parquet(args.data_dir + "/train_parquet")\
        .drop("tweet_timestamp", "enaging_user_account_creation", "reply_timestamp", "text_tokens",
               "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")
    test_tbl = FeatureTable.read_parquet(args.data_dir + "/test_parquet")\
        .drop("tweet_timestamp", "enaging_user_account_creation", "reply_timestamp", "text_tokens",
               "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")
    train_tbl = train_tbl.cache()
    test_tbl = test_tbl.cache()
    full = train_tbl.concat(test_tbl)

    full, min_max_dict = full.min_max_scale(num_cols)
    index_tbls = full.gen_reindex_mapping(embed_cols)
    full, target_codes = full.target_encode(cat_cols=cat_cols + embed_cols, target_cols=["label"])

    train = train_tbl.transform_min_max_scale(num_cols, min_max_dict)\
        .reindex(embed_cols, index_tbls)\
        .encode_target(target_cols="label", targets=target_codes)\
        .merge_cols(features, "features") \
        .select(["label", "features"])\
        .apply("features", "features", lambda x: DenseVector(x), VectorUDT())
    train.show(5, False)

    test = test_tbl.transform_min_max_scale(num_cols, min_max_dict)\
        .reindex(embed_cols, index_tbls)\
        .encode_target(target_cols="label", targets=target_codes) \
        .merge_cols(features, "features") \
        .select(["label", "features"]) \
        .apply("features", "features", lambda x: DenseVector(x), VectorUDT())

    test.show(5, False)
    print("training size:", train.size())
    train = train.cache()
    print("test size:", test.size())
    test = test.cache()

    train_tbl.uncache()
    test_tbl.uncache()

    preprocess = time.time()
    print("feature preprocessing time: %.2f" % (preprocess - begin))

    params = {"tree_method": 'hist', "eta": 0.1, "gamma": 0.1,
              "min_child_weight": 30, "reg_lambda": 1, "scale_pos_weight": 2,
              "subsample": 1, "objective": "binary:logistic"}

    for eta in [0.05, 0.1, 0.15, 0.2]:
        for max_depth in [4, 6, 8, 10]:
            for num_round in [100, 200, 400]:
                params.update({"eta": eta, "max_depth": max_depth, "num_round": num_round})
                classifier = XGBClassifier(params)
                classifier.setNthread(1)
                xgbmodel = classifier.fit(train.df)
                xgbmodel.setFeaturesCol("features")
                predicts = xgbmodel.transform(test.df)

                predicts.cache()
                gr = [row.label for row in predicts.select("label").collect()]
                predictions = [row.prediction for row in predicts.select("prediction").collect()]
                accuracy = accuracy_score(gr, predictions)
                predicts.unpersist(blocking=True)
                print(params)
                print("Accuracy: %.2f" % (accuracy * 100.0))
    end = time.time()
    print("training time: %.2f" % (end - preprocess))
    print(end - begin)
    stop_orca_context()
