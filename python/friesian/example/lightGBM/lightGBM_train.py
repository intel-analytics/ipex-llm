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

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.friesian.feature import FeatureTable
from pyspark.ml.linalg import DenseVector, VectorUDT
import time
import os
import argparse
from bigdl.dllib.utils.log4Error import *
from bigdl.dllib.nnframes.tree_model import LightGBMClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Please use 0.10.0 version for Spark3.2 and 0.9.5-13-d1b51517-SNAPSHOT version for Spark3.1

num_cols = ["enaging_user_follower_count", 'enaging_user_following_count',
            "engaged_with_user_follower_count", "engaged_with_user_following_count",
            "len_hashtags", "len_domains", "len_links"]
cat_cols = ["engaged_with_user_is_verified", "enaging_user_is_verified", "present_media",
            "tweet_type", "language", 'present_media_language', "enaging_user_id",
            "engaged_with_user_id", "hashtags", "present_links", "present_domains"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LightGBM Training')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=4,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=4,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--model_dir', default='./lightgbm', type=str,
                        help='nativeModel directory name (default: nativeModel)')
    parser.add_argument('--data_dir', type=str, help='data directory')

    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores,
                               memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executors,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executors, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)

    save_path = args.model_dir + "lightgbm"

    features = num_cols + [col + "_te_label" for col in cat_cols]
    begin = time.time()
    train_tbl = FeatureTable.read_parquet(args.data_dir + "/train_parquet") \
        .drop("tweet_timestamp", "enaging_user_account_creation", "reply_timestamp", "text_tokens",
              "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")
    test_tbl = FeatureTable.read_parquet(args.data_dir + "/test_parquet") \
        .drop("tweet_timestamp", "enaging_user_account_creation", "reply_timestamp", "text_tokens",
              "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")

    train_tbl.cache()
    test_tbl.cache()
    full = train_tbl.concat(test_tbl)
    full, target_codes = full.target_encode(cat_cols=cat_cols, target_cols=["label"])
    print(full.size())
    for code in target_codes:
        code.cache()

    train = train_tbl \
        .encode_target(target_cols="label", targets=target_codes) \
        .merge_cols(features, "features") \
        .select(["label", "features"]) \
        .apply("features", "features", lambda x: DenseVector(x), VectorUDT()) \
        .repartition(args.num_executor * args.executor_cores)

    train.show(5, False)

    test = test_tbl \
        .encode_target(target_cols="label", targets=target_codes) \
        .merge_cols(features, "features") \
        .select(["label", "features"]) \
        .apply("features", "features", lambda x: DenseVector(x), VectorUDT()) \
        .repartition(args.num_executor * args.executor_cores)

    test.show(5, False)
    train = train.cache()
    test = test.cache()
    print("training size:", train.size())
    print("test size:", test.size())
    #
    train_tbl.uncache()
    test_tbl.uncache()
    preprocess = time.time()
    print("feature preprocessing time: %.2f" % (preprocess - begin))

    params = {"boosting_type": "gbdt", "num_leaves": 70, "learning_rate": 0.3,
              "min_data_in_leaf": 20, "objective": "binary",
              'num_iterations': 1000,
              'max_depth': 14,
              'lambda_l1': 0.01,
              'lambda_l2': 0.01,
              'bagging_freq': 5,
              'max_bin': 255,
              'early_stopping_round': 20
              }

    params = {"objective": "binary", 'num_iterations': 100}
    for learning_rate in [0.1, 0.2]:
        for max_depth in [7, 14]:
            for num_iterations in [100, 200, 400, 800, 10000]:
                params.update({"learning_rate": learning_rate, "max_depth": max_depth,
                               "num_iterations": num_iterations})

                estimator = LightGBMClassifier(params)
                model = estimator.fit(train.df)
                predictions = model.transform(test.df)
                predictions.cache()
                predictions.show(5, False)
                evaluator = BinaryClassificationEvaluator(labelCol="label",
                                                          rawPredictionCol="rawPrediction")
                auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

                evaluator2 = MulticlassClassificationEvaluator(labelCol="label",
                                                               predictionCol="prediction")
                acc = evaluator2.evaluate(predictions, {evaluator2.metricName: "accuracy"})
                print(params)
                print("AUC: %.2f" % (auc * 100.0))
                print("Accuracy: %.2f" % (acc * 100.0))

                predictions.unpersist(blocking=True)

    model.saveModel(args.model_dir)
    end = time.time()
    print("training time: %.2f" % (end - preprocess))
    print(end - begin)
    stop_orca_context()
