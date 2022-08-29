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

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Please use 0.10.0 version for Spark3.2 and 0.9.5-13-d1b51517-SNAPSHOT version for Spark3.1
spark_conf = {"spark.app.name": "recsys-lightGBM",
              "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
              "spark.driver.memoryOverhead": "8G",
              "spark.jars.packages": "com.microsoft.azure:synapseml_2.12:0.9.5-13-d1b51517-SNAPSHOT",
              "spark.jars.repositories": "https://mmlspark.azureedge.net/maven"}

data_dir = "/Users/guoqiong/intelWork/data/tweet/recsys2021_jennie"
model_dir = "./lightgbm"
num_cols = ["enaging_user_follower_count", 'enaging_user_following_count',
            "engaged_with_user_follower_count", "engaged_with_user_following_count",
            "len_hashtags", "len_domains", "len_links"]
cat_cols = ["engaged_with_user_is_verified", "enaging_user_is_verified", "present_media",
            "tweet_type", "language", 'present_media_language', "hashtags", "present_links",
            "present_domains"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCF Training')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn, standalone or spark-submit.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                        help='The number of cores to use on each executor.')
    parser.add_argument('--executor_memory', type=str, default="4g",
                        help='The amount of memory to allocate on each executor.')
    parser.add_argument('--num_executors', type=int, default=2,
                        help='The number of executors to use in the cluster.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The number of cores to use for the driver.')
    parser.add_argument('--driver_memory', type=str, default="4g",
                        help='The amount of memory to allocate for the driver.')
    parser.add_argument('--backend', type=str, default="ray",
                        help='The backend of TF2 Estimator, either ray or spark.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='The learning rate to train the model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The number of epochs to train the model.')
    parser.add_argument('--batch_size', default=8000, type=int,
                        help='The batch size to train the model.')
    parser.add_argument('--model_dir', default='./', type=str,
                        help='The directory to save the trained model.')
    parser.add_argument('--data_dir', type=str, default="./movielens",
                        help='The directory for the movielens data.')
    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores,
                               memory=args.executor_memory, conf=spark_conf)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executors,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executors, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)

    save_path = args.model_dir + "lightgbm"

    features = num_cols + [col + "_te_label" for col in cat_cols]
    begin = time.time()
    train_tbl = FeatureTable.read_parquet(data_dir + "/train_parquet") \
        .drop("tweet_timestamp", "enaging_user_account_creation", "reply_timestamp", "text_tokens",
              "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")
    test_tbl = FeatureTable.read_parquet(data_dir + "/test_parquet") \
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
        .repartition(20)

    train.show(5, False)

    test = test_tbl \
        .encode_target(target_cols="label", targets=target_codes) \
        .merge_cols(features, "features") \
        .select(["label", "features"]) \
        .apply("features", "features", lambda x: DenseVector(x), VectorUDT()) \
        .repartition(20)

    test.show(5, False)
    train = train.cache()
    test = test.cache()
    print("training size:", train.size())
    print("test size:", test.size())

    train_tbl.uncache()
    test_tbl.uncache()

    from synapse.ml.lightgbm import LightGBMClassifier
    model = LightGBMClassifier(objective="binary", featuresCol="features", labelCol="label")

    model = model.fit(train.df)
    predictions = model.transform(test.df)
    predictions.show(50)
    print(predictions.count())

    evaluator = BinaryClassificationEvaluator(labelCol="label",
                                              rawPredictionCol="rawPrediction")
    auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})

    evaluator2 = MulticlassClassificationEvaluator(labelCol="label",
                                                   predictionCol="prediction")
    acc = evaluator2.evaluate(predictions, {evaluator2.metricName: "accuracy"})

    print("AUC: %.2f" % (auc * 100.0))
    print("Accuracy: %.2f" % (acc * 100.0))

    stop_orca_context()