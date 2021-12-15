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
from pyspark.ml.linalg import DenseVector, SparseVector, Vectors, VectorUDT
from sklearn.metrics import accuracy_score
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql import types as T
from bigdl.dllib.nnframes.nn_classifier import *
import argparse

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
    parser = argparse.ArgumentParser(description='Deep FM Training')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
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
    parser.add_argument('--model_dir', default='snapshot', type=str,
                        help='snapshot directory name (default: snapshot)')
    parser.add_argument('--data_dir', type=str, help='data directory')
    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local")
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executor, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf, object_store_memory="40g", init_ray_on_spark=True)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit", object_store_memory="40g")

    num_cols = ["enaging_user_follower_count", 'enaging_user_following_count',
                "engaged_with_user_follower_count", "engaged_with_user_following_count",
                "len_hashtags", "len_domains", "len_links"]
    # cat_cols = ["engaged_with_user_is_verified", "enaging_user_is_verified", "present_media",
    #             "tweet_type", "language", 'present_media_language']
    # embed_cols = ["enaging_user_id", "engaged_with_user_id", "hashtags", "present_links",
    #               "present_domains"]
    features = num_cols

    train_tbl = FeatureTable.read_parquet(args.data_dir + "/train_parquet")
    test_tbl = FeatureTable.read_parquet(args.data_dir + "/test_parquet")
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    train = assembler.transform(train_tbl.df).select("features", "label").cache()
    test = test_tbl.df

    todense = lambda x: DenseVector(x.toArray())
    todenseUdf = F.udf(todense, VectorUDT())
    train = train.withColumn("features", todenseUdf("features"))
    train.printSchema()
    train.show(5, False)

    # fit model no training data
    classifier = XGBClassifier()
    classifier.setNthread(1)
    classifier.fit(train)
    print(classifier)
    xgbmodel = XGBClassifierModel(classifier.fit(train))
    xgbmodel.setFeaturesCol(features)

    # make predictions for test data
    y_pred = xgbmodel.transform(test)
    y_pred.show()
    y_pred.printSchema()
    gr = [row.label for row in test.select("label").collect()]
    predictions = [row.prediction for row in y_pred.select("prediction").collect()]
    print(predictions)
    # evaluate predictions
    accuracy = accuracy_score(gr, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    stop_orca_context()
