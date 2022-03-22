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

import argparse
from bigdl.friesian.feature import FeatureTable
from bigdl.orca import init_orca_context, stop_orca_context
from pyspark.ml import Pipeline
from pyspark_hnsw.evaluation import KnnSimilarityEvaluator
from pyspark_hnsw.knn import *
from pyspark_hnsw.linalg import Normalizer
from pyspark.ml.linalg import DenseVector, VectorUDT

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
              "spark.app.name": "recsys-hnsw",
              "spark.executor.memoryOverhead": "120g"}

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
    parser.add_argument('--driver_cores', type=int, default=8,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--data_dir', type=str, help='data directory')

    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local")
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executor,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executor, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        raise ValueError(
            "cluster_mode should be one of 'local', 'yarn', 'standalone' and 'spark-submit'"
            ", but got " + args.cluster_mode)

    item_data = FeatureTable.read_parquet(args.data_dir + "/item_ebd.parquet")\
        .rename({"prediction": "features"})\
        .apply("tweet_id", "partition", lambda x: x/100)\
        .cast("partition", "int") \
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

    model = pipeline.fit(item_data.df)

    query_items = item_data.sample(0.10)
    output = model.transform(query_items.df)
    evaluator = KnnSimilarityEvaluator(approximateNeighborsCol='approximate', exactNeighborsCol='exact')
    accuracy = evaluator.evaluate(output)
    outtbl = FeatureTable(output.select("tweet_id", "exact.neighbor"))\
        .apply("neighbor", "neighbor", lambda s: ' '.join([str(elem) for elem in s]))
    outtbl.show(10, False)
    outtbl.df. printSchema()
    print(accuracy)
    outtbl.write_parquet(args.data_dir + "/item_neighbors.parquet")
