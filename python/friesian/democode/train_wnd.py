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
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.data.file import exists, makedirs
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.learn.tf2.estimator import Estimator
from friesian.example.wnd.train.wnd_train_recsys import ColumnFeatureInfo, model_creator
from bigdl.dllib.utils.log4Error import *
import time
import math
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
    parser.add_argument('--model_dir', default='model_wnd', type=str,
                        help='snapshot directory name (default: model_wnd)')
    parser.add_argument('--data_dir', type=str, default="./movielens/processed", help='data directory')

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
                               conf=spark_conf,
                               object_store_memory="8g",
                               init_ray_on_spark=True)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)

    wide_cols = ["gender", "age", "occupation", "zipcode", "genres"]
    wide_cross_cols = ["gender_age", "age_zipcode"]
    indicator_cols = wide_cols + wide_cross_cols
    embed_cols = ["user", "item"]
    num_cols = ["user_visits", "user_mean_rate", "item_visits", "item_mean_rate"]
    cat_cols = wide_cols + wide_cross_cols + embed_cols


    full = FeatureTable.read_parquet(args.data_dir + "/full")\
        .select("label", *cat_cols, *num_cols)
    stats = full.get_stats(cat_cols, "max")
    for key in stats.keys():
        stats[key] += 1
    train_tbl, test_tbl = full.random_split([0.8, 0.2], seed=1)

    wide_dims = [stats[key] for key in wide_cols]
    wide_cross_dims = [stats[key] for key in wide_cross_cols]
    embed_dims = [stats[key] for key in embed_cols]

    column_info = ColumnFeatureInfo(wide_base_cols=wide_cols,
                                    wide_base_dims=wide_dims,
                                    wide_cross_cols=wide_cross_cols,
                                    wide_cross_dims=wide_cross_dims,
                                    indicator_cols=indicator_cols,
                                    indicator_dims=wide_dims + wide_cross_dims,
                                    embed_cols=embed_cols,
                                    embed_in_dims=embed_dims,
                                    embed_out_dims=[8] * len(embed_dims),
                                    continuous_cols=num_cols,
                                    label="label")

    config = {
        "lr": args.lr,
        "hidden_units": [40, 20],
        "column_info": column_info,
        "inter_op_parallelism": 4,
        "intra_op_parallelism": args.executor_cores
    }

    est = Estimator.from_keras(
        model_creator=model_creator,
        verbose=True,
        config=config,
        backend="tf2")

    train_tbl.df.printSchema()
    train_tbl.show(20, False)

    if not exists(args.model_dir):
        makedirs(args.model_dir)

    callbacks = []
    # early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    callbacks.append(EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=5))
    train_count = train_tbl.size()
    print("train size: ", train_count)
    steps = math.ceil(train_count / args.batch_size)
    test_count = test_tbl.size()
    print("test size: ", test_count)
    val_steps = math.ceil(test_count / args.batch_size)

    start = time.time()
    est.fit(data=train_tbl.df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            steps_per_epoch=steps,
            validation_data=test_tbl.df,
            validation_steps=val_steps,
            feature_cols=column_info.feature_cols,
            label_cols=column_info.label_cols)
    end = time.time()
    print("Training time is: ", end - start)
    est.save(os.path.join(args.model_dir, "model-%d.ckpt" % args.epochs))
    model = est.get_model()
    model.save_weights(os.path.join(args.model_dir, "model.h5"))

    stop_orca_context()
