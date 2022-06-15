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
from bigdl.orca.learn.tf2.estimator import Estimator
from friesian.example.two_tower.model import *
from bigdl.dllib.utils.log4Error import *


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


def train(config, train_tbl, test_tbl, epochs=1, batch_size=128, model_dir='.'):
    two_tower = TwoTowerModel(config["user_col_info"], config["item_col_info"])

    def model_creator(config):
        model = two_tower.build_model()
        print(model.summary())
        optimizer = tf.keras.optimizers.Adam(config["lr"])
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy', 'Recall', 'AUC'])
        return model

    estimator = Estimator.from_keras(model_creator=model_creator,
                                     verbose=False,
                                     config=config)

    callbacks = []
    from tensorflow.keras.callbacks import EarlyStopping
    callbacks.append(EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=5))

    train_count, test_count = train_tbl.size(), test_tbl.size()
    train_df, test_df = train_tbl.df, test_tbl.df
    steps_per_epoch = math.ceil(train_count / batch_size)
    val_steps = math.ceil(test_count / batch_size)
    feature_cols = config["user_col_info"].get_name_list() + config["item_col_info"].get_name_list()
    print("Total number of train records: {}".format(train_count))
    print("Total number of val records: {}".format(test_count))

    estimator.fit(train_df, epochs=epochs, batch_size=batch_size,
                  feature_cols=feature_cols,
                  label_cols=['label'],
                  callbacks=callbacks,
                  validation_data=test_df,
                  steps_per_epoch=steps_per_epoch,
                  validation_steps=val_steps)

    model = estimator.get_model()
    user_model = get_1tower_model(model, two_tower.user_col_info)
    item_model = get_1tower_model(model, two_tower.item_col_info)
    tf.saved_model.save(model, os.path.join(model_dir, "twotower-model"))
    tf.saved_model.save(user_model, os.path.join(model_dir, "user-model"))
    tf.saved_model.save(item_model, os.path.join(model_dir, "item-model"))
    estimator.save(os.path.join(model_dir, "twotower_model.ckpt"))
    print("saved models")
    return estimator

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
                               conf=spark_conf, extra_python_lib="../example/two_tower/model.py",
                               object_store_memory="8g",
                               init_ray_on_spark=True)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)

    full = FeatureTable.read_parquet(args.data_dir + "/full")
    stats = full.get_stats(["user", "item", "zipcode", "gender", "age", "occupation", "zipcode", "genres"], "max")
    for key in stats.keys():
        stats[key] += 1
    print(stats)
    for items in stats:
        print(items)
    train_tbl, test_tbl = full.random_split([0.8, 0.2], seed=1)
    user_info = ColumnInfoTower(indicator_cols=["gender", "age", "occupation"],
                                indicator_dims=[stats["gender"], stats["age"], stats["occupation"]],
                                embed_cols=["user", "zipcode"],
                                embed_in_dims=[stats["user"], stats["zipcode"]],
                                embed_out_dims=[16, 16],
                                numerical_cols=["user_visits", "user_mean_rate"],
                                numerical_dims=[1, 1],
                                name="user")
    item_info = ColumnInfoTower(indicator_cols=["genres"],
                                indicator_dims=[stats["genres"]],
                                embed_cols=["item"],
                                embed_in_dims=[stats["item"]],
                                embed_out_dims=[16],
                                numerical_cols=["item_visits", "item_mean_rate"],
                                numerical_dims=[1, 1],
                                name="item")
    train_config = {"lr": 1e-3,
                    "user_col_info": user_info,
                    "item_col_info": item_info,
                    "inter_op_parallelism": 4,
                    "intra_op_parallelism": args.executor_cores}

    train_tbl.df.printSchema()
    train_tbl.show(20, False)

    train(train_config, train_tbl, test_tbl, epochs=args.epochs, batch_size=args.batch_size,
          model_dir=args.model_dir)

    stop_orca_context()
