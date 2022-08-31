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

from dienl import SparseFeat, VarLenSparseFeat, DIEN
import torch.nn as nn
import torch.optim as optim

from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.metrics import Accuracy, BinaryCrossEntropy
from bigdl.friesian.feature import FeatureTable
from pyspark.sql.window import Window
from pyspark.sql.functions import desc, rank, col

import argparse
from bigdl.dllib.utils.log4Error import *


EMBEDDING_DIM = 18
MAX_HIST_LEN = 100
hash_flag = False

feature_columns = []
behavior_feature_list = []


def model_creator(config):
    model = DIEN(feature_columns, behavior_feature_list, dnn_hidden_units=[4, 4, 4],
                 dnn_dropout=0.4, gru_type="AUGRU", use_negsampling=False,
                 device='cpu', seed=2, att_hidden_units=(64, 16),
                 att_activation="relu", init_std=0.1)
    return model


def optim_creator(model, config):
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow DIEN Training/Inference')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn, standalone or spark-submit.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=2,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executors', type=int, default=8,
                        help='The number of executors.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='train epoch')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--data_dir', type=str, default="./preprocessed", help='data directory')
    args = parser.parse_args()

    conf = {"spark.executor.memoryOverhead": "130g",
            "spark.network.timeout": "10000000",
            "spark.sql.broadcastTimeout": "7200",
            "spark.sql.shuffle.partitions": "2000",
            "spark.locality.wait": "0s",
            "spark.sql.crossJoin.enabled": "true",
            "spark.task.cpus": "1",
            "spark.executor.heartbeatInterval": "200s",
            "spark.driver.maxResultSize": "40G",
            "spark.eventLog.enabled": "true",
            "spark.app.name": "recsys-demo-train",
            "spark.rpc.message.maxSize": "256"}

    if args.cluster_mode == "local":
        init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        init_orca_context("standalone", master=args.master,
                          cores=args.executor_cores, num_nodes=args.num_executors,
                          memory=args.executor_memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          conf=conf, extra_python_lib="./dienl.py")
    elif args.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=args.executor_cores,
                          num_nodes=args.num_executors, memory=args.executor_memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          conf=conf, object_store_memory="80g", extra_python_lib="./dienl.py")
    elif args.cluster_mode == "spark-submit":
        init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                          " 'spark-submit', but got " + args.cluster_mode)

    # Read Data
    tbl = FeatureTable.read_parquet(args.data_dir + "data") \
                      .rename({'item_hist_seq': 'hist_item_id', 'item': 'item_id',
                               'category': 'cate_id', 'category_hist_seq': 'hist_cate_id',
                               'item_hist_seq_len': 'seq_length'}) \
                      .apply('label', 'label',
                             lambda x: 0.0 if float(x[0]) == 1.0 else 1.0, "float")
    # windowSpec1 = Window.partitionBy("user").orderBy(desc("time"))
    # tbl = tbl.append_column("rank1", rank().over(windowSpec1))
    # tbl = tbl.filter(col('rank1') == 1)
    train_data, test_data = tbl.split([0.8, 0.2], seed=1)

    usertbl = FeatureTable.read_parquet(args.data_dir + "/user.parquet/*")
    itemtbl = FeatureTable.read_parquet(args.data_dir + "/item.parquet/*")
    cattbl = FeatureTable.read_parquet(args.data_dir + "/category.parquet/*")

    n_uid = usertbl.get_stats("id", "max")["id"] + 1
    n_mid = itemtbl.get_stats("id", "max")["id"] + 1
    n_cat = cattbl.get_stats("id", "max")["id"] + 1
    train_data.show(5)

    print("train size: ", train_data.size())
    print("test size: ", test_data.size())
    print("user size: ", n_uid)
    print("item size: ", n_mid)
    print("category size: ", n_cat)

    # build input feature table
    feature_columns = [
        SparseFeat('user', n_uid, embedding_dim=EMBEDDING_DIM, use_hash=hash_flag),
        SparseFeat('item_id', n_mid, embedding_dim=EMBEDDING_DIM, use_hash=hash_flag),
        SparseFeat('cate_id', n_cat, embedding_dim=EMBEDDING_DIM, use_hash=hash_flag)]
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item_id',
                                                    vocabulary_size=n_mid,
                                                    embedding_dim=EMBEDDING_DIM,
                                                    embedding_name='item_id'),
                                         maxlen=MAX_HIST_LEN,
                                         length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_cate_id',
                                                    vocabulary_size=n_cat,
                                                    embedding_dim=EMBEDDING_DIM,
                                                    embedding_name='cate_id'),
                                         maxlen=MAX_HIST_LEN,
                                         length_name="seq_length")]

    behavior_feature_list = ["item_id", "cate_id"]

    # Create model
    device = 'cpu'
    criterion = nn.BCELoss()
    orca_estimator = Estimator.from_torch(model=model_creator,
                                          optimizer=optim_creator,
                                          metrics=[Accuracy(), BinaryCrossEntropy()],
                                          loss=criterion,
                                          backend="ray",
                                          use_tqdm=True)

    # Train model
    orca_estimator.fit(data=train_data.df,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       feature_cols=["user", "item_id", "cate_id",
                                     "hist_item_id", "seq_length", "hist_cate_id"],
                       label_cols=["label"])
    res = orca_estimator.evaluate(
        data=test_data.df,
        feature_cols=["user", "item_id", "cate_id",
                      "hist_item_id", "seq_length", "hist_cate_id"],
        label_cols=["label"])
    for r in res:
        print(r, ":", res[r])
    stop_orca_context()
