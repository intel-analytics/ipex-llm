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
import numpy as np
from pyparsing import col
import torch
import os

from dienl import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names, DIEN
from pyspark.ml.linalg import DenseVector
import torch.nn as nn
import numpy as np
from pyspark.sql.functions import lit, rand, col
import torch.optim as optim
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, BinaryCrossEntropy
from bigdl.orca import OrcaContext
from torch.nn import functional as F
from bigdl.orca.learn.trigger import EveryEpoch

import argparse
import math
import time

from bigdl.orca import init_orca_context, stop_orca_context, OrcaContext
from bigdl.orca.learn.pytorch import Estimator
from bigdl.friesian.feature import FeatureTable
from pyspark.sql.window import Window
from pyspark.sql.functions import desc, rank, col

input_meta = '/dien/meta_Books.csv'
input_transaction = '/dien/reviews_Books.json'
output = '/dien/data/'

n_uid = 8026325
n_mid = 2370586
n_cat = 2752

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
MAX_HIST_LEN = 100
SEED = 3
best_auc = 0.0
hash_flag = False
use_neg = False

# build input feature table
feature_columns = [SparseFeat('user', n_uid, embedding_dim=EMBEDDING_DIM, use_hash=hash_flag),
                    SparseFeat('item_id', n_mid, embedding_dim=EMBEDDING_DIM, use_hash=hash_flag),
                    SparseFeat('cate_id', n_cat, embedding_dim=EMBEDDING_DIM, use_hash=hash_flag)]
feature_columns += [
    VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=n_mid, embedding_dim=EMBEDDING_DIM, embedding_name='item_id'),
                        maxlen=MAX_HIST_LEN, length_name="seq_length"),
    VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=n_cat, embedding_dim=EMBEDDING_DIM, embedding_name='cate_id'),
                        maxlen=MAX_HIST_LEN,
                        length_name="seq_length")]

behavior_feature_list = ["item_id", "cate_id"]


def model_creator(config):
    model = DIEN(feature_columns, behavior_feature_list, dnn_hidden_units=[4, 4, 4], dnn_dropout=0.4, gru_type="AUGRU",
                 use_negsampling=False, device='cpu', seed=2, att_hidden_units=(64, 16), att_activation="relu", init_std=0.1)#, modelpath=root_path)
    return model

def optim_creator(model, config):
    optimizer = optim.Adam(model.parameters(),
                          lr=config.get("lr", 0.001))
    return optimizer

if __name__ == '__main__':
    OrcaContext.log_output = True  
    executor_cores = 28
    num_executor = 2
    executor_memory = "30g"
    driver_cores = 4
    driver_memory = "36g"
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
            "spark.rpc.message.maxSize":"256"}
    sc = init_orca_context("yarn", cores=executor_cores,
                        num_nodes=num_executor, memory=executor_memory,
                        driver_cores=driver_cores, driver_memory=driver_memory,
                        conf=conf, object_store_memory="80g",
                        env={"KMP_BLOCKTIME": "1",
                                "KMP_AFFINITY": "granularity=fine,compact,1,0",
                                "OMP_NUM_THREADS": "28"},
                        extra_python_lib="./dienl.py")

    # Preprocess
    begin = time.time()
    transaction_tbl = FeatureTable.read_json(input_transaction).select(
        ['reviewerID', 'asin', 'unixReviewTime']) \
        .rename({'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'time'}) \
        .dropna(columns=['user', 'item'])
    transaction_tbl.cache()
    print("Total number of transactions: ", transaction_tbl.size())

    item_tbl = FeatureTable.read_csv(input_meta, delimiter="\t", names=['item', 'category'])\
        .apply("category", "category", lambda x: x.lower() if x is not None else "default")
    item_tbl.cache()
    print("Total number of items: ", item_tbl.size())

    user_index = transaction_tbl.gen_string_idx('user', freq_limit=1)
    item_cat_indices = item_tbl.gen_string_idx(["item", "category"], freq_limit=1)
    item_size = item_cat_indices[0].size()

    item_tbl = item_tbl\
        .encode_string(["item", "category"], item_cat_indices)

    full_tbl = transaction_tbl\
        .encode_string(['user', 'item'], [user_index, item_cat_indices[0]])\
        .add_hist_seq(cols=['item'], user_col="user",
                    sort_col='time', min_len=1, max_len=100, num_seqs=1)\
        .add_neg_hist_seq(item_size, 'item_hist_seq', neg_num=5)\
        .add_negative_samples(item_size, item_col='item', neg_num=1)\
        .add_value_features(columns=["item", "item_hist_seq", "neg_item_hist_seq"],
                            dict_tbl=item_tbl, key="item", value="category") \
        .apply("item_hist_seq", "item_hist_seq_len", len, "int") \
        .pad(cols=['item_hist_seq', 'category_hist_seq',
            'neg_item_hist_seq', 'neg_category_hist_seq'],
            seq_len=100,
            mask_cols=['item_hist_seq']) \
        .apply("label", "label", lambda x: [1 - float(x), float(x)], "array<float>")

    # write out
    user_index.write_parquet(output)
    item_cat_indices[0].write_parquet(output)
    item_cat_indices[1].write_parquet(output)
    item_tbl.write_parquet(output + "item2cat")
    full_tbl.write_parquet(output + "data")

    end = time.time()

    print(f"DIEN preprocessing time: {(end - begin):.2f}s")

    # Read Data
    data_dir = "/dien/data/"
    tbl = FeatureTable.read_parquet(data_dir + "data") \
            .rename({'item_hist_seq': 'hist_item_id', 'item': 'item_id', 'category': 'cate_id','category_hist_seq':'hist_cate_id','item_hist_seq_len':'seq_length'}) \
            .apply('label','label',lambda x: 0.0 if float(x[0]) == 1.0 else 1.0, "float")
    windowSpec1 = Window.partitionBy("user").orderBy(desc("time"))
    tbl = tbl.append_column("rank1", rank().over(windowSpec1))
    tbl = tbl.filter(col('rank1') == 1)
    train_data, test_data = tbl.split([0.8, 0.2], seed=1)

    usertbl = FeatureTable.read_parquet(data_dir + "/user.parquet/*")
    itemtbl = FeatureTable.read_parquet(data_dir + "/item.parquet/*")
    cattbl = FeatureTable.read_parquet(data_dir + "/category.parquet/*")

    n_uid = usertbl.get_stats("id", "max")["id"] + 1
    n_mid = itemtbl.get_stats("id", "max")["id"] + 1
    n_cat = cattbl.get_stats("id", "max")["id"] + 1
    train_data.show(5)

    print("train size: ", train_data.size())
    print("test size: ", test_data.size())
    print("user size: ", n_uid)
    print("item size: ", n_mid)
    print("category size: ", n_cat)


    # Create model
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
        
    criterion = nn.BCELoss()

    orca_estimator = Estimator.from_torch(model=model_creator,
                                        optimizer=optim_creator,
                                        metrics=[Accuracy(), BinaryCrossEntropy()],
                                        loss=criterion,
                                        backend="torch_distributed",
                                        use_tqdm=True)

    # Train model
    orca_estimator.fit(data=train_data.df, 
                    epochs=100, 
                    batch_size=512, 
                    feature_cols=["user","item_id","cate_id","hist_item_id","seq_length","hist_cate_id"],
                    label_cols=["label"])
    res = orca_estimator.evaluate(data=test_data.df)
    for r in res:
        print(r, ":", res[r])
    stop_orca_context()
