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
import time
import os
import sys

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import desc, rank, col, udf
from pyspark.sql.window import Window
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.data.file import exists, makedirs
from bigdl.orca.learn.tf.estimator import Estimator
from bigdl.orca import init_orca_context, stop_orca_context
from model import *


EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
SEED = 3


def build_model(model_type, n_uid, n_mid, n_cat, lr, data_type):
    if model_type == 'DNN':
        model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr)
    elif model_type == 'PNN':
        model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr)
    elif model_type == 'Wide':
        model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr)
    elif model_type == 'DIN':
        model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, lr)
    elif model_type == 'DIN-V2-gru-att-gru':
        model = Model_DIN_V2_Gru_att_Gru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                         ATTENTION_SIZE, lr)
    elif model_type == 'DIN-V2-gru-gru-att':
        model = Model_DIN_V2_Gru_Gru_att(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                         ATTENTION_SIZE, lr)
    elif model_type == 'DIN-V2-gru-qa-attGru':
        model = Model_DIN_V2_Gru_QA_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE, lr)
    elif model_type == 'DIN-V2-gru-vec-attGru':
        model = Model_DIN_V2_Gru_Vec_attGru(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                            ATTENTION_SIZE, lr)
    elif model_type == 'DIEN':
        model = Model_DIN_V2_Gru_Vec_attGru_Neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                                ATTENTION_SIZE, lr, data_type)
    else:
        print("Invalid model_type: %s", model_type)
        sys.exit(1)
    return model


def align_input_features(model):
    input_phs = [model.uid_batch_ph, model.mid_his_batch_ph, model.cat_his_batch_ph, model.mask,
                 model.seq_len_ph, model.mid_batch_ph, model.cat_batch_ph]
    feature_cols = ['user', 'item_hist_seq', 'category_hist_seq', 'item_hist_seq_mask',
                    'item_hist_seq_len', 'item', 'category']
    if model.use_negsampling:
        input_phs.extend([model.noclk_mid_batch_ph, model.noclk_cat_batch_ph])
        feature_cols.extend(['neg_item_hist_seq', 'neg_category_hist_seq'])
    return [input_phs, feature_cols]


def load_dien_data(data_dir):
    tbl = FeatureTable.read_parquet(data_dir + "/data")
    windowSpec1 = Window.partitionBy("user").orderBy(desc("time"))
    tbl = tbl.append_column("rank1", rank().over(windowSpec1))
    tbl = tbl.filter(col('rank1') == 1)
    train_data, test_data = tbl.split([0.8, 0.2], seed=1)
    usertbl = FeatureTable.read_parquet(data_dir + "/user_index/*")
    itemtbl = FeatureTable.read_parquet(data_dir + "/item_index/*")
    cattbl = FeatureTable.read_parquet(data_dir + "/category_index/*")
    n_uid = usertbl.get_stats("id", "max")["id"] + 1
    n_mid = itemtbl.get_stats("id", "max")["id"] + 1
    n_cat = cattbl.get_stats("id", "max")["id"] + 1
    train_data.show()
    print("train size: ", train_data.size())
    print("test size: ", test_data.size())
    print("user size: ", n_uid)
    print("item size: ", n_mid)
    return train_data, test_data, n_uid, n_mid, n_cat


if __name__ == '__main__':
    time_start = time.time()
    parser = argparse.ArgumentParser(description='Tensorflow DIEN Training/Inference')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn, standalone or spark-submit.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=2,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=8,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--model_type', default='DIN-V2-gru-vec-attGru', type=str,
                        help='model type: DIEN (default)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1, type=int, help='train epoch')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument("--data_type", type=str, default='FP32', help="data type: FP32 or FP16")
    parser.add_argument('--model_dir', default='snapshot', type=str,
                        help='snapshot directory name (default: snapshot)')
    parser.add_argument('--data_dir', type=str, default="./preprocessed", help='data directory')
    args = parser.parse_args()

    if args.cluster_mode == "local":
        init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        init_orca_context("standalone", master=args.master,
                          cores=args.executor_cores, num_nodes=args.num_executor,
                          memory=args.executor_memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          init_ray_on_spark=False)
    elif args.cluster_mode == "yarn":
        init_orca_context("yarn-client", cores=args.executor_cores,
                          num_nodes=args.num_executor, memory=args.executor_memory,
                          driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                          init_ray_on_spark=False)
    elif args.cluster_mode == "spark-submit":
        init_orca_context("spark-submit")

    train_data, test_data, n_uid, n_mid, n_cat = load_dien_data(args.data_dir)

    model = build_model(args.model_type, n_uid, n_mid, n_cat, args.lr, args.data_type)
    [inputs, feature_cols] = align_input_features(model)

    estimator = Estimator.from_graph(inputs=inputs, outputs=[model.y_hat],
                                     labels=[model.target_ph], loss=model.loss,
                                     optimizer=model.optim, model_dir=args.model_dir,
                                     metrics={'loss': model.loss, 'accuracy': model.accuracy})

    estimator.fit(train_data.df, epochs=args.epochs, batch_size=args.batch_size,
                  feature_cols=feature_cols, label_cols=['label'], validation_data=test_data.df)

    ckpts_dir = os.path.join(args.model_dir, 'ckpts/')
    if not exists(ckpts_dir):
        makedirs(ckpts_dir)
    snapshot_path = ckpts_dir + "ckpt_" + args.model_type
    estimator.save_tf_checkpoint(snapshot_path)
    time_train = time.time()
    print(f"perf training time: {(time_train - time_start):.2f}")

    result = estimator.evaluate(test_data.df, args.batch_size, feature_cols=feature_cols,
                                label_cols=['label'])

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                              labelCol="label_t",
                                              metricName="areaUnderROC")
    prediction_df = estimator.predict(test_data.df, feature_cols=feature_cols)
    prediction_df.cache()
    transform_label = udf(lambda x: int(x[1]), "int")
    prediction_df = prediction_df.withColumn('label_t', transform_label(col('label')))
    auc = evaluator.evaluate(prediction_df)

    time_end = time.time()
    print('evaluation result:', result)
    print("evaluation AUC score is: ", auc)
    print(f"perf evaluation time: {(time_end - time_train):.2f}")

    stop_orca_context()
