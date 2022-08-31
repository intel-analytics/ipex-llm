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

import os
import time
import yaml
import pickle
import random
import numpy as np
from argparse import ArgumentParser

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable, StringIndex
from bigdl.dllib.utils.log4Error import *


conf = {"spark.network.timeout": "10000000",
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
        "spark.driver.maxResultSize": "40G"}


def prepare_data(rows, config):
    # Columns order:
    # item_hist_seq, item, label, category, category_hist_seq, user, item_hist_seq_len
    lengths_x = [row[6] for row in rows]
    seqs_mid = [row[0] for row in rows]
    seqs_cat = [row[4] for row in rows]

    n_samples = len(seqs_mid)
    maxlen_x = np.max(lengths_x)
    maxlen_padding = (config["exact_maxlen"] != 0)
    if maxlen_padding:
        maxlen_x = max(config["history_maxlen"], maxlen_x)

    mid_his = np.zeros((n_samples, maxlen_x)).astype("int64")
    cat_his = np.zeros((n_samples, maxlen_x)).astype("int64")
    dtype = config["data_type"]
    if dtype == "fp32" or dtype == "bfloat16" or dtype == "int8":
        data_type = "float32"
    elif dtype == "fp16":
        data_type = "float16"
    else:
        invalidInputError(False, "Invalid model data type: %s" % dtype)
    mid_mask = np.zeros((n_samples, maxlen_x)).astype(data_type)
    for idx, [s_x, s_y] in enumerate(zip(seqs_mid, seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y

    uids = np.array([row[5] for row in rows])
    mids = np.array([row[1] for row in rows])
    cats = np.array([row[3] for row in rows])
    target = np.array([row[2] for row in rows])
    sl = np.array(lengths_x)
    feed_data = [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl]
    return feed_data


def infer_main(partition, config):
    import tensorflow as tf
    from tensorflow.core.protobuf import rewriter_config_pb2
    from utils import calc_auc

    seed = config["seed"]
    np.random.seed(seed)
    random.seed(seed)
    if tf.__version__[0] == "1":
        tf.compat.v1.set_random_seed(seed)
    elif tf.__version__[0] == "2":
        tf.random.set_seed(seed)

    init_start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(config["cores_per_instance"])
    os.environ["KMP_AFFINITY"] = config["kmp_affinity"]
    os.environ["KMP_SETTINGS"] = "1"
    if config["AMX"]:
        os.environ["DNNL_MAX_CPU_ISA"] = "AVX512_CORE_AMX"
    if config["dnnl_verbose"]:
        os.environ["MKL_VERBOSE"] = "1"
        os.environ["DNNL_VERBOSE"] = "1"
    init_end = time.time()
    init_time = init_end - init_start

    model_restore_start = time.time()
    with tf.io.gfile.GFile(config["model_path"], "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    input_layers = ["Inputs/uid_batch_ph",
                    "Inputs/mid_batch_ph",
                    "Inputs/cat_batch_ph",
                    "Inputs/mid_his_batch_ph",
                    "Inputs/cat_his_batch_ph",
                    "Inputs/mask",
                    "Inputs/target_ph"]

    if config["graph_type"] == "dynamic":
        input_layers.append("Inputs/seq_len_ph")

    input_tensor = [graph.get_tensor_by_name(x + ":0") for x in input_layers]
    output_layers = ["dien/fcn/add_6",
                     "dien/fcn/Metrics/Mean_1"]
    output_tensor = [graph.get_tensor_by_name(x + ":0") for x in output_layers]

    dtype = config["data_type"]
    if dtype == "bfloat16" or dtype == "int8":
        graph_options = tf.compat.v1.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                remapping=rewriter_config_pb2.RewriterConfig.AGGRESSIVE,
                auto_mixed_precision_mkl=rewriter_config_pb2.RewriterConfig.ON))
    elif dtype == "fp32":
        graph_options = tf.compat.v1.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                remapping=rewriter_config_pb2.RewriterConfig.AGGRESSIVE))
    else:
        invalidInputError(False, f"Unsupported data type: {dtype}")

    session_config = tf.compat.v1.ConfigProto(graph_options=graph_options)
    session_config.intra_op_parallelism_threads = config["num_intra_threads"]
    session_config.inter_op_parallelism_threads = config["num_inter_threads"]
    sess = tf.compat.v1.Session(graph=graph, config=session_config)
    model_restore_end = time.time()
    model_restore_time = model_restore_end - model_restore_start

    infer_start = time.time()
    accuracy_sum = 0.
    stored_arr = []
    eval_time = 0
    sample_freq = 9999999
    options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    data_size = 0
    i = 0
    buffer = []
    while True:
        run = False
        row = next(partition, [])
        if not row:  # End of the partition
            if len(buffer) > 0:
                run = True
            else:
                break
        elif len(buffer) == config["batch_size"]:
            run = True
        else:
            buffer.append(row)
        if run:
            data_size += len(buffer)
            feed_data = prepare_data(buffer, config)
            i += 1

            start_time = time.time()
            if i % sample_freq == 0:
                prob, acc = sess.run(output_tensor,
                                     options=options,
                                     run_metadata=run_metadata,
                                     feed_dict=dict(zip(input_tensor, feed_data)))
            else:
                prob, acc = sess.run(output_tensor,
                                     feed_dict=dict(zip(input_tensor, feed_data)))

            end_time = time.time()
            eval_time += end_time - start_time

            target = feed_data[6]

            accuracy_sum += acc
            prob_1 = prob[:, 0].tolist()
            target_1 = target[:, 0].tolist()

            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, t])
            if not row:
                break
            buffer = [row]
    if data_size > 0:
        test_auc = calc_auc(stored_arr)
        accuracy_sum = accuracy_sum / i
        infer_end = time.time()
        infer_time = infer_end - infer_start
        total_recommendations = data_size
        # total_recommendations = i * config["batch_size"]
        thpt_forward_pass = float(i * config["batch_size"]) / float(eval_time)
        return [[test_auc, accuracy_sum, total_recommendations,
                 (init_time, model_restore_time, infer_time, thpt_forward_pass)]]
    else:
        return [[0, 0, 0, 0]]


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cluster_mode", type=str, default="local",
                        help="The cluster mode, such as local, yarn, standalone or spark-submit.")
    parser.add_argument("--master", type=str, default=None,
                        help="The master url, only used when cluster mode is standalone.")
    parser.add_argument("--executor_cores", type=int, default=48,
                        help="The executor core number.")
    parser.add_argument("--executor_memory", type=str, default="160g",
                        help="The executor memory.")
    parser.add_argument("--num_executors", type=int, default=8,
                        help="The number of executors.")
    parser.add_argument("--driver_cores", type=int, default=4,
                        help="The driver core number.")
    parser.add_argument("--driver_memory", type=str, default="36g",
                        help="The driver memory.")
    parser.add_argument("--input_transaction", type=str, required=True,
                        help="The path to the user transaction file.")
    parser.add_argument("--input_meta", type=str, required=True,
                        help="The path to the item metadata file.")
    parser.add_argument("--index_folder", type=str, default="./",
                        help="The folder for user, item and category string indices.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    num_tasks = args.executor_cores * args.num_executors
    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores, memory=args.executor_memory)
        num_tasks = args.executor_cores
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executors,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores,
                               driver_memory=args.driver_memory, conf=conf,
                               extra_python_lib="utils.py")
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executors, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=conf, extra_python_lib="utils.py")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn' and 'standalone', "
                          "but got " + args.cluster_mode)

    begin = time.time()
    transaction_tbl = FeatureTable.read_json(args.input_transaction).select(
        ["reviewerID", "asin", "unixReviewTime"]) \
        .rename({"reviewerID": "user", "asin": "item", "unixReviewTime": "time"}) \
        .dropna(columns=["user", "item"])
    transaction_tbl.cache()

    def process_single_meta(row):
        obj = eval(row)
        cat = obj["categories"][0][-1]
        return [obj["asin"], cat]

    item_tbl = FeatureTable.read_text(args.input_meta)\
        .apply("value", "value", process_single_meta, dtype="array<string>")\
        .apply("value", "item", lambda x: x[0])\
        .apply("value", "category", lambda x: x[1])\
        .drop("value")

    # item_tbl = FeatureTable.read_csv(args.input_meta, delimiter="\t", names=["item", "category"])

    # Currently long id is not supported for add_negative_samples and add_value_features,
    # cast to int.
    with open(args.index_folder + "vocs/cat_voc.pkl", "rb") as f:
        category_df = sc.parallelize(list(pickle.load(f).items())).toDF(["category", "id"])
        category_index = StringIndex(category_df, "category").cast("id", "int")
    with open(args.index_folder + "vocs/mid_voc.pkl", "rb") as f:
        item_df = sc.parallelize(list(pickle.load(f).items())).toDF(["item", "id"])
        item_index = StringIndex(item_df, "item").cast("id", "int")
    with open(args.index_folder + "vocs/uid_voc.pkl", "rb") as f:
        user_df = sc.parallelize(list(pickle.load(f).items())).toDF(["user", "id"])
        user_index = StringIndex(user_df, "user").cast("id", "int")
    # user_index = StringIndex.read_parquet(args.index_folder + "user.parquet")
    # item_index = StringIndex.read_parquet(args.index_folder + "item.parquet")
    # category_index = StringIndex.read_parquet(args.index_folder + "category.parquet")
    item_size = item_index.size()

    item_tbl = item_tbl\
        .encode_string(["item", "category"], [item_index, category_index])\
        .fillna(0, ["item", "category"])
    item_tbl.cache()

    # Encode users should be performed after generating history sequence.
    # Otherwise unknown users will be all merged to user 0, resulting in data loss
    # and also the task for user 0 might be OOM.
    full_tbl = transaction_tbl\
        .encode_string(["item"], [item_index]) \
        .fillna(0, ["item"])\
        .add_hist_seq(cols=["item"], user_col="user",
                      sort_col="time", min_len=2, max_len=100, num_seqs=1)\
        .add_negative_samples(item_size, item_col="item", neg_num=1)\
        .add_value_features(columns=["item", "item_hist_seq"],
                            dict_tbl=item_tbl, key="item", value="category")\
        .encode_string(["user"], [user_index]) \
        .fillna(0, ["user"])

    full_tbl = full_tbl.drop("time") \
        .apply("label", "label", lambda x: [float(x), 1 - float(x)], "array<float>") \
        .apply("item_hist_seq", "item_hist_seq_len", len, "int")

    df = full_tbl.df.repartition(num_tasks)
    df = df.sortWithinPartitions("item_hist_seq_len", ascending=False)

    config = yaml.safe_load(open("config_runtime.yaml", "r"))
    rdd = df.rdd
    eval_res = rdd.mapPartitions(lambda iter: infer_main(iter, config)).collect()
    data_size = 0
    for test_auc, test_accuracy, total_recommendations, stats in eval_res:
        if total_recommendations > 0:  # Print the evaluation result of each task/partition
            init_time, model_restore_time, infer_time, thpt_forward_pass = stats
            data_size += total_recommendations
            print(f"<Forward pass> ({total_recommendations} samples) infer_time={infer_time} | "
                  f"model_restore_time={model_restore_time} | init_time={init_time} | "
                  f"throughput={thpt_forward_pass} | accurary={test_accuracy} | auc={test_auc}")

    end = time.time()
    print(f"DIEN end-to-end inference time: {(end - begin):.2f}s")
    print("Total number of processed records: {}".format(data_size))
    stop_orca_context()
