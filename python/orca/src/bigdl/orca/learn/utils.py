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

from contextlib import closing
import socket
import sys
import tempfile
import shutil

from bigdl.dllib.utils import log4Error
from bigdl.dllib.utils.common import callBigDlFunc
from bigdl.dllib.utils.file_utils import get_file_list, is_local_path
from bigdl.orca.data import SparkXShards
from bigdl.orca.data.utils import get_size
from bigdl.orca.data.file import put_local_dir_tree_to_remote, put_local_file_to_remote,\
    get_remote_file_to_local, get_remote_dir_to_local
from bigdl.dllib.utils.utils import convert_row_to_numpy
from functools import partial
import numpy as np
from bigdl.orca.common import SafePickle
import os
import subprocess
from bigdl.dllib.utils.log4Error import *


def find_latest_checkpoint(model_dir, model_type="bigdl"):
    import os
    import re
    import datetime
    ckpt_path = None
    latest_version = None
    optim_prefix = None
    optim_regex = None
    if model_type == "bigdl":
        optim_regex = ".*\.([0-9]+)$"
    elif model_type == "pytorch":
        optim_regex = "TorchModel[0-9a-z]*\.([0-9]+)$"
    elif model_type == "tf":
        optim_regex = "TFParkTraining\.([0-9]+)$"
    else:
        ValueError("Only bigdl, pytorch and tf are supported for now.")

    file_list = get_file_list(model_dir, recursive=True)
    optim_dict = {}
    pattern_re = re.compile('(.*)(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(.*)optimMethod-'
                            + optim_regex)
    for file_path in file_list:
        matched = pattern_re.match(file_path)
        if matched is not None:
            try:
                # check if dir name is date time
                timestamp = matched.group(2)
                datetime.datetime.strptime(timestamp, '%Y-%m-%d_%H-%M-%S')
                if timestamp in optim_dict:
                    optim_dict[timestamp].append((int(matched.group(4)),
                                                  os.path.dirname(file_path),
                                                  os.path.basename(file_path).split('.')[0]))
                else:
                    optim_dict[timestamp] = [(int(matched.group(4)),
                                              os.path.dirname(file_path),
                                              os.path.basename(file_path).split('.')[0])]
            except:
                continue
    if optim_dict:
        latest_timestamp = max(optim_dict)
        latest_version, ckpt_path, optim_prefix = max(optim_dict[latest_timestamp],
                                                      key=lambda version_path: version_path[0])

    return ckpt_path, optim_prefix, latest_version


def convert_predict_rdd_to_xshard(data, prediction_rdd):
    import numpy as np

    def group_index(iter):
        for data in iter:
            size = get_size(data["x"])
            for i in range(size):
                yield size

    def transform_predict(predictions):
        # case 1: each prediction is a list of np array
        if isinstance(predictions[0], list):
            return [np.array([prediction[i] for prediction in predictions])
                    for i in range(len(predictions[0]))]
        # case 2: each prediction is a single np array
        else:
            return np.array(predictions)

    def group(iter):
        this_index = 0
        buffer = []
        this_count = None
        for (count, pred) in iter:
            if this_index == 0:
                this_count = count
            if this_index < this_count:
                buffer.append(pred)
                this_index += 1
            if this_index == this_count:
                yield transform_predict(buffer)
                buffer.clear()
                this_index = 0

    def add_pred(shard_pred):
        shard, pred = shard_pred
        shard["prediction"] = pred
        return shard

    indexed_rdd = data.rdd.mapPartitions(group_index)
    grouped_pred = indexed_rdd.zip(prediction_rdd).mapPartitions(group)
    result_rdd = data.rdd.zip(grouped_pred).map(add_pred)
    return SparkXShards(result_rdd)


def update_predict_xshards(xshards, pred_xshards):
    def update_dict(d1_d2):
        d1, d2 = d1_d2
        d1.update(d2)
        return d1

    result = SparkXShards(xshards.rdd.zip(pred_xshards.rdd).map(update_dict),
                          class_name="builtins.dict")
    return result


def add_predict_to_pd_xshards(xshards, pred_xshards):
    def add_prediction(df_preds):
        df, preds = df_preds
        preds = preds["prediction"]
        if isinstance(preds[0], np.ndarray):
            df["prediction"] = [pred.tolist() for pred in preds]
        else:
            df["prediction"] = [pred for pred in preds]
        return df

    result = SparkXShards(xshards.rdd.zip(pred_xshards.rdd).map(add_prediction),
                          class_name="pandas.core.frame.DataFrame")
    return result


def get_length(input):
    if isinstance(input, (list, tuple)):
        return get_length(input[0])
    elif isinstance(input, dict):
        return get_length(list(input.values())[0])
    else:
        return input.shape[0]


def filter_elem(input, i):
    if isinstance(input, (list, tuple)):
        return [filter_elem(elem, i) for elem in input]
    elif isinstance(input, dict):
        return {k: filter_elem(v, i) for k, v in input.items()}
    else:
        return input[i]


def convert_predict_xshards_to_dataframe(df, pred_shards, output_cols=None):
    def flatten(data):
        length = get_length(data)

        data = list(data.values())

        for i in range(length):
            # Always yield a list here
            yield filter_elem(data, i)

    pred_rdd = pred_shards.rdd.flatMap(flatten)
    result = convert_predict_rdd_to_dataframe(df, pred_rdd, output_cols)
    return result


def convert_predict_rdd_to_dataframe(df, prediction_rdd, output_cols=None):
    from pyspark.sql import Row
    from pyspark.ml.linalg import Vectors

    def convert_elem(elem):
        # list of np array
        if isinstance(elem, (list, tuple)):
            return [convert_elem(i) for i in elem]
        # dict of np array as values
        elif isinstance(elem, dict):
            return {k: convert_elem(v) for k, v in elem.items()}
        # scalar in basic type
        elif isinstance(elem, np.ScalarType):
            return float(elem)
        # np ndarray
        else:
            dim = len(elem.shape)
            if dim in [0, 1]:
                # np 1-D array
                return Vectors.dense(elem)
            else:
                # multi-dimensional array
                return elem.tolist()

    def combine(pair):
        if not output_cols:
            # a singleton list in pair[1] and stacked like [f1, f2] + [output1]
            if isinstance(pair[1], (list, tuple)) and len(pair[1]) == 1:
                return Row(*([pair[0][col] for col in pair[0].__fields__] +
                             convert_elem(pair[1])))
            else:
                # a multiple list in pair[1] and stacked like [f1, f2] + [[output1], [output2]]
                return Row(*([pair[0][col] for col in pair[0].__fields__] +
                             [convert_elem(pair[1])]))
        elif not isinstance(pair[1], (list, tuple)):
            # if pair[1] is not iterable, don't split them into list
            return Row(*([pair[0][col] for col in pair[0].__fields__] +
                         [convert_elem(pair[1])]))
        else:
            # a multiple columns in pair[1] and merged like [f1, f2] + [output1, output2]
            return Row(*([pair[0][col] for col in pair[0].__fields__] +
                         [convert_elem(item) for item in pair[1]]))

    combined_rdd = df.rdd.zip(prediction_rdd).map(combine)
    if output_cols is None:
        columns = df.columns + ["prediction"]
    else:
        columns = df.columns + output_cols
    # Converting to DataFrame will trigger the computation
    # to infer the schema of the prediction column.
    result_df = combined_rdd.toDF(columns)
    return result_df


def _stack_arrs(arrs):
    if isinstance(arrs, list):
        # stack arrs if arrs is a list.
        return np.stack(arrs)
    else:
        # do nothing if arrs are not a list.
        return arrs


def _merge_rows(results):
    if isinstance(results, dict):
        return results

    try:
        result_arrs = [_stack_arrs(l) for l in results]
    except ValueError:
        log4Error.invalidInputError(False, "Elements in the same column must have the same "
                                           "shape, please drop, pad or truncate the columns "
                                           "that do not meet this requirement.")
    if len(result_arrs) == 1:
        result_arrs = result_arrs[0]
    else:
        result_arrs = tuple(result_arrs)
    return result_arrs


def _generate_output_dict(feature_lists, label_lists, feature_cols=None, label_cols=None):
    feature_arrs = _merge_rows(feature_lists)
    if label_cols is not None:
        label_arrs = _merge_rows(label_lists)
        return {"x": feature_arrs, "y": label_arrs}
    else:
        return {"x": feature_arrs}


def _generate_feature_dict(feature_lists, label_lists, feature_cols=None, label_cols=None):
    cols = feature_cols
    feature_arrs = _merge_rows(feature_lists)
    if feature_cols and len(feature_cols) == 1:
        feature_arrs = tuple(feature_arrs)
    data = feature_arrs
    if label_cols is not None:
        cols = cols + label_cols
        label_arrs = _merge_rows(label_lists)
        if len(label_cols) == 1:
            label_arrs = tuple(label_arrs)
        data = data + label_arrs
    result = dict()
    for i in range(0, len(cols)):
        result[cols[i]] = data[i]
    return result


def _generate_output_pandas_df(feature_lists, label_lists, feature_cols, label_cols=None):
    import pandas as pd
    feature_arrs = _merge_rows(feature_lists)
    label_cols = [] if label_cols is None else label_cols
    df = pd.DataFrame(columns=feature_cols + label_cols)
    if isinstance(feature_arrs, np.ndarray):
        feature_arrs = feature_arrs.reshape(-1)
    else:
        feature_arrs = list(map(lambda x: x.reshape(-1), feature_arrs))
    for i, feature_col in enumerate(feature_cols):
        df[feature_col] = feature_arrs[i]
    if label_cols:
        label_arrs = _merge_rows(label_lists)
    for i, label_col in enumerate(label_cols):
        df[label_col] = label_arrs[i]
    return df


def arrays2others(iter, feature_cols, label_cols, shard_size=None, generate_func=None):
    def init_result_lists(first_row, cols):
        if shard_size:
            # pre allocate numpy array when shard_size is provided
            if isinstance(first_row, np.ndarray):
                return [np.empty((shard_size,) + first_row.shape, first_row.dtype)]
            if isinstance(first_row, dict):
                res = dict()
                for k, _ in first_row.items():
                    res[k] = np.empty((shard_size,) + first_row[k].shape, first_row[k].dtype)
                return res
            else:
                return [np.empty((shard_size,) + r.shape, r.dtype) for r in first_row]
        else:
            return [[] for r in cols]

    def add_row(data, results, current):
        if not isinstance(data, (list, tuple, dict)):
            arrays = [data]
        else:
            arrays = data

        iter = arrays.items() if isinstance(arrays, dict) else enumerate(arrays)
        for i, arr in iter:
            if shard_size:
                current = current % shard_size
                results[i][current] = arr
            else:
                results[i].append(arr)

    feature_lists = None
    label_lists = None
    counter = 0

    for row in iter:
        if feature_lists is None:
            feature_lists = init_result_lists(row[0], feature_cols)
        add_row(row[0], feature_lists, counter)
        if label_cols is not None:
            if label_lists is None:
                label_lists = init_result_lists(row[1], label_cols)
            add_row(row[1], label_lists, counter)
        counter += 1

        if shard_size and counter % shard_size == 0:
            # output put a shard when current shard is full
            yield generate_func(feature_lists, label_lists, feature_cols, label_cols)
            feature_lists = None
            label_lists = None

    if feature_lists is not None:
        if shard_size:
            # remove empty part of the ndarray in the last shard
            rest_size = counter % shard_size
            if isinstance(feature_lists, dict):
                feature_lists = {k: v[0:rest_size] for k, v in feature_lists.items()}
            else:
                feature_lists = [feature[0:rest_size] for feature in feature_lists]
            if label_cols is not None:
                if isinstance(label_lists, dict):
                    label_lists = {k: v[0:rest_size] for k, v in label_lists.items()}
                else:
                    label_lists = [label[0:rest_size] for label in label_lists]
        # output last shard
        yield generate_func(feature_lists, label_lists, feature_cols, label_cols)


arrays2dict = partial(arrays2others, generate_func=_generate_output_dict)
arrays2feature_dict = partial(arrays2others, generate_func=_generate_feature_dict)
arrays2pandas = partial(arrays2others, generate_func=_generate_output_pandas_df)


def transform_to_shard_dict(data, feature_cols, label_cols=None):
    def single_col_to_numpy(col_series, dtype):
        if dtype == np.ndarray:
            # In this case, directly calling to_numpy will make the result
            # ndarray have type np.object.
            # Need to explicitly specify the dtype.
            dtype = col_series.iloc[0].dtype
            return np.array([i.tolist() for i in col_series], dtype=dtype)
        else:
            return col_series.to_numpy()

    def to_shard_dict(df):
        result = dict()
        col_types = df.dtypes
        if len(feature_cols) == 1:
            featureLists = df[feature_cols[0]].tolist()
            result["x"] = np.stack(featureLists, axis=0)
        else:
            result["x"] = [single_col_to_numpy(df[feature_col], col_types[feature_col])
                           for feature_col in feature_cols]

        if label_cols:
            y = [single_col_to_numpy(df[label_col], col_types[label_col])
                 for label_col in label_cols]
            if len(label_cols) == 1:
                y = y[0]
            result["y"] = y

        return result

    data = data.transform_shard(to_shard_dict)
    data._set_class_name("builtins.dict")
    return data


def process_xshards_of_pandas_dataframe(data, feature_cols, label_cols=None, validation_data=None,
                                        mode=None):
    data = transform_to_shard_dict(data, feature_cols, label_cols)
    if mode == "fit":
        if validation_data is not None:
            invalidInputError(validation_data._get_class_name() == 'pandas.core.frame.DataFrame',
                              "train data and validation data should be both XShards of Pandas"
                              " DataFrame")
            validation_data = transform_to_shard_dict(validation_data, feature_cols, label_cols)
        return data, validation_data
    else:
        return data


def _dataframe_to_xshards(data, feature_cols, label_cols=None,
                          accept_str_col=False, shard_size=None):
    schema = data.schema
    numpy_rdd = data.rdd.map(lambda row: convert_row_to_numpy(row,
                                                              schema,
                                                              feature_cols,
                                                              label_cols,
                                                              accept_str_col))
    shard_rdd = numpy_rdd.mapPartitions(lambda x: arrays2dict(x,
                                                              feature_cols,
                                                              label_cols,
                                                              shard_size))
    return SparkXShards.lazy(shard_rdd, class_name="builtins.dict")


def dataframe_to_xshards_of_feature_dict(data, feature_cols, label_cols=None,
                                         accept_str_col=False):
    '''
    This function transforms a spark dataframe to xshards of feature dict (all feature_cols
    and label_cols will be merged to one dictionary).
    :param data: a spark dataframe
    :param feature_cols: the col names you would like to select and transform to ndarray
    :param label_cols: the target col name you would like to select and transform to ndarray
           typically, you may leave this to None since there is no difference to use
           feature_cols directly.
    :param accept_str_col: bool, states if allow str to be a valid type.

    :return: a sparkxshards of ndarray dictionary with feature_cols+label_cols as keys.
    '''
    from bigdl.orca import OrcaContext
    schema = data.schema
    shard_size = OrcaContext._shard_size
    numpy_rdd = data.rdd.map(lambda row: convert_row_to_numpy(row,
                                                              schema,
                                                              feature_cols,
                                                              label_cols,
                                                              accept_str_col))
    shard_rdd = numpy_rdd.mapPartitions(lambda x: arrays2feature_dict(x,
                                                                      feature_cols,
                                                                      label_cols,
                                                                      shard_size))
    return SparkXShards(shard_rdd)


def dataframe_to_xshards_of_pandas_df(data, feature_cols, label_cols=None, accept_str_col=False):
    '''
    This function transform a spark dataframe to xshards of pandas dataframe.
    :param data: a spark dataframe
    :param feature_cols: the col names you would like to select and transform to pandas
    :param label_cols: the target col name you would like to select and transform to pandas
           typically, you may leave this to None since there is no difference to use
           feature_cols directly.
    :param accept_str_col: bool, states if allow str to be a valid type.

    :return: a sparkxshards of pandas dataframe with feature_cols+label_cols as header.
    '''
    from bigdl.orca import OrcaContext
    schema = data.schema
    shard_size = OrcaContext._shard_size
    numpy_rdd = data.rdd.map(lambda row: convert_row_to_numpy(row,
                                                              schema,
                                                              feature_cols,
                                                              label_cols,
                                                              accept_str_col))
    shard_rdd = numpy_rdd.mapPartitions(lambda x: arrays2pandas(x,
                                                                feature_cols,
                                                                label_cols,
                                                                shard_size))
    return SparkXShards(shard_rdd)


def dataframe_to_xshards(data, validation_data, feature_cols, label_cols, mode="fit",
                         num_workers=None, accept_str_col=False, shard_size=None):
    from pyspark.sql import DataFrame
    valid_mode = {"fit", "evaluate", "predict"}
    invalidInputError(mode in valid_mode,
                      f"invalid mode {mode} "
                      f"mode should be one of {valid_mode}")
    invalidInputError(validation_data is None or isinstance(validation_data, DataFrame),
                      "validation data must be a spark DataFrame when data is a DataFrame")
    invalidInputError(feature_cols is not None,
                      "feature_col must be provided if data is a spark dataframe")

    if mode != "predict":
        invalidInputError(label_cols is not None,
                          "label_cols must be provided if data is a spark dataframe")
        # avoid empty partition for worker
        if data.rdd.getNumPartitions() < num_workers:
            data = data.repartition(num_workers)
        if validation_data is not None:
            num_data_part = data.rdd.getNumPartitions()
            validation_data = validation_data.repartition(num_data_part)

    data = _dataframe_to_xshards(data, feature_cols, label_cols, accept_str_col, shard_size)
    if validation_data is not None:
        validation_data = _dataframe_to_xshards(validation_data, feature_cols, label_cols,
                                                accept_str_col, shard_size)

    return data, validation_data


def maybe_dataframe_to_xshards(data, validation_data, feature_cols, label_cols, mode="fit",
                               num_workers=None, accept_str_col=False, shard_size=None):
    from pyspark.sql import DataFrame
    if isinstance(data, DataFrame):
        data, validation_data = dataframe_to_xshards(data, validation_data,
                                                     feature_cols=feature_cols,
                                                     label_cols=label_cols,
                                                     mode=mode,
                                                     num_workers=num_workers,
                                                     accept_str_col=accept_str_col,
                                                     shard_size=shard_size)
    return data, validation_data


def bigdl_metric_results_to_dict(results):
    return dict([(r.method, r.result) for r in results])


def make_data_creator(refs):
    def data_creator(config, batch_size):
        return refs

    return data_creator


def openvino_output_to_sdf(df, rdd, names, shapes):
    return callBigDlFunc("float", "openVINOOutputToSDF", df, rdd, names, shapes)


def get_arrow_hex_str(batched_data, names):
    import pyarrow as pa
    sink = pa.BufferOutputStream()
    pred_arrow = pa.record_batch(batched_data, names=names)
    with pa.ipc.new_stream(sink, pred_arrow.schema) as writer:
        writer.write_batch(pred_arrow)
    pred_arrow = sink.getvalue().hex()
    pred_arrow = pred_arrow.decode("utf-8")
    sink.close()
    return pred_arrow


def data_length(data):
    x = data["x"]
    if isinstance(x, np.ndarray):
        return x.shape[0]
    else:
        return x[0].shape[0]


def save_pkl(data, path):
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import uuid
        file_name = str(uuid.uuid1()) + ".pkl"
        temp_path = os.path.join(tempfile.gettempdir(), file_name)
        with open(temp_path, 'wb') as f:
            SafePickle.dump(data, f)
        put_local_file_to_remote(temp_path, path)
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3')
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        content = SafePickle.dumps(data)
        s3_client.put_object(Bucket=bucket, Key=key, Body=content)
    else:
        if path.startswith("file://"):
            path = path[len("file://"):]
        with open(path, 'wb') as f:
            SafePickle.dump(data, f)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def find_ip_and_free_port(pre_iter):
    from pyspark import BarrierTaskContext
    tc = BarrierTaskContext().get()
    infos = tc.getLocalProperty("addresses").split(",")
    address = infos[tc.partitionId()].split(":")[0]
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return [f"{address}:{s.getsockname()[1]}"]


def get_rank(cluster_info):
    # As task placement may not be identical between two different jobs,
    # we cannot simply index cluster_info using partitionId to get current
    # ip and port.
    # The approach here is to first get all tasks' ip in this job and compute
    # a local rank by counting how many tasks has the same ip but with lower id.
    # We then use the local rank to find the right slot in cluster_info to find
    # the right global_rank.
    from pyspark import BarrierTaskContext
    tc = BarrierTaskContext().get()
    infos = tc.getLocalProperty("addresses").split(",")
    idx = tc.partitionId()
    local_ip = infos[idx].split(":")[0]
    local_rank = 0
    for i in range(0, idx):
        if infos[i].startswith(local_ip):
            local_rank += 1
    global_rank = -1
    local_count = 0
    for node in cluster_info:
        if node.startswith(local_ip):
            local_count += 1
        global_rank += 1
        if local_count == local_rank + 1:
            break
    return global_rank


def duplicate_stdout_stderr_to_file(log_path):
    tee = subprocess.Popen(["tee", log_path], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def get_specific_object_from_callbacks(class_type, callbacks):
    for c in callbacks:
        if isinstance(c, class_type):
            return c
    return None


def replace_specific_object_from_callbacks(callbacks, original_class_type,
                                           new_class_type, rank=None):
    for c in callbacks:
        if isinstance(c, original_class_type):
            callbacks.remove(c)
            new_callback = new_class_type(c, rank)
            callbacks.append(new_callback)


def get_replaced_path(original_filepath):
    base_name = os.path.basename(original_filepath)
    temp_dir = tempfile.mkdtemp()
    return os.path.join(temp_dir, base_name)


def process_tensorboard_in_callbacks(callbacks, mode="train", rank=None):
    import tensorflow as tf

    class EpochCopyCallback(tf.keras.callbacks.Callback):
        def __init__(self, local_dir, remote_dir, rank=None):
            super(EpochCopyCallback, self).__init__()
            self.local_dir = local_dir
            self.remote_dir = remote_dir
            self.rank = rank

        def on_epoch_end(self, epoch, logs=None):
            if self.rank is not None:
                if self.rank == 0:
                    put_local_dir_tree_to_remote(self.local_dir, self.remote_dir)

    class TrainBatchCopyCallback(tf.keras.callbacks.Callback):
        def __init__(self, local_dir, remote_dir, freq, rank=None):
            super(TrainBatchCopyCallback, self).__init__()
            self.local_dir = local_dir
            self.remote_dir = remote_dir
            self.freq = freq
            self.rank = rank

        def on_train_batch_end(self, batch, logs=None):
            if self.rank is not None:
                if self.rank == 0:
                    if batch % self.freq == 0:
                        put_local_dir_tree_to_remote(self.local_dir, self.remote_dir)

    class BatchCopyCallback(tf.keras.callbacks.Callback):
        def __init__(self, local_dir, remote_dir, freq, rank=None):
            super(BatchCopyCallback, self).__init__()
            self.local_dir = local_dir
            self.remote_dir = remote_dir
            self.freq = freq
            self.rank = rank

        def on_test_batch_end(self, batch, logs=None):
            if self.rank is not None:
                if self.rank == 0:
                    if batch % self.freq == 0:
                        put_local_dir_tree_to_remote(self.local_dir, self.remote_dir)

    tensorboard = get_specific_object_from_callbacks(tf.keras.callbacks.TensorBoard,
                                                     callbacks)
    if tensorboard:
        original_log_dir = tensorboard.log_dir
        replaced_log_dir = get_replaced_path(original_log_dir)
        tensorboard.log_dir = replaced_log_dir

        if tensorboard.update_freq == 'epoch':
            # create copy callback for epoch
            copy_callback = EpochCopyCallback(replaced_log_dir, original_log_dir, rank)
        else:
            # to avoid frequent copy, set update freq > 10
            update_freq = tensorboard.update_freq if tensorboard.update_freq > 10 \
                else 10
            if mode == "fit":
                # create copy callback for batch
                copy_callback = TrainBatchCopyCallback(replaced_log_dir, original_log_dir,
                                                       update_freq, rank)
            else:
                copy_callback = BatchCopyCallback(replaced_log_dir, original_log_dir,
                                                  update_freq, rank)
        callbacks.append(copy_callback)
        return replaced_log_dir
    return None


def get_latest_checkpoint(checkpoint_dir):
    import tensorflow as tf
    if is_local_path(checkpoint_dir):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        return checkpoint_path
    else:
        try:
            temp_dir = tempfile.mkdtemp()
            # get checkpoint file
            remote_checkpoint_file = os.path.join(checkpoint_dir, "checkpoint")
            local_checkpoint_file = os.path.join(temp_dir, "checkpoint")
            get_remote_file_to_local(remote_checkpoint_file, local_checkpoint_file)
            ckpt_name = None
            with open(local_checkpoint_file) as f:
                import re
                lines = f.readlines()
                for line in lines:
                    m = re.compile("^model_checkpoint_path: \"(.*)\"$").match(line)
                    if m:
                        ckpt_name = m.group(1)
                        break
            if ckpt_name:
                return os.path.join(checkpoint_dir, ckpt_name)
            return None
        finally:
            shutil.rmtree(temp_dir)


def load_model(filepath, custom_objects=None, compile=True):
    import tensorflow as tf
    if is_local_path(filepath):
        model = tf.keras.models.load_model(filepath,
                                           custom_objects=custom_objects,
                                           compile=compile
                                           )
    else:
        file_name = os.path.basename(filepath)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file_name)
        try:
            if filepath.endswith('.h5') or filepath.endswith('.keras'):
                get_remote_file_to_local(filepath, temp_path)
            else:
                get_remote_dir_to_local(filepath, temp_path)

            model = tf.keras.models.load_model(temp_path,
                                               custom_objects=custom_objects,
                                               compile=compile
                                               )
        finally:
            shutil.rmtree(temp_dir)
    return model


def save_model(model, filepath, overwrite=True, include_optimizer=True, save_format=None,
               signatures=None, options=None, filemode=None):
    if is_local_path(filepath):
        model.save(filepath, overwrite=overwrite, include_optimizer=include_optimizer,
                   save_format=save_format, signatures=signatures, options=options)
    else:
        file_name = os.path.basename(filepath)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file_name)
        try:
            model.save(temp_path, overwrite=overwrite, include_optimizer=include_optimizer,
                       save_format=save_format, signatures=signatures, options=options)
            if save_format == 'h5' or filepath.endswith('.h5') or filepath.endswith('.keras'):
                # hdf5 format
                put_local_file_to_remote(temp_path, filepath, filemode)
            else:
                # tf format
                put_local_dir_tree_to_remote(temp_path, filepath)
        finally:
            shutil.rmtree(temp_dir)


def get_driver_node_ip():
    """
    Returns the IP address of the current node.

    :return: the IP address of the current node.
    """
    import ray
    return ray._private.services.get_node_ip_address()


def get_partition_id():
    from pyspark import BarrierTaskContext, TaskContext
    if TaskContext.get():
        partition_id = TaskContext.get().partitionId()
    else:
        partition_id = BarrierTaskContext().get().partitionId()
    return partition_id
