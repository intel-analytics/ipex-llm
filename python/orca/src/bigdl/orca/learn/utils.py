#
# Copyright 2018 Analytics Zoo Authors.
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

from zoo.common.utils import get_file_list
from zoo.orca.data import SparkXShards
from zoo.orca.data.utils import get_size
from zoo.util.utils import convert_row_to_numpy
import numpy as np


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
    from zoo.orca.data import SparkXShards

    def group_index(iter):
        for data in iter:
            size = get_size(data["x"])
            for i in range(size):
                print(size)
                yield size

    def transform_predict(predictions):
        # list of np array
        if isinstance(predictions[0], list):
            predictions = np.array(predictions).T.tolist()
            result = [np.array(predict) for predict in predictions]
            return result
        # np array
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


def update_predict_xshards(xshard, pred_xshards):
    def updates(d1_d2):
        d1, d2 = d1_d2
        d1.update(d2)
        return d1

    result = SparkXShards(xshard.rdd.zip(pred_xshards.rdd).map(updates))
    return result


def convert_predict_xshards_to_dataframe(df, pred_shards):
    def flatten(data):
        data = data["prediction"]
        is_list = isinstance(data, list)
        is_tuple = isinstance(data, tuple)
        if is_list or is_tuple:
            length = data[0].shape[0]
            ls_data = data
        else:
            length = data.shape[0]
            ls_data = [data]

        for i in range(length):
            row = [elem[i] for elem in ls_data]
            if is_list:
                yield row
            elif is_tuple:
                yield tuple(row)
            else:
                yield row[0]

    pred_rdd = pred_shards.rdd.flatMap(flatten)
    result = convert_predict_rdd_to_dataframe(df, pred_rdd)
    return result


def convert_predict_rdd_to_dataframe(df, prediction_rdd):
    from pyspark.sql import Row
    from pyspark.sql.types import StructType, StructField, FloatType, ArrayType
    from pyspark.ml.linalg import VectorUDT, Vectors

    def combine(pair):
        # list of np array
        if isinstance(pair[1], list):
            row = Row(*([pair[0][col] for col in pair[0].__fields__] +
                        [[Vectors.dense(elem) for elem in pair[1]]]))
            return row, ArrayType(VectorUDT())
        # scalar
        elif len(pair[1].shape) == 0:
            row = Row(*([pair[0][col] for col in pair[0].__fields__] + [float(pair[1].item(0))]))
            return row, FloatType()
        # np array
        else:
            row = Row(*([pair[0][col] for col in pair[0].__fields__] + [Vectors.dense(pair[1])]))
            return row, VectorUDT()

    combined_rdd = df.rdd.zip(prediction_rdd).map(combine)
    type = combined_rdd.map(lambda data: data[1]).first()
    result_rdd = combined_rdd.map(lambda data: data[0])
    schema = StructType(df.schema.fields + [StructField('prediction', type)])
    result_df = result_rdd.toDF(schema)
    return result_df


def arrays2dict(iter, feature_cols, label_cols):

    feature_lists = [[] for col in feature_cols]
    if label_cols is not None:
        label_lists = [[] for col in label_cols]
    else:
        label_lists = None

    for row in iter:
        # feature
        if not isinstance(row[0], list):
            features = [row[0]]
        else:
            features = row[0]

        for i, arr in enumerate(features):
            feature_lists[i].append(arr)

        # label
        if label_cols is not None:
            if not isinstance(row[1], list):
                labels = [row[1]]
            else:
                labels = row[1]

            for i, arr in enumerate(labels):
                label_lists[i].append(arr)

    feature_arrs = [np.stack(l) for l in feature_lists]
    if len(feature_arrs) == 1:
        feature_arrs = feature_arrs[0]
    else:
        feature_arrs = tuple(feature_arrs)
    if label_lists is not None:
        label_arrs = [np.stack(l) for l in label_lists]
        if len(label_arrs) == 1:
            label_arrs = label_arrs[0]
        else:
            label_arrs = tuple(label_arrs)
        return [{"x": feature_arrs, "y": label_arrs}]

    return [{"x": feature_arrs}]


def _dataframe_to_xshards(data, feature_cols, label_cols=None):
    schema = data.schema
    numpy_rdd = data.rdd.map(lambda row: convert_row_to_numpy(row,
                                                              schema,
                                                              feature_cols,
                                                              label_cols))
    shard_rdd = numpy_rdd.mapPartitions(lambda x: arrays2dict(x,
                                                              feature_cols,
                                                              label_cols))
    return SparkXShards(shard_rdd)


def dataframe_to_xshards(data, validation_data, feature_cols, label_cols, mode="fit"):
    from pyspark.sql import DataFrame
    valid_mode = {"fit", "evaluate", "predict"}
    assert mode in valid_mode, f"invalid mode {mode} " \
                               f"mode should be one of {valid_mode}"
    assert validation_data is None or isinstance(validation_data, DataFrame), \
        "validation data must be a spark DataFrame when data is a DataFrame"
    assert feature_cols is not None, \
        "feature_col must be provided if data is a spark dataframe"

    if mode != "predict":
        assert label_cols is not None, \
            "label_cols must be provided if data is a spark dataframe"

    data = _dataframe_to_xshards(data, feature_cols, label_cols)
    if validation_data is not None:
        validation_data = _dataframe_to_xshards(validation_data, feature_cols, label_cols)

    return data, validation_data


def maybe_dataframe_to_xshards(data, validation_data, feature_cols, label_cols, mode="fit"):
    from pyspark.sql import DataFrame
    if isinstance(data, DataFrame):
        data, validation_data = dataframe_to_xshards(data, validation_data,
                                                     feature_cols=feature_cols,
                                                     label_cols=label_cols, mode=mode)
    return data, validation_data
