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


def convert_predict_to_xshard(prediction_rdd):
    import numpy as np
    from zoo.orca.data import SparkXShards

    def transform_predict(iter):
        predictions = list(iter)
        # list of np array
        if isinstance(predictions[0], list):
            predictions = np.array(predictions).T.tolist()
            result = [np.array(predict) for predict in predictions]
            return [{'prediction': result}]
        # np array
        else:
            return [{'prediction': np.array(predictions)}]

    return SparkXShards(prediction_rdd.mapPartitions(transform_predict))


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
    if label_lists is not None:
        label_arrs = [np.stack(l) for l in label_lists]
        if len(label_arrs) == 1:
            label_arrs = label_arrs[0]
        return [{"x": feature_arrs, "y": label_arrs}]

    return [{"x": feature_arrs}]


def dataframe_to_xshards(data, feature_cols, label_cols=None):
    schema = data.schema
    numpy_rdd = data.rdd.map(lambda row: convert_row_to_numpy(row,
                                                              schema,
                                                              feature_cols,
                                                              label_cols))
    shard_rdd = numpy_rdd.mapPartitions(lambda x: arrays2dict(x,
                                                              feature_cols,
                                                              label_cols))
    return SparkXShards(shard_rdd)


def maybe_dataframe_to_xshards(data, validation_data, feature_cols, label_cols, mode="fit"):
    from pyspark.sql import DataFrame
    if isinstance(data, DataFrame):
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

        data = dataframe_to_xshards(data, feature_cols, label_cols)
        if validation_data is not None:
            validation_data = dataframe_to_xshards(validation_data, feature_cols, label_cols)
    return data, validation_data
