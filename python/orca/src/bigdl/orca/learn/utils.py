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
        # np ndarray
        else:
            dim = len(pair[1].shape)
            if dim == 1:
                # np 1-D array
                row = Row(*([pair[0][col] for col in pair[0].__fields__] +
                            [Vectors.dense(pair[1])]))
                return row, VectorUDT()
            else:
                # multi-dimensional array
                structType = FloatType()
                for _ in range(dim):
                    structType = ArrayType(structType)
                row = Row(*([pair[0][col] for col in pair[0].__fields__] + [pair[1].tolist()]))
                return row, structType

    combined_rdd = df.rdd.zip(prediction_rdd).map(combine)
    type = combined_rdd.map(lambda data: data[1]).first()
    result_rdd = combined_rdd.map(lambda data: data[0])
    schema = StructType(df.schema.fields + [StructField('prediction', type)])
    result_df = result_rdd.toDF(schema)
    return result_df


def arrays2dict(iter, feature_cols, label_cols, shard_size=None):
    def init_result_lists():
        feature_lists = [[] for col in feature_cols]
        if label_cols is not None:
            label_lists = [[] for col in label_cols]
        else:
            label_lists = None
        return feature_lists, label_lists

    def add_row(data, results):
        if not isinstance(data, list):
            arrays = [data]
        else:
            arrays = data

        for i, arr in enumerate(arrays):
            results[i].append(arr)

    def merge_rows(results):
        result_arrs = [np.stack(l) for l in results]
        if len(result_arrs) == 1:
            result_arrs = result_arrs[0]
        else:
            result_arrs = tuple(result_arrs)
        return result_arrs

    def generate_output(feature_lists, label_lists):
        feature_arrs = merge_rows(feature_lists)
        if label_cols is not None:
            label_arrs = merge_rows(label_lists)
            return {"x": feature_arrs, "y": label_arrs}
        else:
            return {"x": feature_arrs}

    feature_lists, label_lists = init_result_lists()
    counter = 0

    for row in iter:
        counter += 1
        add_row(row[0], feature_lists)
        if label_cols is not None:
            add_row(row[1], label_lists)

        if shard_size and counter % shard_size == 0:
            yield generate_output(feature_lists, label_lists)
            feature_lists, label_lists = init_result_lists()

    if feature_lists[0]:
        yield generate_output(feature_lists, label_lists)


def transform_to_shard_dict(data, feature_cols, label_cols=None):
    def to_shard_dict(df):
        result = dict()
        result["x"] = [df[feature_col].to_numpy() for feature_col in feature_cols]
        if label_cols:
            result["y"] = df[label_cols[0]].to_numpy()
        return result
    data = data.transform_shard(to_shard_dict)
    return data


def process_xshards_of_pandas_dataframe(data, feature_cols, label_cols=None, validation_data=None,
                                        mode=None):
    data = transform_to_shard_dict(data, feature_cols, label_cols)
    if mode == "fit":
        if validation_data:
            assert validation_data._get_class_name() == 'pandas.core.frame.DataFrame',\
                "train data and validation data should be both XShards of Pandas DataFrame"
            validation_data = transform_to_shard_dict(validation_data, feature_cols, label_cols)
        return data, validation_data
    else:
        return data


def _dataframe_to_xshards(data, feature_cols, label_cols=None, accept_str_col=False):
    from zoo.orca import OrcaContext
    schema = data.schema
    shard_size = OrcaContext._shard_size
    numpy_rdd = data.rdd.map(lambda row: convert_row_to_numpy(row,
                                                              schema,
                                                              feature_cols,
                                                              label_cols,
                                                              accept_str_col))
    shard_rdd = numpy_rdd.mapPartitions(lambda x: arrays2dict(x,
                                                              feature_cols,
                                                              label_cols,
                                                              shard_size))
    return SparkXShards(shard_rdd)


def dataframe_to_xshards(data, validation_data, feature_cols, label_cols, mode="fit",
                         num_workers=None, accept_str_col=False):
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
        # avoid empty partition for worker
        if data.rdd.getNumPartitions() < num_workers:
            data = data.repartition(num_workers)
        if validation_data is not None:
            num_data_part = data.rdd.getNumPartitions()
            validation_data = validation_data.repartition(num_data_part)

    data = _dataframe_to_xshards(data, feature_cols, label_cols, accept_str_col)
    if validation_data is not None:
        validation_data = _dataframe_to_xshards(validation_data, feature_cols, label_cols,
                                                accept_str_col)

    return data, validation_data


def maybe_dataframe_to_xshards(data, validation_data, feature_cols, label_cols, mode="fit",
                               num_workers=None, accept_str_col=False):
    from pyspark.sql import DataFrame
    if isinstance(data, DataFrame):
        data, validation_data = dataframe_to_xshards(data, validation_data,
                                                     feature_cols=feature_cols,
                                                     label_cols=label_cols,
                                                     mode=mode,
                                                     num_workers=num_workers,
                                                     accept_str_col=accept_str_col)
    return data, validation_data


def bigdl_metric_results_to_dict(results):
    return dict([(r.method, r.result) for r in results])
