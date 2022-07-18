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
import numpy as np

from bigdl.dllib.utils.file_utils import get_file_list
from bigdl.dllib.utils.utils import convert_row_to_numpy
from bigdl.dllib.utils.log4Error import *


def list_s3_file(file_path, env):
    path_parts = file_path.split('/')
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    access_key_id = env["AWS_ACCESS_KEY_ID"]
    secret_access_key = env["AWS_SECRET_ACCESS_KEY"]

    import boto3
    s3_client = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).client('s3', verify=False)
    # file
    if os.path.splitext(file_path)[1] != '':
        return ["s3://" + file_path]
    else:
        keys = []
        resp = s3_client.list_objects_v2(Bucket=bucket,
                                         Prefix=key)
        for obj in resp['Contents']:
            keys.append(obj['Key'])
        file_paths = [os.path.join("s3://" + bucket, file) for file in keys]
        return file_paths


def extract_one_path(file_path, env):
    file_url_splits = file_path.split("://")
    prefix = file_url_splits[0]
    if prefix == "s3":
        file_paths = list_s3_file(file_url_splits[1], env)
    elif prefix == "hdfs":
        import pyarrow as pa
        fs = pa.hdfs.connect()
        if fs.isfile(file_path):
            file_paths = [file_path]
        else:
            file_paths = get_file_list(file_path)
    else:  # Local file path; could be a relative path.
        from os.path import isfile, abspath, join
        if isfile(file_path):
            file_paths = [abspath(file_path)]
        else:
            # An error would be already raised here if the path is invalid.
            file_paths = [abspath(join(file_path, file)) for file in os.listdir(file_path)]
    return file_paths


def check_type_and_convert(data, allow_tuple=True, allow_list=True):
    """
    :param allow_tuple: boolean, if the model accepts a tuple as input. Default: True
    :param allow_list: boolean, if the model accepts a list as input. Default: True
    :return:
    """
    def check_and_convert(convert_data):
        if isinstance(convert_data, np.ndarray):
            return [convert_data]
        elif isinstance(convert_data, tuple) and \
                all([isinstance(d, np.ndarray) for d in convert_data]):
            return _convert_list_tuple(convert_data, allow_tuple=allow_tuple,
                                       allow_list=allow_list)
        elif isinstance(convert_data, list) and \
                all([isinstance(d, np.ndarray) for d in convert_data]):
            return _convert_list_tuple(convert_data, allow_tuple=allow_tuple,
                                       allow_list=allow_list)
        else:
            invalidInputError(False,
                              "value of x and y should be a ndarray, "
                              "a tuple of ndarrays or a list of ndarrays")

    result = {}
    invalidInputError(isinstance(data, dict), "each shard should be an dict")
    invalidInputError("x" in data, "key x should in each shard")
    x = data["x"]
    result["x"] = check_and_convert(x)
    if "y" in data:
        y = data["y"]
        result["y"] = check_and_convert(y)
    return result


def get_spec(allow_tuple=True, allow_list=True):
    """
    :param allow_tuple: boolean, if the model accepts a tuple as input. Default: True
    :param allow_list: boolean, if the model accepts a list as input. Default: True
    :return:
    """
    def _get_spec(data):
        data = check_type_and_convert(data, allow_tuple, allow_list)
        feature_spec = [(feat.dtype, feat.shape[1:])
                        for feat in data["x"]]
        if "y" in data:
            label_spec = [(label.dtype, label.shape[1:])
                          for label in data["y"]]
        else:
            label_spec = None
        return feature_spec, label_spec
    return _get_spec


# todo this might be very slow
def flatten_xy(allow_tuple=True, allow_list=True):
    """
    :param allow_tuple: boolean, if the model accepts a tuple as input. Default: True
    :param allow_list: boolean, if the model accepts a list as input. Default: True
    :return:
    """
    def _flatten_xy(data):
        data = check_type_and_convert(data, allow_tuple, allow_list)
        features = data["x"]

        has_label = "y" in data
        labels = data["y"] if has_label else None
        length = features[0].shape[0]

        for i in range(length):
            fs = [feat[i] for feat in features]
            if has_label:
                ls = [l[i] for l in labels]
                yield (fs, ls)
            else:
                yield (fs,)
    return _flatten_xy


def combine(data_list):
    item = data_list[0]
    if isinstance(item, dict):
        res = {}
        for k, v in item.items():
            res[k] = np.concatenate([data[k] for data in data_list], axis=0)
    elif isinstance(item, list) or isinstance(item, tuple):
        res = []
        for i in range(len(data_list[0])):
            res.append(np.concatenate([data[i] for data in data_list], axis=0))
        if isinstance(item, tuple):
            res = tuple(res)
    elif isinstance(data_list[0], np.ndarray):
        res = np.concatenate(data_list, axis=0)
    else:
        invalidInputError(False,
                          "value of x and y should be an ndarray, a dict of ndarrays, a tuple"
                          " of ndarrays or a list of ndarrays, please check your input")
    return res


def ray_partition_get_data_label(partition_data,
                                 allow_tuple=True,
                                 allow_list=True,
                                 has_label=True):
    """
    :param partition_data: The data partition from Spark RDD, which should be a list of records.
    :param allow_tuple: Boolean. Whether the model accepts a tuple as input. Default is True.
    :param allow_list: Boolean. Whether the model accepts a list as input. Default is True.
    :param has_label: Boolean. Whether the data partition contains labels.
    :return: Concatenated data for the data partition.
    """
    data_list = [data['x'] for data in partition_data]
    label_list = [data['y'] for data in partition_data]

    data = _convert_list_tuple(combine(data_list),
                               allow_tuple=allow_tuple, allow_list=allow_list)
    if has_label:
        label = _convert_list_tuple(combine(label_list),
                                    allow_tuple=allow_tuple, allow_list=allow_list)
    else:
        label = None

    return data, label


def ray_partitions_get_data_label(partition_list,
                                  allow_tuple=True,
                                  allow_list=True,
                                  has_label=True):
    partition_data = [item for partition in partition_list for item in partition]
    data, label = ray_partition_get_data_label(partition_data,
                                               allow_tuple=allow_tuple,
                                               allow_list=allow_list,
                                               has_label=has_label)
    return data, label


def ray_partitions_get_tf_dataset(partition_list, has_label=True):
    import tensorflow as tf
    partition_data = [item for partition in partition_list for item in partition]
    if len(partition_data) != 0:

        sample = partition_data[0]
        keys = sample.keys()
        if "x" in keys:
            if has_label:
                invalidInputError("y" in keys, "key y should in each shard if has_label=True")
            data, label = ray_partition_get_data_label(partition_data,
                                                       allow_tuple=True,
                                                       allow_list=False)
            dataset = tf.data.Dataset.from_tensor_slices((data, label))
        elif "ds_def" in keys and "elem_spec" in keys:
            from tensorflow.python.distribute.coordinator.values import \
                deserialize_dataset_from_graph
            from functools import reduce
            dataset_list = [deserialize_dataset_from_graph(serialized_dataset["ds_def"],
                                                           serialized_dataset["elem_spec"])
                            for serialized_dataset in partition_data]
            dataset = reduce(lambda x, y: x.concatenate(y), dataset_list)
        else:
            invalidInputError(False,
                              "value of x and y should be a ndarray, "
                              "a tuple of ndarrays or a list of ndarrays")
    else:
        # TODO: may cause error
        dataset = tf.data.Dataset.from_tensor_slices(([], []))

    return dataset


# todo: this might be very slow
def xshard_to_sample(data):
    from bigdl.dllib.utils.file_utils import Sample
    data = check_type_and_convert(data, allow_list=True, allow_tuple=False)
    features = data["x"]
    length = features[0].shape[0]
    if "y" in data:
        labels = data["y"]
    else:
        labels = np.array([[-1] * length])

    for i in range(length):
        fs = [feat[i] for feat in features]
        ls = [l[i] for l in labels]
        if len(fs) == 1:
            fs = fs[0]
        if len(ls) == 1:
            ls = ls[0]
        yield Sample.from_ndarray(fs, ls)


def row_to_sample(row, schema, feature_cols, label_cols):
    from bigdl.dllib.utils.common import Sample
    if label_cols:
        feature, label = convert_row_to_numpy(row, schema, feature_cols, label_cols)
        sample = Sample.from_ndarray(feature, label)
    else:
        feature, = convert_row_to_numpy(row, schema, feature_cols, label_cols)
        sample = Sample.from_ndarray(feature, np.array([0.0]))
    return sample


def read_pd_hdfs_file_list(iterator, file_type, **kwargs):
    import pyarrow as pa
    fs = pa.hdfs.connect()
    dfs = []
    for x in iterator:
        with fs.open(x, 'rb') as f:
            df = read_pd_file(f, file_type, **kwargs)
            dfs.append(df)
    import pandas as pd
    return [pd.concat(dfs)]


def read_pd_s3_file_list(iterator, file_type, **kwargs):
    access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    import boto3
    s3_client = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).client('s3', verify=False)
    dfs = []
    for x in iterator:
        path_parts = x.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = read_pd_file(obj['Body'], file_type, **kwargs)
        dfs.append(df)
    import pandas as pd
    return [pd.concat(dfs)]


def read_pd_file(path, file_type, **kwargs):
    import pandas as pd
    if file_type == "csv":
        df = pd.read_csv(path, **kwargs)
    elif file_type == "json":
        df = pd.read_json(path, **kwargs)
    else:
        invalidInputError(False,
                          "Unsupported file type: %s. Only csv and json files are "
                          "supported for now" % file_type)
    return df


def get_class_name(obj):
    if obj.__class__.__module__ != 'builtins':
        return '.'.join([obj.__class__.__module__, obj.__class__.__name__])
    return obj.__class__.__name__


def _convert_list_tuple(data, allow_tuple, allow_list):
    if isinstance(data, list):
        if not allow_list and allow_tuple:
            return tuple(data)
    else:
        if not allow_tuple and allow_list:
            return list(data)
    return data


def process_spark_xshards(spark_xshards, num_workers):
    from bigdl.orca.data.ray_xshards import RayXShards
    data = spark_xshards
    ray_xshards = RayXShards.from_spark_xshards(data)
    return ray_xshards


def index_data(x, i):
    if isinstance(x, np.ndarray):
        return x[i]
    elif isinstance(x, dict):
        res = {}
        for k, v in x.items():
            res[k] = v[i]
        return res
    elif isinstance(x, tuple):
        return tuple(item[i] for item in x)
    elif isinstance(x, list):
        return [item[i] for item in x]
    else:
        invalidInputError(False,
                          "data should be an ndarray, a dict of ndarrays, a tuple of ndarrays"
                          " or a list of ndarrays, please check your input")


def get_size(x):
    if isinstance(x, np.ndarray):
        return len(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            return len(v)
    elif isinstance(x, tuple) or isinstance(x, list):
        return len(x[0])
    else:
        invalidInputError(False,
                          "data should be an ndarray, a dict of ndarrays, a tuple of ndarrays"
                          " or a list of ndarrays, please check your input")


def spark_df_to_rdd_pd(df, squeeze=False, index_col=None,
                       dtype=None, index_map=None):
    from bigdl.orca.data import SparkXShards
    from bigdl.orca import OrcaContext
    columns = df.columns

    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    to_array = F.udf(lambda v: v.toArray().tolist(), T.ArrayType(T.FloatType()))
    for colName, colType in df.dtypes:
        if colType == 'vector':
            df = df.withColumn(colName, to_array(colName))

    shard_size = OrcaContext._shard_size
    pd_rdd = df.rdd.mapPartitions(to_pandas(df.columns, squeeze, index_col, dtype, index_map,
                                            batch_size=shard_size))
    return pd_rdd


def spark_df_to_pd_sparkxshards(df, squeeze=False, index_col=None,
                                dtype=None, index_map=None):
    pd_rdd = spark_df_to_rdd_pd(df, squeeze, index_col, dtype, index_map)
    from bigdl.orca.data import SparkXShards
    spark_xshards = SparkXShards(pd_rdd)
    return spark_xshards


def to_pandas(columns, squeeze=False, index_col=None, dtype=None, index_map=None,
              batch_size=None):

    def postprocess(pd_df):
        if dtype is not None:
            if isinstance(dtype, dict):
                for col, type in dtype.items():
                    if isinstance(col, str):
                        if col not in pd_df.columns:
                            invalidInputError(False,
                                              "column to be set type is not"
                                              " in current dataframe")
                        pd_df[col] = pd_df[col].astype(type)
                    elif isinstance(col, int):
                        if index_map[col] not in pd_df.columns:
                            invalidInputError(False,
                                              "column index to be set type is not"
                                              " in current dataframe")
                        pd_df[index_map[col]] = pd_df[index_map[col]].astype(type)
            else:
                pd_df = pd_df.astype(dtype)
        if squeeze and len(pd_df.columns) == 1:
            pd_df = pd_df.iloc[:, 0]
        if index_col:
            pd_df = pd_df.set_index(index_col)
        return pd_df

    def f(iter):
        import pandas as pd
        counter = 0
        data = []
        for row in iter:
            counter += 1
            data.append(row)
            if batch_size and counter % batch_size == 0:
                pd_df = pd.DataFrame(data, columns=columns)
                pd_df = postprocess(pd_df)
                yield pd_df
                data = []
        if data:
            pd_df = pd.DataFrame(data, columns=columns)
            pd_df = postprocess(pd_df)
            yield pd_df

    return f


def spark_xshards_to_ray_dataset(spark_xshards):
    from bigdl.orca.data.ray_xshards import RayXShards
    import ray

    ray_xshards = RayXShards.from_spark_xshards(spark_xshards)
    partition_refs = ray_xshards.get_refs()

    ray_dataset = ray.data.from_pandas_refs(partition_refs)
    return ray_dataset


def generate_string_idx(df, columns, freq_limit, order_by_freq):
    from bigdl.dllib.utils.file_utils import callZooFunc
    return callZooFunc("float", "generateStringIdx", df, columns, freq_limit, order_by_freq)


def check_col_exists(df, columns):
    df_cols = df.columns
    col_not_exist = list(filter(lambda x: x not in df_cols, columns))
    if len(col_not_exist) > 0:
        invalidInputError(False,
                          str(col_not_exist) + " do not exist in this Table")
