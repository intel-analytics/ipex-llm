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
import os
import numpy as np

from zoo.common import get_file_list


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
            raise ValueError("value of x and y should be a ndarray, "
                             "a tuple of ndarrays or a list of ndarrays")

    result = {}
    assert isinstance(data, dict), "each shard should be an dict"
    assert "x" in data, "key x should in each shard"
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


def ray_partition_get_data_label(partition_data, allow_tuple=True, allow_list=True):
    """
    :param partition_data:
    :param allow_tuple: boolean, if the model accepts a tuple as input. Default: True
    :param allow_list: boolean, if the model accepts a list as input. Default: True
    :return:
    """
    from functools import reduce

    def combine_dict(dict1, dict2):
        return {key: np.concatenate((value, dict2[key]), axis=0)
                for (key, value) in dict1.items()}

    def combine_list(list1, list2):
        return [np.concatenate((list1[index], list2[index]), axis=0)
                for index in range(0, len(list1))]

    def combine_tuple(tuple1, tuple2):
        return tuple(np.concatenate((tuple1[index], tuple2[index]), axis=0)
                     for index in range(0, len(tuple1)))

    def check_type_and_combine(data_list):
        if isinstance(data_list[0], dict):
            return reduce(lambda dict1, dict2: combine_dict(dict1, dict2), data_list)
        elif isinstance(data_list[0], np.ndarray):
            return reduce(lambda array1, array2: np.concatenate((array1, array2), axis=0),
                          data_list)
        elif isinstance(data_list[0], list):
            data = reduce(lambda list1, list2: combine_list(list1, list2), data_list)
            data = _convert_list_tuple(data, allow_tuple=allow_tuple, allow_list=allow_list)
            return data
        elif isinstance(data_list[0], tuple):
            data = reduce(lambda tuple1, tuple2: combine_tuple(tuple1, tuple2), data_list)
            data = _convert_list_tuple(data, allow_tuple=allow_tuple, allow_list=allow_list)
            return data
        else:
            raise ValueError(
                "value of x and y should be a ndarray, a dict of ndarrays, a tuple of ndarrays"
                " or a list of ndarrays, please check")

    data_list = [data['x'] for data in partition_data]
    label_list = [data['y'] for data in partition_data]

    data = check_type_and_combine(data_list)
    label = check_type_and_combine(label_list)

    return data, label


# todo: this might be very slow
def to_sample(data):
    from bigdl.util.common import Sample
    data = check_type_and_convert(data, allow_list=True, allow_tuple=False)
    features = data["x"]
    labels = data["y"]
    length = features[0].shape[0]

    for i in range(length):
        fs = [feat[i] for feat in features]
        ls = [l[i] for l in labels]
        if len(fs) == 1:
            fs = fs[0]
        if len(ls) == 1:
            ls = ls[0]
        yield Sample.from_ndarray(np.array(fs), np.array(ls))


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
        raise Exception("Unsupported file type: %s. Only csv and json files are "
                        "supported for now" % file_type)
    return df


def get_class_name(obj):
    if obj.__class__.__module__ != 'builtins':
        return '.'.join([obj.__class__.__module__, obj.__class__.__name__])
    return obj.__class__.__name__


def _convert_list_tuple(data, allow_tuple, allow_list):
    if not allow_list and not allow_tuple:
        raise ValueError("value of x and y should be a ndarray, but get a " +
                         data.__class__.__name__ + " instead")
    if isinstance(data, list):
        if not allow_list and allow_tuple:
            return tuple(data)
    else:
        if not allow_tuple and allow_list:
            return list(data)
    return data
