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

import random
import ray
from functools import reduce
from pyspark.context import SparkContext

from bigdl.util.common import get_node_and_core_number

from zoo.ray import RayContext
from zoo.orca.data.shard import RayDataShards, RayPartition, SparkDataShards
from zoo.orca.data.utils import *


def read_csv(file_path, context, **kwargs):
    """
    Read csv files to DataShards
    :param file_path: could be a csv file, multiple csv file paths separated by comma,
     a directory containing csv files.
     Supported file systems are local file system, hdfs, and s3.
    :param context: SparkContext or RayContext
    :return: DataShards
    """
    if isinstance(context, RayContext):
        return read_file_ray(context, file_path, "csv", **kwargs)
    elif isinstance(context, SparkContext):
        return read_file_spark(context, file_path, "csv", **kwargs)
    else:
        raise Exception("Context type should be RayContext or SparkContext")


def read_json(file_path, context, **kwargs):
    """
    Read json files to DataShards
    :param file_path: could be a json file, multiple json file paths separated by comma,
     a directory containing json files.
     Supported file systems are local file system, hdfs, and s3.
    :param context: SparkContext or RayContext
    :return: DataShards
    """
    if isinstance(context, RayContext):
        return read_file_ray(context, file_path, "json", **kwargs)
    elif isinstance(context, SparkContext):
        return read_file_spark(context, file_path, "json", **kwargs)
    else:
        raise Exception("Context type should be RayContext or SparkContext")


def read_file_ray(context, file_path, file_type, **kwargs):
    file_paths = []
    # extract all file paths
    if isinstance(file_path, list):
        [file_paths.extend(extract_one_path(path, file_type, context.env)) for path in file_path]
    else:
        file_paths = extract_one_path(file_path, file_type, context.env)

    num_executors = context.num_ray_nodes
    num_cores = context.ray_node_cpu_cores
    num_partitions = num_executors * num_cores

    # split files to partitions
    random.shuffle(file_paths)
    # remove empty partitions
    file_partition_list = [partition for partition
                           in list(chunk(file_paths, num_partitions)) if partition]
    # create shard actor to read data
    shards = [RayPandasShard.remote() for i in range(len(file_partition_list))]
    done_ids, undone_ids = \
        ray.wait([shard.read_file_partitions.remote(file_partition_list[i], file_type, **kwargs)
                  for i, shard in enumerate(shards)], num_returns=len(shards))
    assert len(undone_ids) == 0

    # create initial partition
    partitions = [RayPartition([shard]) for shard in shards]
    data_shards = RayDataShards(partitions)
    return data_shards


def read_file_spark(context, file_path, file_type, **kwargs):
    file_url_splits = file_path.split("://")
    prefix = file_url_splits[0]
    node_num, core_num = get_node_and_core_number()

    file_paths = []
    if isinstance(file_path, list):
        [file_paths.extend(extract_one_path(path, file_type, os.environ)) for path in file_path]
    else:
        file_paths = extract_one_path(file_path, file_type, os.environ)

    rdd = context.parallelize(file_paths, node_num * core_num)

    if prefix == "hdfs":
        def loadFile(iterator):
            import pandas as pd
            import pyarrow as pa
            fs = pa.hdfs.connect()

            for x in iterator:
                with fs.open(x, 'rb') as f:
                    if file_type == "csv":
                        df = pd.read_csv(f, **kwargs)
                    elif file_type == "json":
                        df = pd.read_json(f, **kwargs)
                    else:
                        raise Exception("Unsupported file type")
                    yield df

        pd_rdd = rdd.mapPartitions(loadFile)
    elif prefix == "s3":
        def loadFile(iterator):
            access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
            secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            import boto3
            import pandas as pd
            s3_client = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            ).client('s3', verify=False)
            for x in iterator:
                path_parts = x.split("://")[1].split('/')
                bucket = path_parts.pop(0)
                key = "/".join(path_parts)
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                if file_type == "json":
                    df = pd.read_json(obj['Body'], **kwargs)
                elif file_type == "csv":
                    df = pd.read_csv(obj['Body'], **kwargs)
                else:
                    raise Exception("Unsupported file type")
                yield df

        pd_rdd = rdd.mapPartitions(loadFile)
    else:
        def loadFile(iterator):
            import pandas as pd
            for x in iterator:
                if file_type == "csv":
                    df = pd.read_csv(x, **kwargs)
                elif file_type == "json":
                    df = pd.read_json(x, **kwargs)
                else:
                    raise Exception("Unsupported file type")
                yield df

        pd_rdd = rdd.mapPartitions(loadFile)

    data_shards = SparkDataShards(pd_rdd)
    return data_shards


@ray.remote
class RayPandasShard(object):
    """
    Actor to read csv/json file to Pandas DataFrame and manipulate data
    """
    def __init__(self, data=None):
        self.data = data

    def read_file_partitions(self, paths, file_type, **kwargs):
        df_list = []
        import pandas as pd
        prefix = paths[0].split("://")[0]
        if prefix == "hdfs":
            import pyarrow as pa
            fs = pa.hdfs.connect()
            for path in paths:
                with fs.open(path, 'rb') as f:
                    if file_type == "json":
                        df = pd.read_json(f, **kwargs)
                    elif file_type == "csv":
                        df = pd.read_csv(f, **kwargs)
                    else:
                        raise Exception("Unsupported file type")
                    df_list.append(df)
        elif prefix == "s3":
            import boto3
            access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
            secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            s3_client = boto3.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
            ).client('s3', verify=False)
            for path in paths:
                path_parts = path.split("://")[1].split('/')
                bucket = path_parts.pop(0)
                key = "/".join(path_parts)
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                if file_type == "json":
                    df = pd.read_json(obj['Body'], **kwargs)
                elif file_type == "csv":
                    df = pd.read_csv(obj['Body'], **kwargs)
                else:
                    raise Exception("Unsupported file type")
                df_list.append(df)
        else:
            for path in paths:
                if file_type == "json":
                    df = pd.read_json(path, **kwargs)
                elif file_type == "csv":
                    df = pd.read_csv(path, **kwargs)
                else:
                    raise Exception("Unsupported file type")
                df_list.append(df)
        self.data = pd.concat(df_list)
        return 0

    def transform(self, func, *args):
        self.data = func(self.data, *args)
        return 0

    def get_data(self):
        return self.data
