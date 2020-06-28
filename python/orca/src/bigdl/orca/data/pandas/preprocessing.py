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

from bigdl.util.common import get_node_and_core_number
from pyspark.context import SparkContext

from zoo.ray import RayContext
from zoo.orca.data.shard import RayXShards, RayPartition, SparkXShards
from zoo.orca.data.utils import *


def read_csv(file_path, context, **kwargs):
    """
    Read csv files to XShards
    :param file_path: could be a csv file, multiple csv file paths separated by comma,
     a directory containing csv files.
     Supported file systems are local file system, hdfs, and s3.
    :param context: SparkContext or RayContext
    :return: XShards
    """
    if isinstance(context, RayContext):
        return read_file_ray(context, file_path, "csv", **kwargs)
    elif isinstance(context, SparkContext):
        return read_file_spark(context, file_path, "csv", **kwargs)
    else:
        raise Exception("Context type should be RayContext or SparkContext")


def read_json(file_path, context, **kwargs):
    """
    Read json files to XShards
    :param file_path: could be a json file, multiple json file paths separated by comma,
     a directory containing json files.
     Supported file systems are local file system, hdfs, and s3.
    :param context: SparkContext or RayContext
    :return: XShards
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

    if not file_paths:
        raise Exception("The file path is invalid/empty or does not include csv/json files")

    num_executors = context.num_ray_nodes
    num_cores = context.ray_node_cpu_cores
    num_partitions = num_executors * num_cores

    # split files to partitions
    random.shuffle(file_paths)
    # remove empty partitions
    file_partition_list = [partition for partition
                           in list(chunk(file_paths, num_partitions)) if partition]
    import ray
    # create shard actor to read data
    Shard = ray.remote(RayPandasShard)
    shards = [Shard.remote() for i in range(len(file_partition_list))]
    done_ids, undone_ids = \
        ray.wait([shard.read_file_partitions.remote(file_partition_list[i], file_type, **kwargs)
                  for i, shard in enumerate(shards)], num_returns=len(shards))
    assert len(undone_ids) == 0

    # create initial partition
    partitions = [RayPartition([shard]) for shard in shards]
    data_shards = RayXShards(partitions)
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

    if not file_paths:
        raise Exception("The file path is invalid/empty or does not include csv/json files")

    num_files = len(file_paths)
    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores
    rdd = context.parallelize(file_paths, num_partitions)

    if prefix == "hdfs":
        pd_rdd = rdd.mapPartitions(lambda iter: read_pd_hdfs_file_list(iter, file_type, **kwargs))
    elif prefix == "s3":
        pd_rdd = rdd.mapPartitions(lambda iter: read_pd_s3_file_list(iter, file_type, **kwargs))
    else:
        def loadFile(iterator):
            for x in iterator:
                df = read_pd_file(x, file_type, **kwargs)
                yield df

        pd_rdd = rdd.mapPartitions(loadFile)

    data_shards = SparkXShards(pd_rdd)
    return data_shards


class RayPandasShard(object):
    """
    Actor to manipulate data
    """

    def __init__(self, data=None):
        self.data = data

    def read_file_partitions(self, paths, file_type, **kwargs):
        import pandas as pd
        prefix = paths[0].split("://")[0]
        if prefix == "hdfs":
            df_list = read_pd_hdfs_file_list(paths, file_type, **kwargs)
        elif prefix == "s3":
            df_list = read_pd_s3_file_list(paths, file_type, **kwargs)
        else:
            df_list = [read_pd_file(path, file_type, **kwargs) for path in paths]
        self.data = pd.concat(df_list)
        return 0

    def transform(self, func, *args):
        self.data = func(self.data, *args)
        return 0

    def get_data(self):
        return self.data
