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

from bigdl.util.common import get_node_and_core_number
from zoo import init_nncontext, ZooContext
from zoo.orca.data import SparkXShards
from zoo.orca.data.utils import *


def read_csv(file_path, **kwargs):
    """
    Read csv files to SparkXShards of pandas DataFrames.

    :param file_path: A csv file path, a list of multiple csv file paths, or a directory
    containing csv files. Local file system, HDFS, and AWS S3 are supported.
    :param kwargs: You can specify read_csv options supported by pandas.
    :return: An instance of SparkXShards.
    """
    return read_file_spark(file_path, "csv", **kwargs)


def read_json(file_path, **kwargs):
    """
    Read json files to SparkXShards of pandas DataFrames.

    :param file_path: A json file path, a list of multiple json file paths, or a directory
    containing json files. Local file system, HDFS, and AWS S3 are supported.
    :param kwargs: You can specify read_json options supported by pandas.
    :return: An instance of SparkXShards.
    """
    return read_file_spark(file_path, "json", **kwargs)


def read_file_spark(file_path, file_type, **kwargs):
    sc = init_nncontext()
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

    if ZooContext.orca_pandas_read_backend == "pandas":
        num_files = len(file_paths)
        total_cores = node_num * core_num
        num_partitions = num_files if num_files < total_cores else total_cores
        rdd = sc.parallelize(file_paths, num_partitions)

        if prefix == "hdfs":
            pd_rdd = rdd.mapPartitions(
                lambda iter: read_pd_hdfs_file_list(iter, file_type, **kwargs))
        elif prefix == "s3":
            pd_rdd = rdd.mapPartitions(
                lambda iter: read_pd_s3_file_list(iter, file_type, **kwargs))
        else:
            def loadFile(iterator):
                for x in iterator:
                    df = read_pd_file(x, file_type, **kwargs)
                    yield df

            pd_rdd = rdd.mapPartitions(loadFile)
    else:
        from pyspark.sql import SQLContext
        sqlContext = SQLContext.getOrCreate(sc)
        spark = sqlContext.sparkSession
        # TODO: add S3 confidentials
        if file_type == "json":
            df = spark.read.json(file_paths, **kwargs)
        elif file_type == "csv":
            df = spark.read.csv(file_paths, **kwargs)
        else:
            raise Exception("Unsupported file type")
        if df.rdd.getNumPartitions() < node_num:
            df = df.repartition(node_num)

        def to_pandas(columns):
            def f(iter):
                import pandas as pd
                data = list(iter)
                yield pd.DataFrame(data, columns=columns)

            return f

        pd_rdd = df.rdd.mapPartitions(to_pandas(df.columns))

    data_shards = SparkXShards(pd_rdd)
    return data_shards
