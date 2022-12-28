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

from bigdl.orca import OrcaContext
from bigdl.dllib.utils.common import (get_node_and_core_number,
                                      get_spark_sql_context,
                                      get_spark_context)
from bigdl.dllib.utils.log4Error import invalidInputError


from typing import (Union, List, Dict)

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy import ndarray
    from pyspark.rdd import PipelinedRDD, RDD

from bigdl.orca.data.file import *
from bigdl.orca.data.utils import *
from bigdl.orca.data import SparkXShards


def read_images(file_path):
    sc = OrcaContext.get_spark_context()
    node_num, core_num = get_node_and_core_number()

    file_url_splits = file_path.split("://")
    prefix = file_url_splits[0]

    file_paths = []
    if isinstance(file_path, list):
        [file_paths.extend(extract_one_path(path, os.environ)) for path in file_path]
    else:
        file_paths = extract_one_path(file_path, os.environ)

    if not file_paths:
        invalidInputError(False,
                          "The file path is invalid or empty, please check your data")

    num_files = len(file_paths)
    print(file_paths)
    if prefix == "hdfs" or prefix == "s3":
        file_paths = [prefix + "://" + f for f in file_paths]
    print(file_paths)

    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores
    rdd = sc.parallelize(file_paths, num_partitions)

    def load_image(iterator):
        for f in iterator:
            print(f)
            image = open_image(f)
            yield {'filename': f, 'x': image}

    im_rdd = rdd.mapPartitions(load_image)

    return SparkXShards(im_rdd)


