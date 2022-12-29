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
from PIL import Image, ImageDraw
import numpy as np
from bigdl.orca import OrcaContext
from bigdl.dllib.utils.common import (get_node_and_core_number,
                                      get_spark_sql_context,
                                      get_spark_context)
from bigdl.dllib.utils.log4Error import invalidInputError
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf

from typing import (Union, List, Dict)

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy import ndarray
    from pyspark.rdd import PipelinedRDD, RDD

from bigdl.orca.data.file import *
from bigdl.orca.data.utils import *
from bigdl.orca.data import SparkXShards


def read_images_pil(file_path):
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
    # print(file_paths)

    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores
    rdd = sc.parallelize(file_paths, num_partitions)

    def load_image(iterator):
        for f in iterator:
            image = open_image(f)
            yield {'origin': f, 'pilimage': image}

    im_rdd = rdd.mapPartitions(load_image)

    return SparkXShards(im_rdd)


def read_images_spark(file_path):
    spark = OrcaContext.get_spark_session()
    image_df = spark.read.format("image").load(file_path)

    def convert_bgr_array_to_rgb_array(img_array):
        B, G, R = img_array.T
        return np.array((R, G, B)).T

    def to_pil(image_spark):
        mode = 'RGBA' if (image_spark.image.nChannels == 4) else 'RGB'
        img = Image.frombytes(mode=mode, data=bytes(image_spark.image.data),
                          size=[image_spark.image.width, image_spark.image.height])

        converted_img_array = convert_bgr_array_to_rgb_array(np.asarray(img))
        image = Image.fromarray(converted_img_array)
        return {"origin": image_spark.image.origin, "pilimage": image}

    image_rdd = image_df.rdd.map(lambda x: to_pil(x))

    return SparkXShards(image_rdd)

