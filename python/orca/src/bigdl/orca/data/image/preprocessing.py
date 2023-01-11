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
from bigdl.orca import OrcaContext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy import ndarray

from bigdl.orca.data.file import *
from bigdl.orca.data.utils import *
from bigdl.orca.data import SparkXShards
from pyspark.sql.functions import col, udf

def get_file_paths(file_path):
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
    return file_paths


def read_images_pil(file_path):
    sc = OrcaContext.get_spark_context()

    file_paths = get_file_paths(file_path)
    num_files = len(file_paths)
    node_num, core_num = get_node_and_core_number()
    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores

    rdd = sc.parallelize(file_paths, num_partitions)

    def load_image(iterator):
        for f in iterator:
            image = open_image(f)
            yield {'pilimage':image, 'origin': f}

    im_rdd = rdd.mapPartitions(load_image)

    return SparkXShards(im_rdd)


def read_images_spark(file_path,
                      label_func=None,
                      target_path=None,
                      image_type=".jpg",
                      target_type=".png"):
    spark = OrcaContext.get_spark_session()

    img_paths = get_file_paths(file_path)
    img_paths = list(filter(
        lambda fn: fn.endswith(image_type) and (not fn.startswith(".")), img_paths))
    num_files = len(img_paths)
    node_num, core_num = get_node_and_core_number()
    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores

    image_df = spark.read.format("image").load(img_paths)
    print("image modes", image_df.select("image.mode").distinct().collect())
    print("image nChannels", image_df.select("image.nChannels").distinct().collect())

    def convert_bgr_array_to_rgb_array(img_array):
        B, G, R, = img_array.T
        return np.array((R, G, B)).T

    def to_pil(image_spark):
        mode = 'L'
        if image_spark.image.nChannels == 4:
            mode = 'RGBA'
        elif image_spark.image.nChannels == 3:
            mode = 'RGB'
        img = Image.frombytes(mode=mode, data=bytes(image_spark.image.data),
                              size=[image_spark.image.width, image_spark.image.height])
        if mode in ['RGB', 'RGBA']:
            img = img.convert("RGB") if mode == 'RGBA' else img
            converted_img_array = convert_bgr_array_to_rgb_array(np.asarray(img)[:, :, :])
            img = Image.fromarray(converted_img_array)

        return img, label_func(image_spark.image.origin) if label_func else img

    image_rdd = image_df.orderBy(col("image.origin")).rdd\
        .map(lambda x: to_pil(x)).repartition(num_partitions)

    if target_path:
        target_paths = get_file_paths(target_path)
        target_paths = list(filter(
            lambda fn: fn.endswith(target_type) and (not fn.startswith(".")), target_paths))

        assert(len(target_paths) == len(img_paths),
               "number of target files and image files should be the same")
        target_df = spark.read.format("image").load(target_paths)

        target_rdd = target_df.orderBy(col("image.origin")).rdd\
            .map(lambda x: to_pil(x)).repartition(num_partitions)

        image_rdd = image_rdd.zip(target_rdd)
    return SparkXShards(image_rdd)
