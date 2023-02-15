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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy import ndarray

from bigdl.orca.data.file import *
from bigdl.orca.data.utils import *
from bigdl.orca.data import SparkXShards
from pyspark.sql.functions import col
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.dllib.utils.common import get_node_and_core_number


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


def read_images(file_path: str,
                label_func: Callable = None,
                target_path: str = None,
                image_type: str = ".jpg",
                target_type: str = ".png",
                backend: str = 'pillow'):

    backend = backend.lower()
    invalidInputError(backend == "spark" or backend == "pillow",
                      "backend of read_images must be either spark or pillow")
    if backend == 'spark':
        images = read_images_spark(file_path, label_func, target_path, image_type, target_type)
    else:
        images = read_images_pil(file_path, label_func, target_path, image_type, target_type)
    return images


def read_images_pil(file_path: str,
                    label_func: Callable = None,
                    target_path: str = None,
                    image_type: str = ".jpg",
                    target_type: str = ".png"
                    ) -> "SparkXShards":
    """
    Read images into a SparkXShards using PIL.

    :param file_path: str. A HDFS path or local path of images.
    :param label_func: Callable. A function to get label from filename. Default is None.
    :param target_path: str. A HDFS path or local path of target images.
    :param image_type: str. Suffix of images.
    :param target_type: str. Suffix of target images.
    :return: A new SparkXShards of tuple of image, target.
    """
    sc = OrcaContext.get_spark_context()

    img_paths = get_file_paths(file_path)
    img_paths = list(filter(
        lambda fn: fn.endswith(image_type) and (not fn.startswith(".")), img_paths))
    num_files = len(img_paths)
    node_num, core_num = get_node_and_core_number()
    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores

    rdd = sc.parallelize(sorted(img_paths), num_partitions)

    def load_image(iterator):
        for f in iterator:
            img = open_image(f)
            if label_func:
                img = img, label_func(f)
            yield img

    image_rdd = rdd.mapPartitions(load_image)

    if target_path:
        invalidInputError(label_func is None,
                          "label_func should be None when reading targets directly")
        target_paths = get_file_paths(target_path)
        target_paths = list(filter(
            lambda fn: fn.endswith(target_type) and (not fn.startswith(".")), target_paths))

        invalidInputError(len(target_paths) == len(img_paths),
                          "number of target files and image files should be the same")

        t_rdd = sc.parallelize(sorted(target_paths), num_partitions)
        target_rdd = t_rdd.mapPartitions(load_image)
        image_rdd = image_rdd.zip(target_rdd)

    return SparkXShards(image_rdd)


def read_images_spark(file_path: str,
                      label_func: Callable = None,
                      target_path: str = None,
                      image_type: str = ".jpg",
                      target_type: str = ".png"
                      ) -> "SparkXShards":
    """
    Read images into a SparkXShards using Spark backend.

    :param file_path: str. A HDFS path or local path of images.
    :param label_func: Callable. A function to get label from filename. Default is None.
    :param target_path: str. A HDFS path or local path of target images.
    :param image_type: str. Suffix of images.
    :param target_type: str. Suffix of target images.
    :return: A new SparkXShards of tuple of image, target.
    """
    spark = OrcaContext.get_spark_session()

    img_paths = get_file_paths(file_path)
    img_paths = list(filter(
        lambda fn: fn.endswith(image_type) and (not fn.startswith(".")), img_paths))
    num_files = len(img_paths)
    node_num, core_num = get_node_and_core_number()
    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores

    image_df = spark.read.format("image").load(img_paths)

    def convert_bgr_array_to_rgb_array(img_array):
        B, G, R, = img_array.T
        return np.array((R, G, B)).T

    def to_pil(image_spark):
        """ PIL mode of image is converted from  channel of spark image,
            it is referenced from spark code of ImageSchema.scala, decode method
            https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/image/ImageSchema.scala#L131-L190
            mapping of spark image nChnnels: {1 : 'L', 3: 'RGB', 4: 'RGBA'}
        """
        if image_spark.image.nChannels == 4:
            mode = 'RGBA'
        elif image_spark.image.nChannels == 3:
            mode = 'RGB'
        elif image_spark.image.nChannels == 1:
            mode = 'L'
        else:
            invalidInputError(False, "invalid nChannels of spark image, "
                                     "please use read_images_pil instead")
        from PIL import Image
        img = Image.frombytes(mode=mode, data=bytes(image_spark.image.data),
                              size=[image_spark.image.width, image_spark.image.height])
        if mode in ['RGB', 'RGBA']:
            img = img.convert("RGB") if mode == 'RGBA' else img
            converted_img_array = convert_bgr_array_to_rgb_array(np.asarray(img)[:, :, :])
            img = Image.fromarray(converted_img_array)

        if label_func:
            img = img, label_func(image_spark.image.origin)

        return img

    image_rdd = image_df.orderBy(col("image.origin")).rdd\
        .map(lambda x: to_pil(x)).repartition(num_partitions)

    if target_path:
        invalidInputError(label_func is None,
                          "label_func should be None when reading targets directly")
        target_paths = get_file_paths(target_path)
        target_paths = list(filter(
            lambda fn: fn.endswith(target_type) and (not fn.startswith(".")), target_paths))

        invalidInputError(len(target_paths) == len(img_paths),
                          "number of target files and image files should be the same")
        target_df = spark.read.format("image").load(target_paths)

        target_rdd = target_df.orderBy(col("image.origin")).rdd\
            .map(lambda x: to_pil(x)).repartition(num_partitions)

        image_rdd = image_rdd.zip(target_rdd)
    return SparkXShards(image_rdd)
