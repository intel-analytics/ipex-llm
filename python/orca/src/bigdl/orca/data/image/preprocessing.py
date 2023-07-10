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
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from numpy import ndarray

from bigdl.orca.data.file import *
from bigdl.orca.data.utils import *
from bigdl.orca.data import SparkXShards
from pyspark.sql.functions import col, explode, collect_list
from bigdl.dllib.utils.log4Error import invalidInputError
from bigdl.dllib.utils.common import get_node_and_core_number
import os.path as osp
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore


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
                label_func: Callable=None,
                target_path: str=None,
                image_type: str=".jpg",
                target_type: str=".png",
                backend: str="pillow"):

    backend = backend.lower()
    invalidInputError(backend == "spark" or backend == "pillow",
                      "backend of read_images must be either spark or pillow")
    if backend == 'spark':
        images = read_images_spark(file_path, label_func, target_path, image_type, target_type)
    else:
        images = read_images_pil(file_path, label_func, target_path, image_type, target_type)
    return images


def read_images_pil(file_path: str,
                    label_func: Callable=None,
                    target_path: str=None,
                    image_type: str=".jpg",
                    target_type: str=".png"
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
                      label_func: Callable=None,
                      target_path: str=None,
                      image_type: str=".jpg",
                      target_type: str=".png"
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
        from PIL import Image, ImageDraw
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


def read_voc(file_path: str="VOCdevkit",
             split_names: Optional[List[Tuple[int, str]]]=None,
             classes: Optional[List[str]]=None,
             diff: bool=False,
             max_samples: int=None
             ) -> "SparkXShards":
    """
    Read VOC images into a SparkXShards. code is ported from
    https://github.com/intel-analytics/BigDL/blob/main/python/orca/src/bigdl/orca/data/
    image/voc_dataset.py

    :param file_path: str. A HDFS path or local path of images.
    :param split_names: splits_names: tuple, ((year, trainval)).
    :param classes: str. A HDFS path or local path of target images.
    :param diff: boolean, False ignore voc xml difficult value.

    :param max_samples: int. max samples returned.
    :return: A new SparkXShards of tuple of image, target.
             target is a ndarray of [[x1, y1, x2, y2, cls, difficult]]
    """

    spark = OrcaContext.get_spark_session()
    anno_path = osp.join('{}', 'Annotations', '{}.xml')
    image_path = osp.join('{}', 'JPEGImages', '{}.jpg')

    split_names = split_names if split_names else [(2009, "trainval")]
    CLASSES = classes if classes else ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                                       'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                                       'train', 'tvmonitor']
    cat2label = {cat: i for i, cat in enumerate(CLASSES)}

    def get_imgids(splits_names: List[Tuple[int, str]]) -> List[Tuple[str, str]]:
        img_ids = []
        for year, txtname in splits_names:
            vocfolder = osp.join(file_path, "VOC{}".format(year))
            txtpath = osp.join(vocfolder, 'ImageSets', 'Main', txtname + '.txt')
            try:
                with open(txtpath, 'r', encoding='utf-8') as f:
                    img_ids += [(vocfolder, line.strip()) for line in f.readlines()]
            except:
                continue
        return img_ids

    def _check_label(label: "ndarray", width: int=1, height: int=1) -> None:
        """Check if label is correct."""
        from bigdl.dllib.utils.log4Error import invalidInputError
        xmin = label[:, 0]
        ymin = label[:, 1]
        xmax = label[:, 2]
        ymax = label[:, 3]
        invalidInputError(((0 <= xmin) & (xmin < width)).any(),
                          "xmin must in [0, {}), given {}".format(width, xmin))
        invalidInputError(((0 <= ymin) & (ymin < height)).any(),
                          "ymin must in [0, {}), given {}".format(height, ymin))
        invalidInputError(((xmin < xmax) & (xmax <= width)).any(),
                          "xmax must in ({}, {}], given {}".format(xmin, width, xmax))
        invalidInputError(((ymin < ymax) & (ymax <= height)).any(),
                          "ymax must in ({}, {}], given {}".format(ymin, height, ymax))

    def get_img_label(f):
        image_file = image_path.format(*f)
        label_file = anno_path.format(*f)

        root = ET.parse(label_file).getroot()
        try:
            img = open_image(image_file)
        except FileNotFoundError as e:
            invalidOperationError(False, str(e), cause=e)

        width, height = img.size

        # load label [[x1, y1, x2, y2, cls, difficult]]
        label = []
        for obj in root.iter('object'):
            try:
                difficult = int(obj.find('difficult').text)
            except ValueError:
                difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in CLASSES:
                logging.warning(f"{cls_name} isn't included in {CLASSES}")
                continue
            cls_id = cat2label[cls_name]
            xml_box = obj.find('bndbox')
            xmin = float(int(xml_box.find('xmin').text) / width)
            ymin = float(int(xml_box.find('ymin').text) / height)
            xmax = float(int(xml_box.find('xmax').text) / width)
            ymax = float(int(xml_box.find('ymax').text) / height)
            label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        label = np.array(label).astype(np.float32)
        if not diff:
            label = label[..., :5]

        try:
            _check_label(label, width, height)
        except RuntimeError as e:
            logging.warning("Invalid label at %s, %s", anno_path, e)
        return img, label

    img_paths = get_imgids(split_names)
    num_files = len(img_paths)
    node_num, core_num = get_node_and_core_number()
    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores
    rdd = spark.sparkContext.parallelize(img_paths, num_partitions)

    def load_image(iterator):
        for f in iterator:
            img, labels = get_img_label(f)
            yield img, labels

    image_rdd = rdd.mapPartitions(load_image)
    if max_samples:
        image_rdd = spark.sparkContext.parallelize(image_rdd.take(max_samples))
    return SparkXShards(image_rdd)


def read_coco(file_path: str,
              split: str="train"):
    """
    Read coco 2017 images into a SparkXShards using Spark backend.

    :param file_path: str. A HDFS path or local path of images.
    :param split: str. a split to read.
    :return: A new SparkXShards of tuple of image, target.
            target is a dictionary with a list of bbox, list of category_id, list of area,
            list of iscrouwd, list of segmentation.
            Each bbox contains four values in pixels [x_min, y_min, width, height].
    """
    spark = OrcaContext.get_spark_session()
    df = spark.read.json(file_path + "/annotations/instances_" + split + "2017.json")
    ann_df = df.select(explode(col("annotations")).alias("annotations"))
    ann_df = ann_df.select(col("annotations.area").alias("area"),
                           col("annotations.bbox").alias("bbox"),
                           col("annotations.category_id").alias("category_id"),
                           col("annotations.id").alias("id"),
                           col("annotations.image_id").alias("image_id"),
                           col("annotations.iscrowd").alias("iscrowd"),
                           col("annotations.segmentation").alias("segmentation"))

    ann_df = ann_df.groupby("image_id")\
        .agg(collect_list(col("bbox")).alias("bbox"),
             collect_list(col("category_id")).alias("category_id"),
             collect_list(col("area")).alias("area"),
             collect_list(col("iscrowd")).alias("iscrowd"),
             collect_list(col("segmentation")).alias("segmentation"))

    ann_rdd = ann_df.rdd.map(
        lambda x: (x['image_id'],
                   (x['bbox'], x['category_id'], x["area"], x["iscrowd"], x["segmentation"])))

    image_path = file_path + split + "2017/"
    file_names = get_file_paths(image_path)
    file_names = list(filter(
        lambda fn: fn.endswith('.jpg') and (not fn.startswith(".")), file_names))
    num_files = len(file_names)
    node_num, core_num = get_node_and_core_number()
    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores
    # file_names is a rdd of(id, filename)
    file_names = spark.sparkContext.parallelize(file_names, num_partitions)\
        .map(lambda x: (int(x.split("/")[-1].split(".")[0]), x))

    def load_image(iterator):
        for f in iterator:
            try:
                img = open_image(f[1]).convert("RGB")
                yield f[0], img
            except FileNotFoundError as e:
                invalidOperationError(False, str(e), cause=e)
            yield f[0], None

    image_rdd = file_names.mapPartitions(load_image)\
        .filter(lambda x: x[1])

    def transform_data(x):
        image = x[1][0]
        target = dict()
        target['bbox'] = x[1][1][0]
        target['category_id'] = x[1][1][1]
        target['area'] = x[1][1][2]
        target['iscrowd'] = x[1][1][3]
        target['segmentation'] = x[1][1][4]
        return image, target

    out_rdd = image_rdd.join(ann_rdd)
    out_rdd = out_rdd.map(lambda x: transform_data(x))

    return SparkXShards(out_rdd)
