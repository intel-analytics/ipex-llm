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
from py4j.protocol import Py4JError

from bigdl.orca.data.utils import *
from bigdl.orca.data.shard import SparkXShards
from bigdl.orca import OrcaContext
from bigdl.dllib.nncontext import init_nncontext, ZooContext
from bigdl.dllib.utils.common import (get_node_and_core_number,
                                      get_spark_sql_context,
                                      get_spark_context)
from bigdl.dllib.utils import nest
from bigdl.dllib.utils.log4Error import invalidInputError

import numpy as np
import pyspark.sql.functions as F
from pyspark import RDD
import torch

from typing import (Union, List, Dict)

from typing import TYPE_CHECKING, Any
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    NoReturn
)

if TYPE_CHECKING:
    from numpy import ndarray
    from pyspark.rdd import PipelinedRDD, RDD


class ImageShards(SparkXShards):

    def __init__(self,
                 rdd: Union["PipelinedRDD", "RDD"],
                 transient: bool=False,
                 class_name: str=None) -> None:
        super(ImageShards, self).__init__(rdd, transient, class_name)


def read_im_file(filepath):
    from PIL import Image
    im = Image.open(filepath)
    return im

def read_im_file_cv(filepath):
    import cv2
    im = cv2.imread(filepath, mode='RGB')
    return im


def read_images(file_path, with_label=False, label_func=None):
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

    print(file_paths[0])
    num_files = len(file_paths)

    total_cores = node_num * core_num
    num_partitions = num_files if num_files < total_cores else total_cores
    rdd = sc.parallelize(file_paths, num_partitions)

    def loadImage(iterator):
        for f in iterator:
            image = read_im_file(f)
            yield {'filename': f, 'x': image}

    def loadImage1(f):
        image = read_im_file(f)
        return {'filename': f, 'x': image}

    im_rdd = rdd.mapPartitions(loadImage)

    # im_rdd = rdd.map(loadImage1)

    return ImageShards(im_rdd)

def read_images1(file_path):
    sc = OrcaContext.get_spark_context()
    node_num, core_num = get_node_and_core_number()

    def get_subdire(file_path):
        file_paths1 = []
        if isinstance(file_path, list):
            [file_paths1.extend(extract_one_path(path, os.environ)) for path in file_path]
        else:
            file_paths1 = extract_one_path(file_path, os.environ)

        if not file_paths1:
            invalidInputError(False,
                              "The file path is invalid or empty, please check your data")

        num_files = len(file_paths1)

        total_cores = node_num * core_num
        num_partitions = num_files if num_files < total_cores else total_cores
        rdd1 = sc.parallelize(file_paths1, num_partitions)
        return rdd1

    imfilerdd1= get_subdire(file_path + "/PNGImages/")
    imfilerdd2= get_subdire(file_path + "/PedMasks/")

    def loadImage1(f):
        image = read_im_file(f).convert("RGB")
        return image

    def loadmask(f):
        mask = read_im_file(f)
        mask = mask.resize((80, 80))
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        return target

    im_rdd = imfilerdd1.map(loadImage1)
    mask_rdd = imfilerdd2.map(loadmask)

    rdd = im_rdd.zip(mask_rdd)

    # print(rdd.collect()[:1])
    return ImageShards(rdd)