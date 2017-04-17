#
# Licensed to Intel Corporation under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# Intel Corporation licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Part of the code originally from Tensorflow

import numpy as np
from os import listdir
from os.path import join, basename
import struct
from scipy import misc
from util.common import Sample


def read_local_path(folder, has_label=True):
    # read directory, create map
    dirs = listdir(folder)
    # create image path and label list
    image_paths = []
    if has_label:
        dirs.sort()
        for d in dirs:
            for f in listdir(join(folder, d)):
                image_paths.append((join(join(folder, d), f), dirs.index(d) + 1))
    else:
        for f in dirs:
            image_paths.append((join(folder, f), -1))
    return image_paths


def read_local(sc, folder, normalize=255.0, has_label=True):
    """
    Read images from local directory
    :param sc: spark context
    :param folder: local directory
    :param normalize: normalization value
    :param has_label: whether the image folder contains label
    :return: RDD of sample
    """
    # read directory, create image paths list
    image_paths = read_local_path(folder, has_label)
    # create rdd
    image_paths_rdd = sc.parallelize(image_paths)
    samples = image_paths_rdd.map(lambda (path, label): (misc.imread(path), np.array(label))) \
        .map(lambda (img, label):
             (resize_image(img, 256, 256), label)) \
        .map(lambda (features, label):
             (((features & 0xff) / normalize), label)) \
        .map(lambda (features, label):
             Sample.from_ndarray(features, label))
    return samples


def read_local_with_name(sc, folder, normalize=255.0, has_label=True):
    """
    Read images from local directory
    :param sc: spark context
    :param folder: local directory
    :param normalize: normalization value
    :param has_label: whether the image folder contains label
    :return: RDD of sample
    """
    # read directory, create image paths list
    image_paths = read_local_path(folder, has_label)
    # create rdd
    image_paths_rdd = sc.parallelize(image_paths)
    samples = image_paths_rdd.map(lambda (path, label): (misc.imread(path), np.array(label), basename(path))) \
        .map(lambda (img, label, name):
             (resize_image(img, 256, 256), label, name)) \
        .map(lambda (features, label, name):
             (((features & 0xff) / normalize), label, name)) \
        .map(lambda (features, label, name):
            (Sample.from_ndarray(features, label), name))
    return samples


def resize_image(img, resize_width, resize_height):
    return misc.imresize(img, (resize_width, resize_height))


def read_seq_file(sc, path, normalize=255.0, has_name=False):
    """
    Read images from sequence file
    :param sc: spark context
    :param path: location of sequence file
    :param normalize: normalize index for the image
    :param has_name: whether the sequence file includes image name
    :return: RDD of sample image
    """
    raw = sc.sequenceFile(path, "org.apache.hadoop.io.Text", "org.apache.hadoop.io.BytesWritable")

    def parse(data):
        img = data[1]
        length = len(img)-8
        metrics = struct.unpack('>ii', img[0:8])
        width = metrics[0]
        height = metrics[1]
        features = np.array(img[8:], dtype="int8")
        normalized_features = (features & 0xff) / normalize
        if has_name:
            key = data[0].split('\n')
            name = key[0]
            label = key[1]
            sample = Sample(normalized_features, [int(label)], features_shape=(height, width, length / width / height),
                            label_shape=[1])
            return sample, name
        else:
            label = data[0]
            sample = Sample(normalized_features, [int(label)], features_shape=(height, width, length / width / height),
                            label_shape=[1])
            return sample
    return raw.map(parse)


def load_mean_file(mean_file):
    """
    Read mean file which contains means for every pixel
    :param mean_file:
    :return:
    """
    mean_array = np.load(mean_file).transpose(1,2,0)
    return mean_array / 255.0
