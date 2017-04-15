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


import gzip

import numpy as np
from os import listdir
from os.path import join
from pyspark import SparkContext
from scipy import misc
import struct
import cv2


from util.common import _py2java
from util.common import _java2py
from util.common import Sample
from util.common import callJavaFunc


def read_local(sc, folder, normalize=255.0):
    '''
    Read images from local directory
    :param sc: spark context
    :param folder: local directory
    :param normalize: normalization value
    :return: RDD of sample
    '''
    # read directory, create map
    dirs = listdir(folder)
    dirs.sort()
    # create image list
    images = []
    for d in dirs:
        for f in listdir(join(folder, d)):
            images.append((join(join(folder, d), f), dirs.index(d)+1))
    # create rdd
    images = sc.parallelize(images)
    # samples = images.map(lambda (path, label): (misc.imread(path), np.array([label]))) \
    samples = images.map(lambda (path, label): (cv2.imread(path, 1), np.array([label]))) \
        .map(lambda (img, label):
             (resize_image(256, 256), label)) \
        .map(lambda (features, label):
             (((features & 0xff) / normalize), label)) \
        .map(lambda (features, label):
             Sample.from_ndarray(features, label))
    return samples


def resize_image(img, resize_width, resize_height):
    # return misc.imresize(img, (resize_width, resize_height))
    return cv2.resize(img,(resize_width, resize_height), interpolation = cv2.INTER_AREA)


def read_seq_file(sc, path):
    '''
    Read images from sequence file
    :param sc: spark context
    :param path: location of sequence file
    :return: RDD of sample
    '''
    raw = sc.sequenceFile(path, "org.apache.hadoop.io.Text", "org.apache.hadoop.io.BytesWritable")
    def parse(data):
        label = data[0]
        img = data[1]
        length = len(img)-8
        metrics = struct.unpack('>ii', img[0:8])
        width = metrics[0]
        height = metrics[1]
        features = np.array(img[8:], dtype="int8")
        sample = Sample(features, [int(label)], features_shape=(width, height, length/width/height), label_shape=[1])
        return sample
    return raw.map(parse)


def load_mean_file(mean_file):
    '''
    Read mean file which contains means for every pixel
    :param mean_file:
    :return:
    '''
    # means = np.fromfile(mean_file)
    mean_array = np.load(mean_file).transpose(1,2,0)
    return mean_array / 255.0


if __name__ == "__main__":
    read_data_sets("/tmp/mnist/")
