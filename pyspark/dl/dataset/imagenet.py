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


from util.common import _py2java
from util.common import _java2py
from util.common import Sample


def read_label(data):
    dataArr = data.split("\n")
    if len(dataArr) == 1:
        return dataArr[0]
    else:
        return dataArr[1]


def read_local(folder, sc, normalize=255.0):
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
    samples = images.map(lambda (path, label): (misc.imread(path), np.array([label]))) \
        .map(lambda (img, label):
             (resize_image(256, 256), label)) \
        .map(lambda (features, label):
             (((features & 0xff) / normalize), label)) \
        .map(lambda (features, label):
             Sample.from_ndarray(features, label))
    return samples


def resize_image(img, resize_width, resize_height):
    return misc.imresize(img, (resize_width, resize_height))


def read_data_sets(sc, folder, file_type, node_num, core_num, data_type="train", normalize=255.0):
    Text = "org.apache.hadoop.io.Text"
    path = join(folder, data_type)

    if file_type == "seq":
        raw = sc.sequenceFile(path, Text, Text, minSplits=node_num * core_num)
        for data in raw.collect():
            print data
        # data = raw.map(lambda image: (_java2py(_py2java(sc, image[0]).copyBytes),
        #                               float(read_label(image[1]))))
    else:
        return read_local(path, sc, normalize)


def load_mean_file(mean_file):
    '''

    :param mean_file:
    :return:
    '''
    means = np.fromfile(mean_file, dtype=float)
    return [float(i) for i in means.ravel()][10:]


if __name__ == "__main__":
    read_data_sets("/tmp/mnist/")
