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

from bigdl.dataset import mnist
from bigdl.dataset.transformer import *
from bigdl.optim.optimizer import *


def get_mnist(sc, data_type="train", location="/tmp/mnist"):
    """
    Get and normalize the mnist data. We would download it automatically
    if the data doesn't present at the specific location.

    :param sc: SparkContext
    :param data_type: training data or testing data
    :param location: Location storing the mnist
    :return: A RDD of (features: Ndarray, label: Ndarray)
    """
    (images, labels) = mnist.read_data_sets(location, data_type)
    images = sc.parallelize(images)
    labels = sc.parallelize(labels + 1) # Target start from 1 in BigDL
    record = images.zip(labels)
    return record


def process_mnist_rdd(sc, options):
    """
    Normalize mnist data and transform into RDD Samples
    """
    train_data = get_mnist(sc, "train", options.dataPath)\
        .map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TRAIN_MEAN, mnist.TRAIN_STD),
                                rec_tuple[1]))\
        .map(lambda t: Sample.from_ndarray(t[0], t[1]))
    test_data = get_mnist(sc, "test", options.dataPath)\
        .map(lambda rec_tuple: (normalizer(rec_tuple[0], mnist.TEST_MEAN, mnist.TEST_STD),
                                rec_tuple[1]))\
        .map(lambda t: Sample.from_ndarray(t[0], t[1]))
    return train_data, test_data


def get_end_trigger(options):
    if options.endTriggerType.lower() == "epoch":
        return MaxEpoch(options.endTriggerNum)
    else:
        return MaxIteration(options.endTriggerNum)

