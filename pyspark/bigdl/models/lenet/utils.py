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
    Get mnist dataset and parallelize into RDDs.
    Data would be downloaded automatically if it doesn't present at the specific location.

    :param sc: SparkContext.
    :param data_type: "train" for training data and "test" for testing data.
    :param location: Location to store mnist dataset.
    :return: RDD of (features: ndarray, label: ndarray).
    """
    (images, labels) = mnist.read_data_sets(location, data_type)
    images = sc.parallelize(images)
    labels = sc.parallelize(labels + 1)  # Target start from 1 in BigDL
    record = images.zip(labels)
    return record


def preprocess_mnist(sc, options):
    """
    Preprocess mnist dataset.
    Normalize and transform into Sample of RDDs.
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
    """
    When to end the optimization based on input option.
    """
    if options.endTriggerType.lower() == "epoch":
        return MaxEpoch(options.endTriggerNum)
    else:
        return MaxIteration(options.endTriggerNum)


def validate_optimizer(optimizer, test_data, options):
    """
    Set validation and checkpoint for distributed optimizer.
    """
    optimizer.set_validation(
        batch_size=options.batchSize,
        val_rdd=test_data,
        trigger=EveryEpoch(),
        val_method=[Top1Accuracy()]
    )
    optimizer.set_checkpoint(EveryEpoch(), options.checkpointPath)
