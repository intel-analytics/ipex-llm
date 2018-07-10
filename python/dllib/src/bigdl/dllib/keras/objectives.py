#
# Copyright 2018 Analytics Zoo Authors.
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

import sys

from zoo.pipeline.api.keras.base import ZooKerasCreator
from bigdl.nn.criterion import Criterion
from bigdl.util.common import JTensor

if sys.version >= '3':
    long = int
    unicode = str


class LossFunction(ZooKerasCreator, Criterion):
    """
    The base class for Keras-style API objectives in Analytics Zoo.
    """
    def __init__(self, jvalue, bigdl_type, *args):
        super(Criterion, self).__init__(jvalue, bigdl_type, *args)

    @classmethod
    def of(cls, jloss, bigdl_type="float"):
        """
        Create a Python LossFunction from a JavaObject.

        # Arguments
        jloss: A java criterion object which created by Py4j
        """
        loss = LossFunction(bigdl_type, jloss)
        loss.value = jloss
        loss.bigdl_type = bigdl_type
        return loss


class SparseCategoricalCrossEntropy(LossFunction):
    """
    A loss often used in multi-class classification problems with SoftMax
    as the last layer of the neural network.

    By default, same as Keras, input(y_pred) is supposed to be probabilities of each class,
    and target(y_true) is supposed to be the class label starting from 0.

    # Arguments
    log_prob_as_input: Boolean. Whether to accept log-probabilities or probabilities
                       as input. Default is False and inputs should be probabilities.
    zero_based_label: Boolean. Whether target labels start from 0. Default is True.
                      If False, labels start from 1.
    weights: Weights of each class if you have an unbalanced training set.
    size_average: Boolean. Whether losses are averaged over observations for each
                  mini-batch. Default is True. If False, the losses are instead
                  summed for each mini-batch.
    padding_value: Int. If the target is set to this value, the training process
                   will skip this sample. In other words, the forward process will
                   return zero output and the backward process will also return
                   zero grad_input. Default is -1.

    >>> metrics = SparseCategoricalCrossEntropy()
    creating: createZooKerasSparseCategoricalCrossEntropy
    """
    def __init__(self, log_prob_as_input=False, zero_based_label=True,
                 weights=None, size_average=True, padding_value=-1, bigdl_type="float"):
        super(SparseCategoricalCrossEntropy, self).__init__(None, bigdl_type,
                                                            log_prob_as_input,
                                                            zero_based_label,
                                                            JTensor.from_ndarray(weights),
                                                            size_average,
                                                            padding_value)


class MeanAbsoluteError(LossFunction):
    """
    A loss that measures the mean absolute value of the element-wise difference
    between the input and the target.

    # Arguments
    size_average: Boolean. Whether losses are averaged over observations for each
              mini-batch. Default is True. If False, the losses are instead
              summed for each mini-batch.

    >>> metrics = MeanAbsoluteError()
    creating: createZooKerasMeanAbsoluteError
    """
    def __init__(self, size_average=True, bigdl_type="float"):
        super(MeanAbsoluteError, self).__init__(None, bigdl_type,
                                                size_average)


mae = MAE = MeanAbsoluteError
