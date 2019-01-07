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
    weights: A Numpy array. Weights of each class if you have an unbalanced training set.
    size_average: Boolean. Whether losses are averaged over observations for each
                  mini-batch. Default is True. If False, the losses are instead
                  summed for each mini-batch.
    padding_value: Int. If the target is set to this value, the training process
                   will skip this sample. In other words, the forward process will
                   return zero output and the backward process will also return
                   zero grad_input. Default is -1.

    >>> loss = SparseCategoricalCrossEntropy()
    creating: createZooKerasSparseCategoricalCrossEntropy
    >>> import numpy as np
    >>> np.random.seed(1128)
    >>> weights = np.random.uniform(0, 1, (2,)).astype("float32")
    >>> loss = SparseCategoricalCrossEntropy(weights=weights)
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

    >>> loss = MeanAbsoluteError()
    creating: createZooKerasMeanAbsoluteError
    """
    def __init__(self, size_average=True, bigdl_type="float"):
        super(MeanAbsoluteError, self).__init__(None, bigdl_type,
                                                size_average)


mae = MAE = MeanAbsoluteError


class BinaryCrossEntropy(LossFunction):
    """
    A loss that measures the Binary Cross Entropy between the target and the output

    # Arguments
    size_average: Boolean. Whether losses are averaged over observations for each
                mini-batch. Default is True. If False, the losses are instead
                summed for each mini-batch.
    weights: weights over the input dimension

    >>> loss = BinaryCrossEntropy()
    creating: createZooKerasBinaryCrossEntropy
    """
    def __init__(self, weights=None, size_average=True, bigdl_type="float"):
        super(BinaryCrossEntropy, self).__init__(None, bigdl_type,
                                                 JTensor.from_ndarray(weights),
                                                 size_average)


class CategoricalCrossEntropy(LossFunction):
    """
    This is same with cross entropy criterion, except the target tensor is a one-hot tensor

    >>> loss = CategoricalCrossEntropy()
    creating: createZooKerasCategoricalCrossEntropy
    """
    def __init__(self, bigdl_type="float"):
        super(CategoricalCrossEntropy, self).__init__(None, bigdl_type)


class CosineProximity(LossFunction):
    """
    The negative of the mean cosine proximity between predictions and targets.
    The cosine proximity is defined as below:
        x'(i) = x(i) / sqrt(max(sum(x(i)^2), 1e-12))
        y'(i) = y(i) / sqrt(max(sum(x(i)^2), 1e-12))
        cosine_proximity(x, y) = mean(-1 * x'(i) * y'(i))

    >>> loss = CosineProximity()
    creating: createZooKerasCosineProximity
    """
    def __init__(self, bigdl_type="float"):
        super(CosineProximity, self).__init__(None, bigdl_type)


class Hinge(LossFunction):
    """
    Creates a criterion that optimizes a two-class classification
    hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.

    # Arguments:
    margin: Float. Default is 1.0.
    size_average: Boolean. Whether losses are averaged over observations for each
                  mini-batch. Default is True. If False, the losses are instead
                  summed for each mini-batch.

    >>> loss = Hinge()
    creating: createZooKerasHinge
    """
    def __init__(self, margin=1.0, size_average=True, bigdl_type="float"):
        super(Hinge, self).__init__(None, bigdl_type, float(margin), size_average)


class KullbackLeiblerDivergence(LossFunction):
    """
    Loss calculated as:y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    >>> loss = KullbackLeiblerDivergence()
    creating: createZooKerasKullbackLeiblerDivergence
    """
    def __init__(self, bigdl_type="float"):
        super(KullbackLeiblerDivergence, self).__init__(None, bigdl_type)


class MeanAbsolutePercentageError(LossFunction):
    """
    It caculates diff = K.abs((y - x) / K.clip(K.abs(y), K.epsilon(), Double.MaxValue))
    and return 100 * K.mean(diff) as outpout

    >>> loss = MeanAbsolutePercentageError()
    creating: createZooKerasMeanAbsolutePercentageError
    """
    def __init__(self, bigdl_type="float"):
        super(MeanAbsolutePercentageError, self).__init__(None, bigdl_type)


mape = MAPE = MeanAbsolutePercentageError


class MeanSquaredError(LossFunction):
    """
    A loss that measures the mean squared value of the element-wise difference
    between the input and the target.

    # Arguments
    size_average: Boolean. Whether losses are averaged over observations for each
              mini-batch. Default is True. If False, the losses are instead
              summed for each mini-batch.

    >>> loss = MeanSquaredError()
    creating: createZooKerasMeanSquaredError
    """
    def __init__(self, size_average=True, bigdl_type="float"):
        super(MeanSquaredError, self).__init__(None, bigdl_type,
                                               size_average)


mse = MSE = MeanSquaredError


class MeanSquaredLogarithmicError(LossFunction):
    """
    It calculates:
    first_log = K.log(K.clip(y, K.epsilon(), Double.MaxValue) + 1.)
    second_log = K.log(K.clip(x, K.epsilon(), Double.MaxValue) + 1.)
    and output K.mean(K.square(first_log - second_log))

    >>> loss = MeanSquaredLogarithmicError()
    creating: createZooKerasMeanSquaredLogarithmicError
    """
    def __init__(self, bigdl_type="float"):
        super(MeanSquaredLogarithmicError, self).__init__(None, bigdl_type)


msle = MSLE = MeanSquaredLogarithmicError


class Poisson(LossFunction):
    """
    Loss calculated as: K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)

    >>> loss = Poisson()
    creating: createZooKerasPoisson
    """
    def __init__(self, bigdl_type="float"):
        super(Poisson, self).__init__(None, bigdl_type)


class SquaredHinge(LossFunction):
    """
    Creates a criterion that optimizes a two-class classification
    squared hinge loss (margin-based loss)
    between input x (a Tensor of dimension 1) and output y.

    # Arguments:
    margin: Float. Default is 1.0.
    size_average: Boolean. Whether losses are averaged over observations for each
                  mini-batch. Default is True. If False, the losses are instead
                  summed for each mini-batch.

    >>> loss = SquaredHinge()
    creating: createZooKerasSquaredHinge
    """
    def __init__(self, margin=1.0, size_average=False, bigdl_type="float"):
        super(SquaredHinge, self).__init__(None, bigdl_type, float(margin), size_average)


class RankHinge(LossFunction):
    """
    Hinge loss for pairwise ranking problems.

    # Arguments:
    margin: Float. Default is 1.0.

    >>> loss = RankHinge()
    creating: createZooKerasRankHinge
    """
    def __init__(self, margin=1.0, bigdl_type="float"):
        super(RankHinge, self).__init__(None, bigdl_type, float(margin))
