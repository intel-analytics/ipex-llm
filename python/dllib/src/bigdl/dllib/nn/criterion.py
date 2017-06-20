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


import sys

from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import JTensor
from bigdl.nn.layer import Layer
import numpy as np

if sys.version >= '3':
    long = int
    unicode = str


class Criterion(JavaValue):
    """
    Criterion is helpful to train a neural network.
    Given an input and a target, they compute a gradient according to a given loss function.
    """
    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)
        self.bigdl_type = bigdl_type

    def __str__(self):
        return self.value.toString()

    def forward(self, input, target):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        Takes an input object, and computes the corresponding loss of the criterion,
        compared with `target`

        :param input: ndarray or list of ndarray
        :param target: ndarray or list of ndarray
        :return: value of loss
        """
        output = callBigDlFunc(self.bigdl_type,
                               "criterionForward",
                               self.value,
                               Layer.check_input(input),
                               Layer.check_input(target))
        return output

    def backward(self, input, target):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        Performs a back-propagation step through the criterion, with respect to the given input.

        :param input: ndarray or list of ndarray
        :param target: ndarray or list of ndarray
        :return: ndarray
        """
        output = callBigDlFunc(self.bigdl_type,
                               "criterionBackward",
                               self.value,
                               Layer.check_input(input),
                               Layer.check_input(target))
        return Layer.convert_output(output)

    @classmethod
    def of(cls, jcriterion, bigdl_type="float"):
        """
        Create a python Criterion by a java criterion object

        :param jcriterion: A java criterion object which created by Py4j
        :return: a criterion.
        """
        criterion = Criterion(bigdl_type, jcriterion)
        criterion.value = jcriterion
        criterion.bigdl_type = bigdl_type
        return criterion


class ClassNLLCriterion(Criterion):

    '''
    The negative log likelihood criterion.
    It is useful to train a classification problem with n classes.
    If provided, the optional argument weights should be a 1D Tensor
    assigning weight to each of the classes.


    :param weights: weights of each class
    :param size_average: whether to average or not


    >>> np.random.seed(123)
    >>> weights = np.random.uniform(0, 1, (2,)).astype("float32")
    >>> classNLLCriterion = ClassNLLCriterion(weights,True)
    creating: createClassNLLCriterion
    >>> classNLLCriterion = ClassNLLCriterion()
    creating: createClassNLLCriterion
    '''

    def __init__(self,
                 weights=None,
                 size_average=True,
                 bigdl_type="float"):
        super(ClassNLLCriterion, self).__init__(None, bigdl_type,
                                                JTensor.from_ndarray(weights),
                                                size_average)


class MSECriterion(Criterion):

    '''
    Creates a criterion that measures the mean squared error between n elements
    in the input x and output y:
```
    loss(x, y) = 1/n \sum |x_i - y_i|^2
```


    If x and y are d-dimensional Tensors with a total of n elements,
    the sum operation still operates over all the elements, and divides by n.
    The two Tensors must have the same number of elements (but their sizes might be different).
    The division by n can be avoided if one sets the internal variable sizeAverage to false.
    By default, the losses are averaged over observations for each minibatch. However,
     if the field sizeAverage is set to false, the losses are instead summed.


    >>> mSECriterion = MSECriterion()
    creating: createMSECriterion
    '''

    def __init__(self, bigdl_type="float"):
        super(MSECriterion, self).__init__(None, bigdl_type)


class AbsCriterion(Criterion):

    '''
    measures the mean absolute value of the element-wise difference between input


    >>> absCriterion = AbsCriterion(True)
    creating: createAbsCriterion
    '''

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(AbsCriterion, self).__init__(None, bigdl_type,
                                           size_average)


class ClassSimplexCriterion(Criterion):

    '''
    ClassSimplexCriterion implements a criterion for classification.
    It learns an embedding per class, where each class' embedding is a
    point on an (N-1)-dimensional simplex, where N is the number of classes.

    :param nClasses: the number of classes.


    >>> classSimplexCriterion = ClassSimplexCriterion(2)
    creating: createClassSimplexCriterion
    '''

    def __init__(self,
                 n_classes,
                 bigdl_type="float"):
        super(ClassSimplexCriterion, self).__init__(None, bigdl_type,
                                                    n_classes)


class CosineEmbeddingCriterion(Criterion):

    '''
    Creates a criterion that measures the loss given an input x = {x1, x2},
    a table of two Tensors, and a Tensor label y with values 1 or -1.


    :param margin: a number from -1 to 1, 0 to 0.5 is suggested


    >>> cosineEmbeddingCriterion = CosineEmbeddingCriterion(1e-5, True)
    creating: createCosineEmbeddingCriterion
    '''

    def __init__(self,
                 margin=0.0,
                 size_average=True,
                 bigdl_type="float"):
        super(CosineEmbeddingCriterion, self).__init__(None, bigdl_type,
                                                       margin,
                                                       size_average)


class DistKLDivCriterion(Criterion):

    '''
    The Kullback-Leibler divergence criterion


    :param sizeAverage:


    >>> distKLDivCriterion = DistKLDivCriterion(True)
    creating: createDistKLDivCriterion
    '''

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(DistKLDivCriterion, self).__init__(None, bigdl_type,
                                                 size_average)


class HingeEmbeddingCriterion(Criterion):

    '''
    Creates a criterion that measures the loss given an
    input x which is a 1-dimensional vector and a label y (1 or -1).
    This is usually used for measuring whether two inputs are similar
    or dissimilar,
    e.g. using the L1 pairwise distance, and is typically used for
    learning nonlinear embeddings or semi-supervised learning.


    If x and y are n-dimensional Tensors, the sum operation still operates
    over all the elements, and divides by n (this can be avoided if one sets
    the internal variable sizeAverage to false). The margin has a default
    value of 1, or can be set in the constructor.


    >>> hingeEmbeddingCriterion = HingeEmbeddingCriterion(1e-5, True)
    creating: createHingeEmbeddingCriterion
    '''

    def __init__(self,
                 margin=1.0,
                 size_average=True,
                 bigdl_type="float"):
        super(HingeEmbeddingCriterion, self).__init__(None, bigdl_type,
                                                      margin,
                                                      size_average)


class L1HingeEmbeddingCriterion(Criterion):

    '''
    Creates a criterion that measures the loss given an input x = {x1, x2},
    a table of two Tensors, and a label y (1 or -1):


    :param margin:


    >>> l1HingeEmbeddingCriterion = L1HingeEmbeddingCriterion(1e-5)
    creating: createL1HingeEmbeddingCriterion
    >>> l1HingeEmbeddingCriterion = L1HingeEmbeddingCriterion()
    creating: createL1HingeEmbeddingCriterion
    '''

    def __init__(self,
                 margin=1.0,
                 bigdl_type="float"):
        super(L1HingeEmbeddingCriterion, self).__init__(None, bigdl_type,
                                                        margin)


class MarginCriterion(Criterion):

    '''
    Creates a criterion that optimizes a two-class classification hinge loss (margin-based loss)
    between input x (a Tensor of dimension 1) and output y.


    :param margin: if unspecified, is by default 1.
    :param size_average: size average in a mini-batch


    >>> marginCriterion = MarginCriterion(1e-5, True)
    creating: createMarginCriterion
    '''

    def __init__(self,
                 margin=1.0,
                 size_average=True,
                 bigdl_type="float"):
        super(MarginCriterion, self).__init__(None, bigdl_type,
                                              margin,
                                              size_average)


class MarginRankingCriterion(Criterion):

    '''
    Creates a criterion that measures the loss given an input x = {x1, x2},
    a table of two Tensors of size 1 (they contain only scalars), and a label y (1 or -1).
    In batch mode, x is a table of two Tensors of size batchsize, and y is a Tensor of size
    batchsize containing 1 or -1 for each corresponding pair of elements in the input Tensor.
    If y == 1 then it assumed the first input should be ranked higher (have a larger value) than
    the second input, and vice-versa for y == -1.


    :param margin:


    >>> marginRankingCriterion = MarginRankingCriterion(1e-5, True)
    creating: createMarginRankingCriterion
    '''

    def __init__(self,
                 margin=1.0,
                 size_average=True,
                 bigdl_type="float"):
        super(MarginRankingCriterion, self).__init__(None, bigdl_type,
                                                     margin,
                                                     size_average)


class MultiCriterion(Criterion):

    '''
    a weighted sum of other criterions each applied to the same input and target


    >>> multiCriterion = MultiCriterion()
    creating: createMultiCriterion
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(MultiCriterion, self).__init__(None, bigdl_type)


class MultiLabelMarginCriterion(Criterion):

    '''
    Creates a criterion that optimizes a multi-class multi-classification hinge loss (
    margin-based loss) between input x and output y (which is a Tensor of target class indices)


    :param size_average: size average in a mini-batch


    >>> multiLabelMarginCriterion = MultiLabelMarginCriterion(True)
    creating: createMultiLabelMarginCriterion
    '''

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(MultiLabelMarginCriterion, self).__init__(None, bigdl_type,
                                                        size_average)


class ParallelCriterion(Criterion):

    '''
    ParallelCriterion is a weighted sum of other criterions each applied to a different input
    and target. Set repeatTarget = true to share the target for criterions.


    Use add(criterion[, weight]) method to add criterion. Where weight is a scalar(default 1).


    :param repeat_target: Whether to share the target for all criterions.


    >>> parallelCriterion = ParallelCriterion(True)
    creating: createParallelCriterion
    '''

    def __init__(self,
                 repeat_target=False,
                 bigdl_type="float"):
        super(ParallelCriterion, self).__init__(None, bigdl_type,
                                                repeat_target)


class SmoothL1Criterion(Criterion):

    '''
    Creates a criterion that can be thought of as a smooth version of the AbsCriterion.
    It uses a squared term if the absolute element-wise error falls below 1.
    It is less sensitive to outliers than the MSECriterion and in some
    cases prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).
```
                          | 0.5 * (x_i - y_i)^2^, if |x_i - y_i| < 1
    loss(x, y) = 1/n \sum |
                          | |x_i - y_i| - 0.5,   otherwise
```
    If x and y are d-dimensional Tensors with a total of n elements,
    the sum operation still operates over all the elements, and divides by n.
    The division by n can be avoided if one sets the internal variable sizeAverage to false


    :param size_average: whether to average the loss


    >>> smoothL1Criterion = SmoothL1Criterion(True)
    creating: createSmoothL1Criterion
    '''

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(SmoothL1Criterion, self).__init__(None, bigdl_type,
                                                size_average)


class SmoothL1CriterionWithWeights(Criterion):

    '''
    a smooth version of the AbsCriterion
    It uses a squared term if the absolute element-wise error falls below 1.
    It is less sensitive to outliers than the MSECriterion and in some cases
    prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).

```
   d = (x - y) * w_in
   loss(x, y, w_in, w_out)
              | 0.5 * (sigma * d_i)^2 * w_out          if |d_i| < 1 / sigma / sigma
   = 1/n \sum |
              | (|d_i| - 0.5 / sigma / sigma) * w_out   otherwise
```

    >>> smoothL1CriterionWithWeights = SmoothL1CriterionWithWeights(1e-5, 1)
    creating: createSmoothL1CriterionWithWeights
    '''

    def __init__(self,
                 sigma,
                 num=0,
                 bigdl_type="float"):
        super(SmoothL1CriterionWithWeights, self).__init__(None, bigdl_type,
                                                           sigma,
                                                           num)


class SoftmaxWithCriterion(Criterion):

    '''
    Computes the multinomial logistic loss for a one-of-many classification task,
    passing real-valued predictions through a softmax to get a probability distribution over classes.
    It should be preferred over separate SoftmaxLayer + MultinomialLogisticLossLayer
    as its gradient computation is more numerically stable.

    :param ignoreLabel:   (optional) Specify a label value thatshould be ignored when computing the loss.
    :param normalizeMode: How to normalize the output loss.


    >>> softmaxWithCriterion = SoftmaxWithCriterion()
    creating: createSoftmaxWithCriterion
    >>> softmaxWithCriterion = SoftmaxWithCriterion(1, "FULL")
    creating: createSoftmaxWithCriterion
    '''

    def __init__(self,
                 ignore_label=None,
                 normalize_mode="VALID",
                 bigdl_type="float"):
        super(SoftmaxWithCriterion, self).__init__(None, bigdl_type,
                                                   ignore_label,
                                                   normalize_mode)


class TimeDistributedCriterion(Criterion):
    '''
    This class is intended to support inputs with 3 or more dimensions.
    Apply Any Provided Criterion to every temporal slice of an input.


    :param criterion: embedded criterion
    :param size_average: whether to divide the sequence length


    >>> td = TimeDistributedCriterion(ClassNLLCriterion())
    creating: createClassNLLCriterion
    creating: createTimeDistributedCriterion
    '''

    def __init__(self, criterion, size_average=False, bigdl_type="float"):
        super(TimeDistributedCriterion, self).__init__(
            None, bigdl_type, criterion, size_average)


class CrossEntropyCriterion(Criterion):
    """
    This criterion combines LogSoftMax and ClassNLLCriterion in one single class.


    :param weights: A tensor assigning weight to each of the classes


    >>> np.random.seed(123)
    >>> weights = np.random.uniform(0, 1, (2,)).astype("float32")
    >>> cec = CrossEntropyCriterion(weights)
    creating: createCrossEntropyCriterion
    >>> cec = CrossEntropyCriterion()
    creating: createCrossEntropyCriterion
    """

    def __init__(self,
                 weights=None,
                 size_average=True,
                 bigdl_type="float"):
        super(CrossEntropyCriterion, self).__init__(None, bigdl_type,
                                                    JTensor.from_ndarray(
                                                        weights),
                                                    size_average)


class BCECriterion(Criterion):
    '''
    Creates a criterion that measures the Binary Cross Entropy
    between the target and the output


    :param weights: weights for each class
    :param sizeAverage: whether to average the loss or not


    >>> np.random.seed(123)
    >>> weights = np.random.uniform(0, 1, (2,)).astype("float32")
    >>> bCECriterion = BCECriterion(weights)
    creating: createBCECriterion
    >>> bCECriterion = BCECriterion()
    creating: createBCECriterion
    '''

    def __init__(self,
                 weights=None,
                 size_average=True,
                 bigdl_type="float"):
        super(BCECriterion, self).__init__(None, bigdl_type,
                                           JTensor.from_ndarray(weights),
                                           size_average)


class MultiLabelSoftMarginCriterion(Criterion):
    '''
    A MultiLabel multiclass criterion based on sigmoid:
    the loss is:
```
     l(x,y) = - sum_i y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i])
```
    where p[i] = exp(x[i]) / (1 + exp(x[i]))
    and with weights:
```
     l(x,y) = - sum_i weights[i] (y[i] * log(p[i]) + (1 - y[i]) * log (1 - p[i]))
```

    >>> np.random.seed(123)
    >>> weights = np.random.uniform(0, 1, (2,)).astype("float32")
    >>> multiLabelSoftMarginCriterion = MultiLabelSoftMarginCriterion(weights)
    creating: createMultiLabelSoftMarginCriterion
    >>> multiLabelSoftMarginCriterion = MultiLabelSoftMarginCriterion()
    creating: createMultiLabelSoftMarginCriterion
    '''

    def __init__(self,
                 weights=None,
                 size_average=True,
                 bigdl_type="float"):
        super(MultiLabelSoftMarginCriterion, self).__init__(None, bigdl_type,
                                                            JTensor.from_ndarray(weights),
                                                            size_average)


class MultiMarginCriterion(Criterion):
    '''
    Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss)
    between input x and output y (which is a target class index).


    :param p:
    :param weights:
    :param margin:
    :param size_average:


    >>> np.random.seed(123)
    >>> weights = np.random.uniform(0, 1, (2,)).astype("float32")
    >>> multiMarginCriterion = MultiMarginCriterion(1,weights)
    creating: createMultiMarginCriterion
    >>> multiMarginCriterion = MultiMarginCriterion()
    creating: createMultiMarginCriterion
    '''

    def __init__(self,
                 p=1,
                 weights=None,
                 margin=1.0,
                 size_average=True,
                 bigdl_type="float"):
        super(MultiMarginCriterion, self).__init__(None, bigdl_type,
                                                   p,
                                                   JTensor.from_ndarray(weights),
                                                   margin,
                                                   size_average)


class SoftMarginCriterion(Criterion):
    """
    Creates a criterion that optimizes a two-class classification logistic loss
    between input x (a Tensor of dimension 1) and output y (which is a tensor
    containing either 1s or -1s).

```
           loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x:nElement()
```

    :param sizeaverage: The normalization by the number of elements in the inputcan be disabled by setting


    >>> softMarginCriterion = SoftMarginCriterion(False)
    creating: createSoftMarginCriterion
    >>> softMarginCriterion = SoftMarginCriterion()
    creating: createSoftMarginCriterion
    """

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(SoftMarginCriterion, self).__init__(None, bigdl_type, size_average)


class DiceCoefficientCriterion(Criterion):

    '''
    The Dice-Coefficient criterion
    input: Tensor,target: Tensor

```
    return:      2 * (input intersection target)
            1 - ----------------------------------
                    input union target
```

    >>> diceCoefficientCriterion = DiceCoefficientCriterion(size_average = True, epsilon = 1.0)
    creating: createDiceCoefficientCriterion
    '''

    def __init__(self,
                 size_average,
                 epsilon,
                 bigdl_type="float"):
        super(DiceCoefficientCriterion, self).__init__(None, bigdl_type,
                                                       size_average,
                                                       epsilon)

class L1Cost(Criterion):

    '''
    compute L1 norm for input, and sign of input

    >>> l1Cost = L1Cost()
    creating: createL1Cost
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(L1Cost, self).__init__(None, bigdl_type)

def _test():
    import doctest
    from pyspark import SparkContext
    from bigdl.nn import criterion
    from bigdl.util.common import init_engine
    from bigdl.util.common import create_spark_conf
    globs = criterion.__dict__.copy()
    sc = SparkContext(master="local[4]", appName="test criterion",
                      conf=create_spark_conf())
    globs['sc'] = sc
    init_engine()

    (failure_count, test_count) = doctest.testmod(globs=globs,
                                                  optionflags=doctest.ELLIPSIS)
    if failure_count:
        exit(-1)


if __name__ == "__main__":
    _test()
