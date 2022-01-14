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

from bigdl.dllib.utils.common import JavaValue
from bigdl.dllib.utils.common import callBigDlFunc
from bigdl.dllib.utils.common import JTensor
from bigdl.dllib.nn.layer import Layer
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
        jinput, input_is_table = Layer.check_input(input)
        jtarget, target_is_table = Layer.check_input(target)
        output = callBigDlFunc(self.bigdl_type,
                               "criterionForward",
                               self.value,
                               jinput,
                               input_is_table,
                               jtarget,
                               target_is_table)
        return output

    def backward(self, input, target):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        Performs a back-propagation step through the criterion, with respect to the given input.

        :param input: ndarray or list of ndarray
        :param target: ndarray or list of ndarray
        :return: ndarray
        """
        jinput, input_is_table = Layer.check_input(input)
        jtarget, target_is_table = Layer.check_input(target)
        output = callBigDlFunc(self.bigdl_type,
                               "criterionBackward",
                               self.value,
                               jinput,
                               input_is_table,
                               jtarget,
                               target_is_table)
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
    The negative log likelihood criterion. It is useful to train a classification problem with n
    classes. If provided, the optional argument weights should be a 1D Tensor assigning weight to
    each of the classes. This is particularly useful when you have an unbalanced training set.

    The input given through a forward() is expected to contain log-probabilities/probabilities of
    each class: input has to be a 1D Tensor of size n. Obtaining log-probabilities/probabilities
    in a neural network is easily achieved by adding a LogSoftMax/SoftMax layer in the last layer
    of your neural network. You may use CrossEntropyCriterion instead, if you prefer not to add an
    extra layer to your network. This criterion expects a class index (1 to the number of class) as
    target when calling forward(input, target) and backward(input, target).

    In the log-probabilities case,
    The loss can be described as:
        loss(x, class) = -x[class]
    or in the case of the weights argument it is specified as follows:
        loss(x, class) = -weights[class] * x[class]
    Due to the behaviour of the backend code, it is necessary to set sizeAverage to false when
    calculating losses in non-batch mode.

    Note that if the target is `-1`, the training process will skip this sample.
    In other will, the forward process will return zero output and the backward process
    will also return zero `gradInput`.

    By default, the losses are averaged over observations for each minibatch. However, if the field
    sizeAverage is set to false, the losses are instead summed for each minibatch.

    In particular, when weights=None, size_average=True and logProbAsInput=False, this is same as
    `sparse_categorical_crossentropy` loss in keras.


    :param weights: weights of each class
    :param size_average: whether to average or not
    :param logProbAsInput: indicating whether to accept log-probabilities or probabilities as input.


    >>> np.random.seed(123)
    >>> weights = np.random.uniform(0, 1, (2,)).astype("float32")
    >>> classNLLCriterion = ClassNLLCriterion(weights, True, True)
    creating: createClassNLLCriterion
    >>> classNLLCriterion = ClassNLLCriterion()
    creating: createClassNLLCriterion
    '''

    def __init__(self,
                 weights=None,
                 size_average=True,
                 logProbAsInput=True,
                 bigdl_type="float"):
        super(ClassNLLCriterion, self).__init__(None, bigdl_type,
                                                JTensor.from_ndarray(weights),
                                                size_average, logProbAsInput)


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


class CosineDistanceCriterion(Criterion):
    """
    Creates a criterion that measures the loss given an input and target,
    Loss = 1 - cos(x, y)


    >>> cosineDistanceCriterion = CosineDistanceCriterion(True)
    creating: createCosineDistanceCriterion
    >>> cosineDistanceCriterion.forward(np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ...                                   np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
    0.07272728
    """

    def __init__(self,
                 size_average=True,
                 bigdl_type="float"):
        super(CosineDistanceCriterion, self).__init__(None, bigdl_type,
                                                      size_average)


class CosineEmbeddingCriterion(Criterion):
    """
    Creates a criterion that measures the loss given an input x = {x1, x2},
    a table of two Tensors, and a Tensor label y with values 1 or -1.


    :param margin: a number from -1 to 1, 0 to 0.5 is suggested


    >>> cosineEmbeddingCriterion = CosineEmbeddingCriterion(1e-5, True)
    creating: createCosineEmbeddingCriterion
    >>> cosineEmbeddingCriterion.forward([np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ...                                   np.array([5.0, 4.0, 3.0, 2.0, 1.0])],
    ...                                 [np.ones(5)])
    0.0
    """

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


class CategoricalCrossEntropy(Criterion):
    """
    This criterion is same with cross entropy criterion, except it takes a one-hot format target
    tensor
    >>> cce = CategoricalCrossEntropy()
    creating: createCategoricalCrossEntropy
    """

    def __init__(self, bigdl_type="float"):
        super(CategoricalCrossEntropy, self).__init__(None, bigdl_type)


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
    >>> input1 = np.array([2.1, -2.2])
    >>> input2 = np.array([-0.55, 0.298])
    >>> input = [input1, input2]
    >>> target = np.array([1.0])
    >>> result = l1HingeEmbeddingCriterion.forward(input, target)
    >>> (result == 5.148)
    True
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

    When margin = 1, size_average = True and squared = False, this is the same as hinge loss in
    keras;
    When margin = 1, size_average = False and squared = True, this is the same as squared_hinge loss
    in keras.

    :param margin: if unspecified, is by default 1.
    :param size_average: size average in a mini-batch
    :param squared: whether to calculate the squared hinge loss


    >>> marginCriterion = MarginCriterion(1e-5, True, False)
    creating: createMarginCriterion
    '''

    def __init__(self,
                 margin=1.0,
                 size_average=True,
                 squared=False,
                 bigdl_type="float"):
        super(MarginCriterion, self).__init__(None, bigdl_type,
                                              margin,
                                              size_average,
                                              squared)


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
    >>> mSECriterion = MSECriterion()
    creating: createMSECriterion
    >>> multiCriterion = multiCriterion.add(mSECriterion)
    >>> multiCriterion = multiCriterion.add(mSECriterion)
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(MultiCriterion, self).__init__(None, bigdl_type)

    def add(self, criterion, weight=1.0):
        self.value.add(criterion.value, weight)
        return self


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
    >>> mSECriterion = MSECriterion()
    creating: createMSECriterion
    >>> parallelCriterion = parallelCriterion.add(mSECriterion)
    >>> parallelCriterion = parallelCriterion.add(mSECriterion)
    '''

    def __init__(self,
                 repeat_target=False,
                 bigdl_type="float"):
        super(ParallelCriterion, self).__init__(None, bigdl_type,
                                                repeat_target)

    def add(self, criterion, weight=1.0):
        self.value.add(criterion.value, weight)
        return self


class KLDCriterion(Criterion):
    '''
    Computes the KL-divergence of the input normal distribution to a standard normal distribution.
    The input has to be a table. The first element of input is the mean of the distribution,
    the second element of input is the log_variance of the distribution. The input distribution is
    assumed to be diagonal.
    >>> KLDCriterion = KLDCriterion(True)
    creating: createKLDCriterion
    '''

    def __init__(self, size_average=True, bigdl_type="float"):
        super(KLDCriterion, self).__init__(None, bigdl_type, size_average)


class GaussianCriterion(Criterion):
    '''
    Computes the log-likelihood of a sample x given a Gaussian distribution p.
    >>> GaussianCriterion = GaussianCriterion()
    creating: createGaussianCriterion
    '''

    def __init__(self, bigdl_type="float"):
        super(GaussianCriterion, self).__init__(None, bigdl_type)


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
    passing real-valued predictions through a softmax to get a probability distribution over
    classes.
    It should be preferred over separate SoftmaxLayer + MultinomialLogisticLossLayer
    as its gradient computation is more numerically stable.

    :param ignoreLabel:   (optional) Specify a label value thatshould be ignored when computing the
     loss.
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


class TimeDistributedMaskCriterion(Criterion):
    '''
    This class is intended to support inputs with 3 or more dimensions.
    Apply Any Provided Criterion to every temporal slice of an input.
    In addition, it supports padding mask.

    eg. if the target is [ [-1, 1, 2, 3, -1], [5, 4, 3, -1, -1] ],
      and set the paddingValue property to -1, then the loss of -1 would not
      be accumulated and the loss is only divided by 6 (ont including the amount of
      -1, in this case, we are only interested in 1, 2, 3, 5, 4, 3)

    :param criterion: embedded criterion
    :param padding_value: padding value


    >>> td = TimeDistributedMaskCriterion(ClassNLLCriterion())
    creating: createClassNLLCriterion
    creating: createTimeDistributedMaskCriterion
    '''

    def __init__(self, criterion, padding_value=0, bigdl_type="float"):
        super(TimeDistributedMaskCriterion, self).__init__(
            None, bigdl_type, criterion, padding_value)


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

    def __init__(self, criterion, size_average=False, dimension=2, bigdl_type="float"):
        super(TimeDistributedCriterion, self).__init__(
            None, bigdl_type, criterion, size_average, dimension)


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

    :param sizeaverage: The normalization by the number of elements in the inputcan be disabled by
     setting


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
    >>> diceCoefficientCriterion = DiceCoefficientCriterion()
    creating: createDiceCoefficientCriterion
    '''

    def __init__(self,
                 size_average=True,
                 epsilon=1.0,
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


class CosineProximityCriterion(Criterion):
    '''
    compute the negative of the mean cosine proximity between predictions and targets.
```
   x'(i) = x(i) / sqrt(max(sum(x(i)^2), 1e-12))
   y'(i) = y(i) / sqrt(max(sum(x(i)^2), 1e-12))
   cosine_proximity(x, y) = sum_i(-1 * x'(i) * y'(i))
```

    >>> cosineProximityCriterion = CosineProximityCriterion()
    creating: createCosineProximityCriterion
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CosineProximityCriterion, self).__init__(None, bigdl_type)


class MeanAbsolutePercentageCriterion(Criterion):
    '''
    This method is same as `mean_absolute_percentage_error` loss in keras.
    It caculates diff = K.abs((y - x) / K.clip(K.abs(y), K.epsilon(), Double.MaxValue))
    and return 100 * K.mean(diff) as output. Here, the x and y can have or not have a batch.
    >>> error = MeanAbsolutePercentageCriterion()
    creating: createMeanAbsolutePercentageCriterion
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(MeanAbsolutePercentageCriterion, self).__init__(None, bigdl_type)


class MeanSquaredLogarithmicCriterion(Criterion):
    '''
    This method is same as `mean_squared_logarithmic_error` loss in keras.
    It calculates: first_log = K.log(K.clip(y, K.epsilon(),  Double.MaxValue) + 1.)
    second_log = K.log(K.clip(x, K.epsilon(),  Double.MaxValue) + 1.)
    and output K.mean(K.square(first_log - second_log)). Here, the x and y can have or not have a
    batch.
    >>> error = MeanSquaredLogarithmicCriterion()
    creating: createMeanSquaredLogarithmicCriterion
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(MeanSquaredLogarithmicCriterion, self).__init__(None, bigdl_type)


class KullbackLeiblerDivergenceCriterion(Criterion):
    '''
    compute Kullback Leibler DivergenceCriterion error for intput and target
    This method is same as `kullback_leibler_divergence` loss in keras. Loss calculated as:
    y_true = K.clip(input, K.epsilon(), 1)
    y_pred = K.clip(target, K.epsilon(), 1)
    and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    >>> error = KullbackLeiblerDivergenceCriterion()
    creating: createKullbackLeiblerDivergenceCriterion
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(KullbackLeiblerDivergenceCriterion, self).__init__(None, bigdl_type)


class PoissonCriterion(Criterion):
    '''
    compute Poisson error for input and target, loss calculated as:
    mean(input - target * K.log(input + K.epsilon()), axis=-1)
    >>> error = PoissonCriterion()
    creating: createPoissonCriterion
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(PoissonCriterion, self).__init__(None, bigdl_type)


class TransformerCriterion(Criterion):
    '''
    The criterion that takes two modules to transform input and target, and take
    one criterion to compute the loss with the transformed input and target.

    This criterion can be used to construct complex criterion. For example, the
    `inputTransformer` and `targetTransformer` can be pre-trained CNN networks,
    and we can use the networks' output to compute the high-level feature
    reconstruction loss, which is commonly used in areas like neural style transfer
    (https://arxiv.org/abs/1508.06576), texture synthesis (https://arxiv.org/abs/1505.07376),
    .etc.

    >>> trans = TransformerCriterion(MSECriterion())
    creating: createMSECriterion
    creating: createTransformerCriterion
    '''

    def __init__(self,
                 criterion,
                 input_transformer=None,
                 target_transformer=None,
                 bigdl_type="float"):
        super(TransformerCriterion, self).__init__(None,
                                                   bigdl_type,
                                                   criterion,
                                                   input_transformer,
                                                   target_transformer)


class DotProductCriterion(Criterion):
    '''
    Compute the dot product of input and target tensor.
    Input and target are required to have the same size.
    :param size_average: whether to average over each observations in the same batch

    >>> dp =DotProductCriterion(False)
    creating: createDotProductCriterion
    '''

    def __init__(self,
                 size_average=False,
                 bigdl_type="float"):
        super(DotProductCriterion, self).__init__(None,
                                                  bigdl_type,
                                                  size_average)


class PGCriterion(Criterion):
    '''
    The Criterion to compute the negative policy gradient given a
    multinomial distribution and the sampled action and reward.

    The input to this criterion should be a 2-D tensor representing
    a batch of multinomial distribution, the target should also be
    a 2-D tensor with the same size of input, representing the sampled
    action and reward/advantage with the index of non-zero element in the vector
    represents the sampled action and the non-zero element itself represents
    the reward. If the action is space is large, you should consider using
    SparseTensor for target.

    The loss computed is simple the standard policy gradient,

      loss = - 1/n * sum(R_{n} dot_product log(P_{n}))

    where R_{n} is the reward vector, and P_{n} is the input distribution.

    :param sizeAverage whether to average over each observations in the same batch

    >>> pg = PGCriterion()
    creating: createPGCriterion
    '''

    def __init__(self,
                 sizeAverage=False,
                 bigdl_type="float"):
        super(PGCriterion, self).__init__(None,
                                          bigdl_type,
                                          sizeAverage)


def _test():
    import doctest
    from pyspark import SparkContext
    from bigdl.dllib.nn import criterion
    from bigdl.dllib.utils.common import init_engine
    from bigdl.dllib.utils.common import create_spark_conf
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
