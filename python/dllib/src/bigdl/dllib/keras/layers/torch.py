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

# Layers from Torch wrapped in Keras style.

import sys

from ..engine.topology import ZooKerasLayer

if sys.version >= '3':
    long = int
    unicode = str


class Select(ZooKerasLayer):
    """
    Select an index of the input in the given dim and return the subset part.
    The batch dimension needs to be unchanged.
    The returned tensor has one less dimension: the dimension dim is removed.
    As a result, it is not possible to select() on a 1D tensor.
    For example, if input is: [[1 2 3], [4 5 6]]
    Select(1, 1) will give output [2 5]
    Select(1, -1) will give output [3 6]

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim: The dimension to select. 0-based index. Cannot select the batch dimension.
         -1 means the last dimension of the input.
    index: The index of the dimension to be selected. 0-based index.
           -1 means the last dimension of the input.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the wrapper.
          If not specified, its name will by default to be a generated string.

    >>> select = Select(0, -1, input_shape=(3, 4), name="select1")
    creating: createZooKerasSelect
    """
    def __init__(self, dim, index, input_shape=None, **kwargs):
        super(Select, self).__init__(None,
                                     dim,
                                     index,
                                     list(input_shape) if input_shape else None,
                                     **kwargs)


class Narrow(ZooKerasLayer):
    """
    Narrow the input with the number of dimensions not being reduced.
    The batch dimension needs to be unchanged.
    For example, if input is: [[1 2 3], [4 5 6]]
    Narrow(1, 1, 2) will give output [[2 3], [5 6]]
    Narrow(1, 2, -1) will give output [[3], [6]]

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim: The dimension to narrow. 0-based index. Cannot narrow the batch dimension.
         -1 means the last dimension of the input.
    offset: Non-negative integer. The start index on the given dimension. 0-based index.
    length: The length to narrow. Default is 1.
            Can use a negative length such as -1 in the case where input size is unknown.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> narrow = Narrow(1, 3, input_shape=(5, 6, 7), name="narrow1")
    creating: createZooKerasNarrow
    """
    def __init__(self, dim, offset, length=1, input_shape=None, **kwargs):
        super(Narrow, self).__init__(None,
                                     dim,
                                     offset,
                                     length,
                                     list(input_shape) if input_shape else None,
                                     **kwargs)


class Squeeze(ZooKerasLayer):
    """
    Delete the singleton dimension(s).
    The batch dimension needs to be unchanged.
    For example, if input has size (2, 1, 3, 4, 1):
    Squeeze(1) will give output size (2, 3, 4, 1)
    Squeeze() will give output size (2, 3, 4)

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim: The dimension(s) to squeeze. Can be either int or tuple of int.
         0-based index. Cannot squeeze the batch dimension.
         The selected dimensions must be singleton, i.e. having size 1.
         Default is None, and in this case all the non-batch singleton dimensions will be deleted.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> squeeze1 = Squeeze(1, input_shape=(1, 4, 5))
    creating: createZooKerasSqueeze
    >>> squeeze2 = Squeeze(input_shape=(1, 8, 1, 4))
    creating: createZooKerasSqueeze
    >>> squeeze3 = Squeeze((1, 2), input_shape=(1, 1, 1, 32))
    creating: createZooKerasSqueeze
    """
    def __init__(self, dim=None, input_shape=None, **kwargs):
        if isinstance(dim, int):
            dim = (dim, )
        super(Squeeze, self).__init__(None,
                                      dim,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


class AddConstant(ZooKerasLayer):
    """
    Add a (non-learnable) scalar constant to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    constant: The scalar constant to be added.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> addconstant = AddConstant(1, input_shape=(1, 4, 5))
    creating: createZooKerasAddConstant
    """
    def __init__(self, constant, input_shape=None, **kwargs):
        super(AddConstant, self).__init__(None,
                                          float(constant),
                                          list(input_shape) if input_shape else None,
                                          **kwargs)


class MulConstant(ZooKerasLayer):
    """
    Multiply the input by a (non-learnable) scalar constant.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    constant: The scalar constant to be multiplied.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> mulconstant = MulConstant(2.2, input_shape=(3, 4))
    creating: createZooKerasMulConstant
    """
    def __init__(self, constant, input_shape=None, **kwargs):
        super(MulConstant, self).__init__(None,
                                          float(constant),
                                          list(input_shape) if input_shape else None,
                                          **kwargs)


class LRN2D(ZooKerasLayer):
    """
    Local Response Normalization between different feature maps.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    alpha: Float. The scaling parameter. Default is 0.0001.
    k: Float. A constant.
    beta: Float. The exponent. Default is 0.75.
    n: The number of channels to sum over.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> lrn2d = LRN2D(1e-3, 1.2, 0.4, 4, dim_ordering="tf", input_shape=(4, 5, 6))
    creating: createZooKerasLRN2D
    """
    def __init__(self, alpha=1e-4, k=1.0, beta=0.75, n=5,
                 dim_ordering="th", input_shape=None, **kwargs):
        super(LRN2D, self).__init__(None,
                                    float(alpha),
                                    float(k),
                                    float(beta),
                                    n,
                                    dim_ordering,
                                    list(input_shape) if input_shape else None,
                                    **kwargs)


class ShareConvolution2D(ZooKerasLayer):
    """
    Applies a 2D convolution over an input image composed of several input planes.
    You can also use ShareConv2D as an alias of this layer.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    e.g. input_shape=(3, 128, 128) for 128x128 RGB pictures.

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
    init: String representation of the initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    pad_h: The additional zeros added to the height dimension. Default is 0.
    pad_w: The additional zeros added to the width dimension. Default is 0.
    propagate_back: Whether to propagate gradient back. Default is True.
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> shareconv2d = ShareConvolution2D(32, 3, 4, activation="tanh", input_shape=(3, 128, 128))
    creating: createZooKerasShareConvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, init="glorot_uniform",
                 activation=None, subsample=(1, 1), pad_h=0, pad_w=0, propagate_back=True,
                 dim_ordering="th", W_regularizer=None, b_regularizer=None,
                 bias=True, input_shape=None, **kwargs):
        super(ShareConvolution2D, self).__init__(None,
                                                 nb_filter,
                                                 nb_row,
                                                 nb_col,
                                                 init,
                                                 activation,
                                                 subsample,
                                                 pad_h,
                                                 pad_w,
                                                 propagate_back,
                                                 dim_ordering,
                                                 W_regularizer,
                                                 b_regularizer,
                                                 bias,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)


ShareConv2D = ShareConvolution2D


class CAdd(ZooKerasLayer):
    """
    This layer has a bias with given size.
    The bias will be added element-wise to the input.
    If the element number of the bias matches the input, a simple element-wise addition
    will be done.
    Or the bias will be expanded to the same size of the input.
    The expand means repeat on unmatched singleton dimension (if some unmatched dimension
    isn't a singleton dimension, an error will be raised).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    size: The size of the bias.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is null.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> cadd = CAdd((2, 1), input_shape=(3, ))
    creating: createZooKerasCAdd
    """
    def __init__(self, size, b_regularizer=None, input_shape=None, **kwargs):
        super(CAdd, self).__init__(None,
                                   size,
                                   b_regularizer,
                                   list(input_shape) if input_shape else None,
                                   **kwargs)


class CMul(ZooKerasLayer):
    """
    This layer has a weight with given size.
    The weight will be multiplied element-wise to the input.
    If the element number of the weight matches the input,
    a simple element-wise multiplication will be done.
    Or the bias will be expanded to the same size of the input.
    The expand means repeat on unmatched singleton dimension (if some unmatched dimension isn't
    singleton dimension, an error will be raised).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    size: The size of the bias.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is null.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> cmul = CMul((2, 1), input_shape=(3, ))
    creating: createZooKerasCMul
    """
    def __init__(self, size, W_regularizer=None, input_shape=None, **kwargs):
        super(CMul, self).__init__(None,
                                   size,
                                   W_regularizer,
                                   list(input_shape) if input_shape else None,
                                   **kwargs)


class Exp(ZooKerasLayer):
    """
    Applies element-wise exp to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> exp = Exp(input_shape=(2, 3, 4))
    creating: createZooKerasExp
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Exp, self).__init__(None,
                                  list(input_shape) if input_shape else None,
                                  **kwargs)


class Identity(ZooKerasLayer):
    """
    Identity just return the input to output.
    It's useful in same parallel container to get an origin input.

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> identity = Identity(input_shape=(3, ))
    creating: createZooKerasIdentity
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Identity, self).__init__(None,
                                       list(input_shape) if input_shape else None,
                                       **kwargs)


class Log(ZooKerasLayer):
    """
    Applies a log transformation to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> log = Log(input_shape=(4, 8, 8))
    creating: createZooKerasLog
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Log, self).__init__(None,
                                  list(input_shape) if input_shape else None,
                                  **kwargs)


class Mul(ZooKerasLayer):
    """
    Multiply a single scalar factor to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> mul = Mul(input_shape=(3, 4, 5))
    creating: createZooKerasMul
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Mul, self).__init__(None,
                                  list(input_shape) if input_shape else None,
                                  **kwargs)


class Power(ZooKerasLayer):
    """
    Applies an element-wise power operation with scale and shift to the input.

    f(x) = (shift + scale * x)^power^

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    power: The exponent.
    scale: The scale parameter. Default is 1.
    shift: The shift parameter. Default is 0.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> power = Power(3, input_shape=(3, ))
    creating: createZooKerasPower
    """
    def __init__(self, power, scale=1, shift=0, input_shape=None, **kwargs):
        super(Power, self).__init__(None,
                                    float(power),
                                    float(scale),
                                    float(shift),
                                    list(input_shape) if input_shape else None,
                                    **kwargs)


class Scale(ZooKerasLayer):
    """
    Scale is the combination of CMul and CAdd.

    Computes the element-wise product of the input and weight,
    with the shape of the weight "expand" to match the shape of the input.
    Similarly, perform an expanded bias and perform an element-wise add.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    size: Size of the weight and bias.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> scale = Scale((2, 1), input_shape=(3, ))
    creating: createZooKerasScale
    """
    def __init__(self, size, input_shape=None, **kwargs):
        super(Scale, self).__init__(None,
                                    size,
                                    list(input_shape) if input_shape else None,
                                    **kwargs)


class Sqrt(ZooKerasLayer):
    """
    Applies an element-wise square root operation to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> sqrt = Sqrt(input_shape=(3, ))
    creating: createZooKerasSqrt
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Sqrt, self).__init__(None,
                                   list(input_shape) if input_shape else None,
                                   **kwargs)


class Square(ZooKerasLayer):
    """
    Applies an element-wise square operation to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> square = Square(input_shape=(5, ))
    creating: createZooKerasSquare
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Square, self).__init__(None,
                                     list(input_shape) if input_shape else None,
                                     **kwargs)


class HardShrink(ZooKerasLayer):
    """
    Applies the hard shrinkage function element-wise to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    value: The threshold value. Default is 0.5.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> hardshrink = HardShrink(input_shape=(2, 4, 8))
    creating: createZooKerasHardShrink
    """
    def __init__(self, value=0.5, input_shape=None, **kwargs):
        super(HardShrink, self).__init__(None,
                                         float(value),
                                         list(input_shape) if input_shape else None,
                                         **kwargs)


class HardTanh(ZooKerasLayer):
    """
    Applies the hard tanh function element-wise to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    min_value: The minimum threshold value. Default is -1.
    max_value: The maximum threshold value. Default is 1.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> hardtanh = HardTanh(input_shape=(3, 4))
    creating: createZooKerasHardTanh
    """
    def __init__(self, min_value=-1, max_value=1, input_shape=None, **kwargs):
        super(HardTanh, self).__init__(None,
                                       float(min_value),
                                       float(max_value),
                                       list(input_shape) if input_shape else None,
                                       **kwargs)


class Negative(ZooKerasLayer):
    """
    Computes the negative value of each element of the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> negative = Negative(input_shape=(4, 5, 8))
    creating: createZooKerasNegative
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Negative, self).__init__(None,
                                       list(input_shape) if input_shape else None,
                                       **kwargs)


class PReLU(ZooKerasLayer):
    """
    Applies parametric ReLU, where parameter varies the slope of the negative part.

    Notice: Please don't use weight decay on this.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    n_output_plane: Input map number. Default is 0,
                    which means using PReLU in shared version and has only one parameter.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> prelu = PReLU(input_shape=(3, 4, 8, 8))
    creating: createZooKerasPReLU
    """
    def __init__(self, n_output_plane=0, input_shape=None, **kwargs):
        super(PReLU, self).__init__(None,
                                    n_output_plane,
                                    list(input_shape) if input_shape else None,
                                    **kwargs)


class RReLU(ZooKerasLayer):
    """
    Applies the randomized leaky rectified linear unit element-wise to the input.

    In the training mode, negative inputs are multiplied by a factor drawn
    from a uniform random distribution U(l, u).
    In the evaluation mode, a RReLU behaves like a LeakyReLU with a constant mean
    factor a = (l + u) / 2.
    If l == u, a RReLU essentially becomes a LeakyReLU.
    Regardless of operating in in-place mode a RReLU will internally
    allocate an input-sized noise tensor to store random factors for negative inputs.
    For reference see [Empirical Evaluation of Rectified Activations in Convolutional
    Network](http://arxiv.org/abs/1505.00853).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    lower: Lower boundary of the uniform random distribution. Default is 1.0/8.
    upper: Upper boundary of the uniform random distribution. Default is 1.0/3.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> rrelu = RReLU(input_shape=(3, 4))
    creating: createZooKerasRReLU
    """
    def __init__(self, lower=1.0/8, upper=1.0/3, input_shape=None, **kwargs):
        super(RReLU, self).__init__(None,
                                    float(lower),
                                    float(upper),
                                    list(input_shape) if input_shape else None,
                                    **kwargs)


class SoftShrink(ZooKerasLayer):
    """
    Applies the soft shrinkage function element-wise to the input.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    value: The threshold value. Default is 0.5.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> softshrink = SoftShrink(input_shape=(4, 4, 8, 8))
    creating: createZooKerasSoftShrink
    """
    def __init__(self, value=0.5, input_shape=None, **kwargs):
        super(SoftShrink, self).__init__(None,
                                         float(value),
                                         list(input_shape) if input_shape else None,
                                         **kwargs)


class WithinChannelLRN2D(ZooKerasLayer):
    """
    The local response normalization layer performs a kind of "lateral inhibition"
    by normalizing over local input regions. The local regions extend spatially,
    in separate channels (i.e., they have shape 1 x size x size).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    size: The side length of the square region to sum over. Default is 5.
    alpha: The scaling parameter. Default is 1.0.
    beta: The exponent. Default is 0.75.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> withinchannellrn2d = WithinChannelLRN2D(input_shape=(2, 3, 8, 8))
    creating: createZooKerasWithinChannelLRN2D
    """
    def __init__(self, size=5, alpha=1.0, beta=0.75, input_shape=None, **kwargs):
        super(WithinChannelLRN2D, self).__init__(None,
                                                 size,
                                                 float(alpha),
                                                 float(beta),
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)


class BinaryThreshold(ZooKerasLayer):
    """
    Threshold the input.
    If an input element is smaller than the threshold value,
    it will be replaced by 0; otherwise, it will be replaced by 1.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    value: The threshold value to compare with. Default is 1e-6.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> binarythreshold = BinaryThreshold(input_shape=(2, 3, 4, 5))
    creating: createZooKerasBinaryThreshold
    """
    def __init__(self, value=1e-6, input_shape=None, **kwargs):
        super(BinaryThreshold, self).__init__(None,
                                              float(value),
                                              list(input_shape) if input_shape else None,
                                              **kwargs)


class Threshold(ZooKerasLayer):
    """
    Threshold input Tensor.
    If values in the Tensor smaller than or equal to th, then replace it with v.
    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    # Arguments
    th: The threshold value to compare with. Default is 1e-6.
    v: the value to replace with.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.
    >>> threshold = Threshold(input_shape=(2, 3, 4, 5))
    creating: createZooKerasThreshold
    """
    def __init__(self, th=1e-6, v=0.0, input_shape=None, **kwargs):
        super(Threshold, self).__init__(None,
                                        float(th),
                                        float(v),
                                        list(input_shape) if input_shape else None,
                                        **kwargs)


class GaussianSampler(ZooKerasLayer):
    """
    Takes {mean, log_variance} as input and samples from the Gaussian distribution

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.

    >>> gaussianSampler = GaussianSampler(input_shape=[(3,),(3,)])
    creating: createZooKerasGaussianSampler
    """
    def __init__(self, input_shape=None, **kwargs):
        super(GaussianSampler, self).__init__(None,
                                              list(input_shape) if input_shape else None,
                                              **kwargs)


class ResizeBilinear(ZooKerasLayer):
    """
    Resize the input image with bilinear interpolation. The input image must be a float tensor with
    NHWC or NCHW layout

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_height: output height
    output_width: output width
    align_corner: align corner or not
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> resizeBilinear = ResizeBilinear(10, 20, input_shape=(2, 3, 5, 7))
    creating: createZooKerasResizeBilinear
    """
    def __init__(self, output_height, output_width, align_corner=False,
                 dim_ordering="th", input_shape=None, **kwargs):
        super(ResizeBilinear, self).__init__(None,
                                             output_height,
                                             output_width,
                                             align_corner,
                                             dim_ordering,
                                             list(input_shape) if input_shape else None,
                                             **kwargs)


class SelectTable(ZooKerasLayer):
    """
    Creates a module that takes a list of JTensors as input and outputs the element at index `index`

    # Arguments
    index: the index to be selected. 0-based index
    input_shape: a list of shape tuples, not including batch.

    >>> selectTable = SelectTable(0, input_shape=[[2, 3], [5, 7]])
    creating: createZooKerasSelectTable
    """
    def __init__(self, index, input_shape=None, **kwargs):
        super(SelectTable, self).__init__(None,
                                          index,
                                          list(input_shape) if input_shape else None,
                                          **kwargs)
