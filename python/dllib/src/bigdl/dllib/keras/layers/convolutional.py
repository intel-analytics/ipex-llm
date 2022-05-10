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

from ..engine.topology import ZooKerasLayer
from bigdl.dllib.utils.log4Error import *

if sys.version >= '3':
    long = int
    unicode = str


class Convolution1D(ZooKerasLayer):
    """
    Applies convolution operator for filtering neighborhoods of 1D inputs.
    You can also use Conv1D as an alias of this layer.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    filter_length: The extension (spatial or temporal) of each filter.
    init: String representation of the initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    limits: Optional. Limit value for initialization method.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample_length: Factor by which to subsample output. Int. Default is 1.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> conv1d = Convolution1D(12, 4, input_shape=(3, 16))
    creating: createZooKerasConvolution1D
    """
    def __init__(self, nb_filter, filter_length, init="glorot_uniform", limits=None,
                 activation=None, border_mode="valid", subsample_length=1,
                 W_regularizer=None, b_regularizer=None, bias=True,
                 input_shape=None, **kwargs):
        super(Convolution1D, self).__init__(None,
                                            nb_filter,
                                            filter_length,
                                            init,
                                            list(limits) if limits else None,
                                            activation,
                                            border_mode,
                                            subsample_length,
                                            W_regularizer,
                                            b_regularizer,
                                            bias,
                                            list(input_shape) if input_shape else None,
                                            **kwargs)


class AtrousConvolution1D(ZooKerasLayer):
    """
    Applies an atrous convolution operator for filtering neighborhoods of 1D inputs.
    A.k.a dilated convolution or convolution with holes.
    Border mode currently supported for this layer is 'valid'.
    Bias will be included in this layer.
    You can also use AtrousConv1D as an alias of this layer.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    filter_length: The extension (spatial or temporal) of each filter.
    init: String representation of the initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    border_mode: Only 'valid' is supported for now.
    subsample_length: Factor by which to subsample output. Int. Default is 1.
    atrous_rate: Factor for kernel dilation. Also called filter_dilation elsewhere.
                 Int. Default is 1.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Only 'True' is supported for now.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> atrousconv1d = AtrousConvolution1D(8, 3, input_shape=(3, 12))
    creating: createZooKerasAtrousConvolution1D
    """
    def __init__(self, nb_filter, filter_length, init="glorot_uniform", activation=None,
                 border_mode="valid", subsample_length=1, atrous_rate=1, W_regularizer=None,
                 b_regularizer=None, bias=True, input_shape=None, **kwargs):
        if border_mode != "valid":
            invalidInputError(False, "For AtrousConvolution1D, "
                                     "only border_mode='valid' is supported for now")
        if not bias:
            invalidInputError(False, "For AtrousConvolution1D,"
                                     " only bias=True is supported for now")
        super(AtrousConvolution1D, self).__init__(None,
                                                  nb_filter,
                                                  filter_length,
                                                  init,
                                                  activation,
                                                  subsample_length,
                                                  atrous_rate,
                                                  W_regularizer,
                                                  b_regularizer,
                                                  list(input_shape) if input_shape else None,
                                                  **kwargs)


class Convolution2D(ZooKerasLayer):
    """
    Applies a 2D convolution over an input image composed of several input planes.
    You can also use Conv2D as an alias of this layer.
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
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> conv2d = Convolution2D(32, 3, 3, input_shape=(3, 128, 128), name="convolution2d_1")
    creating: createZooKerasConvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col,
                 init="glorot_uniform", activation=None,
                 border_mode="valid", subsample=(1, 1), dim_ordering="th",
                 W_regularizer=None, b_regularizer=None, bias=True,
                 input_shape=None, pads=None,  **kwargs):
        super(Convolution2D, self).__init__(None,
                                            nb_filter,
                                            nb_row,
                                            nb_col,
                                            init,
                                            activation,
                                            border_mode,
                                            subsample,
                                            dim_ordering,
                                            W_regularizer,
                                            b_regularizer,
                                            bias,
                                            list(input_shape) if input_shape else None,
                                            pads,
                                            **kwargs)


class Deconvolution2D(ZooKerasLayer):
    """
    Transposed convolution operator for filtering windows of 2D inputs.
    The need for transposed convolutions generally arises from the desire to use a transformation
    going in the opposite direction of a normal convolution, i.e., from something that has
    the shape of the output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with said convolution.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    You can also use Deconv2D as an alias of this layer.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    e.g. input_shape=(3, 128, 128) for 128x128 RGB pictures.

    # Arguments
    nb_filter: Number of transposed convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
    output_shape: Output shape of the transposed convolution operation. Tuple of int.
    init: String representation of the initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    border_mode: Only 'valid' is supported for now.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> deconv2d = Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14), input_shape=(3, 12, 12))
    creating: createZooKerasDeconvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, output_shape, init="glorot_uniform",
                 activation=None, border_mode="valid", subsample=(1, 1), dim_ordering="th",
                 W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, **kwargs):
        if border_mode != "valid":
            invalidInputError(False,
                              "For Deconvolution2D, only border_mode='valid' is supported for now")
        super(Deconvolution2D, self).__init__(None,
                                              nb_filter,
                                              nb_row,
                                              nb_col,
                                              init,
                                              activation,
                                              subsample,
                                              dim_ordering,
                                              W_regularizer,
                                              b_regularizer,
                                              bias,
                                              list(input_shape) if input_shape else None,
                                              **kwargs)


class AtrousConvolution2D(ZooKerasLayer):
    """
    Applies an atrous Convolution operator for filtering windows of 2D inputs.
    A.k.a dilated convolution or convolution with holes.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    Bias will be included in this layer.
    You can also use AtrousConv2D as an alias of this layer.
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
    border_mode: Only 'valid' is supported for now.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    atrous_rate: Int tuple of length 2. Factor for kernel dilation.
                 Also called filter_dilation elsewhere. Default is (1, 1).
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Only 'True' is supported for now.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> atrousconv2d = AtrousConvolution2D(12, 4, 3, input_shape=(3, 64, 64))
    creating: createZooKerasAtrousConvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, init="glorot_uniform",
                 activation=None, border_mode="valid", subsample=(1, 1),
                 atrous_rate=(1, 1), dim_ordering="th", W_regularizer=None,
                 b_regularizer=None, bias=True, input_shape=None, **kwargs):
        if border_mode != "valid":
            invalidInputError(False, "For AtrousConvolution2D, "
                                     "only border_mode='valid' is supported for now")
        if not bias:
            invalidInputError(False,
                              "For AtrousConvolution2D, only bias=True is supported for now")
        super(AtrousConvolution2D, self).__init__(None,
                                                  nb_filter,
                                                  nb_row,
                                                  nb_col,
                                                  init,
                                                  activation,
                                                  subsample,
                                                  atrous_rate,
                                                  dim_ordering,
                                                  W_regularizer,
                                                  b_regularizer,
                                                  list(input_shape) if input_shape else None,
                                                  **kwargs)


class SeparableConvolution2D(ZooKerasLayer):
    """
    Applies separable convolution operator for 2D inputs.
    Separable convolutions consist in first performing a depthwise spatial convolution (which acts
    on each input channel separately) followed by a pointwise convolution which mixes together the
    resulting output channels. The depth_multiplier argument controls how many output channels are
    generated per input channel in the depthwise step.
    You can also use SeparableConv2D as an alias of this layer.
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
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample: Int tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    depth_multiplier: How many output channel to use per input channel for the depthwise
                      convolution step. Int. Default is 1.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    depthwise_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                           applied to the depthwise weights matrices. Default is None.
    pointwise_regularizer: An instance of [[Regularizer]], applied to the pointwise weights
                           matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> separableconv2d = SeparableConvolution2D(12, 3, 4, input_shape=(3, 32, 32))
    creating: createZooKerasSeparableConvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, init="glorot_uniform",
                 activation=None, border_mode="valid", subsample=(1, 1), depth_multiplier=1,
                 dim_ordering="th", depthwise_regularizer=None, pointwise_regularizer=None,
                 b_regularizer=None, bias=True, input_shape=None, **kwargs):
        super(SeparableConvolution2D, self).__init__(None,
                                                     nb_filter,
                                                     nb_row,
                                                     nb_col,
                                                     init,
                                                     activation,
                                                     border_mode,
                                                     subsample,
                                                     depth_multiplier,
                                                     dim_ordering,
                                                     depthwise_regularizer,
                                                     pointwise_regularizer,
                                                     b_regularizer,
                                                     bias,
                                                     list(input_shape) if input_shape else None,
                                                     **kwargs)


class Convolution3D(ZooKerasLayer):
    """
    Applies convolution operator for filtering windows of three-dimensional inputs.
    You can also use Conv3D as an alias of this layer.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    kernel_dim1: Length of the first dimension in the convolution kernel.
    kernel_dim2: Length of the second dimension in the convolution kernel.
    kernel_dim3: Length of the third dimension in the convolution kernel.
    init: String representation of the initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample: Int tuple of length 3. Factor by which to subsample output.
               Also called strides elsewhere. Default is (1, 1, 1).
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> conv3d = Convolution3D(32, 3, 4, 5, input_shape=(3, 64, 64, 64))
    creating: createZooKerasConvolution3D
    """
    def __init__(self, nb_filter, kernel_dim1, kernel_dim2, kernel_dim3,
                 init="glorot_uniform", activation=None, border_mode="valid",
                 subsample=(1, 1, 1), dim_ordering="th", W_regularizer=None,
                 b_regularizer=None, bias=True, input_shape=None, **kwargs):
        super(Convolution3D, self).__init__(None,
                                            nb_filter,
                                            kernel_dim1,
                                            kernel_dim2,
                                            kernel_dim3,
                                            init,
                                            activation,
                                            border_mode,
                                            subsample,
                                            dim_ordering,
                                            W_regularizer,
                                            b_regularizer,
                                            bias,
                                            list(input_shape) if input_shape else None,
                                            **kwargs)


class UpSampling1D(ZooKerasLayer):
    """
    UpSampling layer for 1D inputs.
    Repeats each temporal step 'length' times along the time axis.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    length: Int. UpSampling factor. Default is 2.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> upsampling1d = UpSampling1D(length=3, input_shape=(3, 12))
    creating: createZooKerasUpSampling1D
    """
    def __init__(self, length=2, input_shape=None, **kwargs):
        super(UpSampling1D, self).__init__(None,
                                           length,
                                           list(input_shape) if input_shape else None,
                                           **kwargs)


class UpSampling2D(ZooKerasLayer):
    """
    UpSampling layer for 2D inputs.
    Repeats the rows and columns of the data by size[0] and size[1] respectively.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    size: Int tuple of length 2. UpSampling factors for rows and columns. Default is (2, 2).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> upsampling2d = UpSampling2D(size=(1, 3), input_shape=(3, 16, 16))
    creating: createZooKerasUpSampling2D
    """
    def __init__(self, size=(2, 2), dim_ordering="th", input_shape=None, **kwargs):
        super(UpSampling2D, self).__init__(None,
                                           size,
                                           dim_ordering,
                                           list(input_shape) if input_shape else None,
                                           **kwargs)


class UpSampling3D(ZooKerasLayer):
    """
    UpSampling layer for 2D inputs.
    Repeats the 1st, 2nd and 3rd dimensions of the data by
    size[0], size[1] and size[2] respectively.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    size: Int tuple of length 3. UpSampling factors for dim1, dim2 and dim3. Default is (2, 2, 2).
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> upsampling3d = UpSampling3D(size=(1, 2, 3), input_shape=(3, 16, 16, 16))
    creating: createZooKerasUpSampling3D
    """
    def __init__(self, size=(2, 2, 2), dim_ordering="th", input_shape=None, **kwargs):
        super(UpSampling3D, self).__init__(None,
                                           size,
                                           dim_ordering,
                                           list(input_shape) if input_shape else None,
                                           **kwargs)


class ZeroPadding1D(ZooKerasLayer):
    """
    Zero-padding layer for 1D input (e.g. temporal sequence).
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    padding: Int or int tuple of length 2.
             If int, how many zeros to add both at the beginning and at the end of
             the padding dimension.
             If tuple of length 2, how many zeros to add in the order '(left_pad, right_pad)'.
             Default is 1.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> zeropadding1d = ZeroPadding1D(padding=2, input_shape=(3, 6))
    creating: createZooKerasZeroPadding1D
    """
    def __init__(self, padding=1, input_shape=None, **kwargs):
        if isinstance(padding, int):
            padding = (padding, padding)
        super(ZeroPadding1D, self).__init__(None,
                                            padding,
                                            list(input_shape) if input_shape else None,
                                            **kwargs)


class ZeroPadding2D(ZooKerasLayer):
    """
    Zero-padding layer for 2D input (e.g. picture).
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    padding: Int tuple of length 2 or length 4.
             If tuple of length 2, how many zeros to add both at the beginning and
             at the end of rows and cols.
             If tuple of length 4, how many zeros to add in the order
             '(top_pad, bottom_pad, left_pad, right_pad)'.
             Default is (1, 1).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> zeropadding2d = ZeroPadding2D(padding=(2, 1), input_shape=(2, 8, 8))
    creating: createZooKerasZeroPadding2D
    """
    def __init__(self, padding=(1, 1), dim_ordering="th", input_shape=None, **kwargs):
        if len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
        super(ZeroPadding2D, self).__init__(None,
                                            padding,
                                            dim_ordering,
                                            list(input_shape) if input_shape else None,
                                            **kwargs)


class ZeroPadding3D(ZooKerasLayer):
    """
    Zero-padding layer for 3D data (spatial or spatio-temporal).
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    padding: Int tuple of length 3. How many zeros to add at the beginning and
             at the end of the 3 padding dimensions.
             Symmetric padding will be applied to each dimension. Default is (1, 1, 1).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> zeropadding3d = ZeroPadding3D(padding=(2, 1, 2), input_shape=(2, 8, 8, 10))
    creating: createZooKerasZeroPadding3D
    """
    def __init__(self, padding=(1, 1, 1), dim_ordering="th", input_shape=None, **kwargs):
        super(ZeroPadding3D, self).__init__(None,
                                            padding,
                                            dim_ordering,
                                            list(input_shape) if input_shape else None,
                                            **kwargs)


class Cropping1D(ZooKerasLayer):
    """
    Cropping layer for 1D input (e.g. temporal sequence).
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    cropping: Int tuple of length 2. How many units should be trimmed off at the beginning and
              end of the cropping dimension. Default is (1, 1).
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> cropping1d = Cropping1D(cropping=(1, 2), input_shape=(8, 8))
    creating: createZooKerasCropping1D
    """
    def __init__(self, cropping=(1, 1), input_shape=None, **kwargs):
        super(Cropping1D, self).__init__(None,
                                         cropping,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)


class Cropping2D(ZooKerasLayer):
    """
    Cropping layer for 2D input (e.g. picture).
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    cropping: Int tuple of tuple of length 2. How many units should be trimmed off
              at the beginning and end of the 2 cropping dimensions (i.e. height and width).
              Default is ((0, 0), (0, 0)).
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> cropping2d = Cropping2D(cropping=((1, 2), (0, 1)), input_shape=(12, 12, 12))
    creating: createZooKerasCropping2D
    """
    def __init__(self, cropping=((0, 0), (0, 0)), dim_ordering="th",
                 input_shape=None, **kwargs):
        super(Cropping2D, self).__init__(None,
                                         cropping[0],
                                         cropping[1],
                                         dim_ordering,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)


class Cropping3D(ZooKerasLayer):
    """
    Cropping layer for 3D data (e.g. spatial or spatio-temporal).
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    cropping: Int tuple of tuple of length 3. How many units should be trimmed off
              at the beginning and end of the 3 cropping dimensions
              (i.e. kernel_dim1, kernel_dim2 and kernel_dim3).
              Default is ((1, 1), (1, 1), (1, 1)).
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> cropping3d = Cropping3D(cropping=((0, 2), (1, 1), (3, 1)), input_shape=(4, 12, 12, 16))
    creating: createZooKerasCropping3D
    """
    def __init__(self, cropping=((1, 1), (1, 1), (1, 1)), dim_ordering="th",
                 input_shape=None, **kwargs):
        super(Cropping3D, self).__init__(None,
                                         cropping[0],
                                         cropping[1],
                                         cropping[2],
                                         dim_ordering,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)


Conv1D = Convolution1D
Conv2D = Convolution2D
Conv3D = Convolution3D
Deconv2D = Deconvolution2D
AtrousConv1D = AtrousConvolution1D
AtrousConv2D = AtrousConvolution2D
SeparableConv2D = SeparableConvolution2D
