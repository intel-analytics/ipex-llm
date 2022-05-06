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


class ConvLSTM2D(ZooKerasLayer):
    """
    Convolutional LSTM.
    The convolution kernel for this layer is a square kernel with equal strides 'subsample'.
    The input of this layer should be 5D, i.e. (samples, time, channels, rows, cols) and
    dim_ordering='th' (Channel First) is expected.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
            Should be equal to nb_row as for a square kernel.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is 'tanh'.
    inner_activation: String representation of the activation function for inner cells.
                      Default is 'hard_sigmoid'.
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    subsample: Tuple of length 2. Factor by which to subsample output.
               Also called strides elsewhere.
               Only support subsample[0] equal to subsample[1] for now. Default is (1, 1).
    border_mode: One of "same" or "valid". Also called padding elsewhere. Default is "valid".
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    U_regularizer: An instance of [[Regularizer]], applied the recurrent weights matrices.
                   Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    return_sequences: Whether to return the full sequence or only return the last output
                      in the output sequence. Default is False.
    go_backwards: Whether the input sequence will be processed backwards. Default is False.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> convlstm2d = ConvLSTM2D(24, 3, 3, input_shape=(4, 32, 32, 32))
    creating: createZooKerasConvLSTM2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, activation="tanh",
                 inner_activation="hard_sigmoid", dim_ordering="th", border_mode="valid",
                 subsample=(1, 1), W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 return_sequences=False, go_backwards=False, input_shape=None, **kwargs):
        if nb_row != nb_col:
            invalidInputError(False, "For ConvLSTM2D, only square kernel is supported for now")
        if border_mode != "same" and border_mode != "valid":
            invalidInputError(False,
                              "For ConvLSTM2D, only support border_mode as 'same' and 'valid'")
        if subsample[0] != subsample[1]:
            invalidInputError(False, "For ConvLSTM2D, only equal strides is supported for now")
        super(ConvLSTM2D, self).__init__(None,
                                         nb_filter,
                                         nb_row,
                                         activation,
                                         inner_activation,
                                         dim_ordering,
                                         subsample[0],
                                         border_mode,
                                         W_regularizer,
                                         U_regularizer,
                                         b_regularizer,
                                         return_sequences,
                                         go_backwards,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)


class ConvLSTM3D(ZooKerasLayer):
    """
    Convolutional LSTM for 3D input.
    The convolution kernel for this layer is a cubic kernel with equal strides for all dimensions.
    The input of this layer should be 6D, i.e. (samples, time, channels, dim1, dim2, dim3),
    and 'CHANNEL_FIRST' (dimOrdering='th') is expected.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_kernel: Length of the first, second and third dimensions in the convolution kernel.
               Cubic kernel.
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    border_mode: Only 'same' is supported for now.
    subsample: Tuple of length 3. Factor by which to subsample output.
               Also called strides elsewhere. Default is (1, 1, 1).
               Only support subsample[0] equal to subsample[1] equal to subsample[2] for now.
    border_mode: One of "same" or "valid". Also called padding elsewhere. Default is "valid".
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    U_regularizer: An instance of [[Regularizer]], applied the recurrent weights matrices.
                   Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    return_sequences: Whether to return the full sequence or only return the last output
                      in the output sequence. Default is False.
    go_backwards: Whether the input sequence will be processed backwards. Default is False.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> convlstm3d = ConvLSTM3D(10, 4, input_shape=(8, 4, 10, 32, 32))
    creating: createZooKerasConvLSTM3D
    """
    def __init__(self, nb_filter, nb_kernel, dim_ordering="th", border_mode="valid",
                 subsample=(1, 1, 1), W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 return_sequences=False, go_backwards=False, input_shape=None, **kwargs):
        if dim_ordering != "th":
            invalidInputError(False, "For ConvLSTM3D, only dim_ordering='th' is supported for now")
        if border_mode != "same" and border_mode != "valid":
            invalidInputError(False,
                              "For ConvLSTM3D, only support border_mode as 'same' and 'valid'")
        if subsample[0] != subsample[1] or subsample[1] != subsample[2]:
            invalidInputError(False, "For ConvLSTM3D, only equal strides is supported for now")
        super(ConvLSTM3D, self).__init__(None,
                                         nb_filter,
                                         nb_kernel,
                                         subsample[0],
                                         border_mode,
                                         W_regularizer,
                                         U_regularizer,
                                         b_regularizer,
                                         return_sequences,
                                         go_backwards,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)
