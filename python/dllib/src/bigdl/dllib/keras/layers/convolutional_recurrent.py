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

from ..engine.topology import ZooKerasLayer

if sys.version >= '3':
    long = int
    unicode = str


class ConvLSTM2D(ZooKerasLayer):
    """
    Convolutional LSTM.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'same'.
    The convolution kernel for this layer is a square kernel with equal strides 'subsample'.
    The input of this layer should be 5D.

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
    border_mode: Only 'same' is supported for now.
    subsample: Tuple of length 2. Factor by which to subsample output.
               Also called strides elsewhere.
               Only support subsample[0] equal to subsample[1] for now. Default is (1, 1).
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
                 inner_activation="hard_sigmoid", dim_ordering="th", border_mode="same",
                 subsample=(1, 1), W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 return_sequences=False, go_backwards=False, input_shape=None, **kwargs):
        if nb_row != nb_col:
            raise ValueError("For ConvLSTM2D, only square kernel is supported for now")
        if border_mode != "same":
            raise ValueError("For ConvLSTM2D, only border_mode='same' is supported for now")
        if subsample[0] != subsample[1]:
            raise ValueError("For ConvLSTM2D, only equal strides is supported for now")
        super(ConvLSTM2D, self).__init__(None,
                                         nb_filter,
                                         nb_row,
                                         activation,
                                         inner_activation,
                                         dim_ordering,
                                         subsample[0],
                                         W_regularizer,
                                         U_regularizer,
                                         b_regularizer,
                                         return_sequences,
                                         go_backwards,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)
