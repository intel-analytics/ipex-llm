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


class SimpleRNN(ZooKerasLayer):
    """
    A fully-connected recurrent neural network cell. The output is to be fed back to input.
    The input of this layer should be 3D, i.e. (batch, time steps, input dim).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: Hidden unit size. Dimension of internal projections and final output.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is 'tanh'.
    return_sequences: Whether to return the full sequence or only return the last output
                      in the output sequence. Default is False.
    go_backwards: Whether the input sequence will be processed backwards. Default is False.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    U_regularizer: An instance of [[Regularizer]], applied the recurrent weights matrices.
                   Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> simplernn = SimpleRNN(16, input_shape=(3, 32))
    creating: createZooKerasSimpleRNN
    """
    def __init__(self, output_dim, activation="tanh", return_sequences=False,
                 go_backwards=False, W_regularizer=None, U_regularizer=None,
                 b_regularizer=None, input_shape=None, **kwargs):
        super(SimpleRNN, self).__init__(None,
                                        output_dim,
                                        activation,
                                        return_sequences,
                                        go_backwards,
                                        W_regularizer,
                                        U_regularizer,
                                        b_regularizer,
                                        list(input_shape) if input_shape else None,
                                        **kwargs)


class GRU(ZooKerasLayer):
    """
    Gated Recurrent Unit architecture.
    The input of this layer should be 3D, i.e. (batch, time steps, input dim).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: Hidden unit size. Dimension of internal projections and final output.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is 'tanh'.
    inner_activation: String representation of the activation function for inner cells.
                      Default is 'hard_sigmoid'.
    return_sequences: Whether to return the full sequence or only return the last output
                      in the output sequence. Default is False.
    go_backwards: Whether the input sequence will be processed backwards. Default is False.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    U_regularizer: An instance of [[Regularizer]], applied the recurrent weights matrices.
                   Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> gru = GRU(24, input_shape=(32, 32))
    creating: createZooKerasGRU
    """
    def __init__(self, output_dim, activation="tanh", inner_activation="hard_sigmoid",
                 return_sequences=False, go_backwards=False, W_regularizer=None,
                 U_regularizer=None, b_regularizer=None, input_shape=None, **kwargs):
        super(GRU, self).__init__(None,
                                  output_dim,
                                  activation,
                                  inner_activation,
                                  return_sequences,
                                  go_backwards,
                                  W_regularizer,
                                  U_regularizer,
                                  b_regularizer,
                                  list(input_shape) if input_shape else None,
                                  **kwargs)


class LSTM(ZooKerasLayer):
    """
    Long Short Term Memory unit architecture.
    The input of this layer should be 3D, i.e. (batch, time steps, input dim).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: Hidden unit size. Dimension of internal projections and final output.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is 'tanh'.
    inner_activation: String representation of the activation function for inner cells.
                      Default is 'hard_sigmoid'.
    return_sequences: Whether to return the full sequence or only return the last output
                      in the output sequence. Default is False.
    go_backwards: Whether the input sequence will be processed backwards. Default is False.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    U_regularizer: An instance of [[Regularizer]], applied the recurrent weights matrices.
                   Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> lstm = LSTM(32, input_shape=(8, 16), name="lstm1")
    creating: createZooKerasLSTM
    """
    def __init__(self, output_dim, activation="tanh", inner_activation="hard_sigmoid",
                 return_sequences=False, go_backwards=False, W_regularizer=None,
                 U_regularizer=None, b_regularizer=None, input_shape=None, **kwargs):
        super(LSTM, self).__init__(None,
                                   output_dim,
                                   activation,
                                   inner_activation,
                                   return_sequences,
                                   go_backwards,
                                   W_regularizer,
                                   U_regularizer,
                                   b_regularizer,
                                   list(input_shape) if input_shape else None,
                                   **kwargs)
