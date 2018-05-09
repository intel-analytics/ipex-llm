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


class LocallyConnected1D(ZooKerasLayer):
    """
    Locally-connected layer for 1D inputs which works similarly to the TemporalConvolution
    layer, except that weights are unshared, that is, a different set of filters is applied
    at each different patch of the input.
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Dimensionality of the output.
    filter_length: The extension (spatial or temporal) of each filter.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    border_mode: Only 'valid' is supported for now.
    subsample_length: Factor by which to subsample output. Int. Default is 1.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> locallyconnected1d = LocallyConnected1D(6, 3, input_shape=(8, 12))
    creating: createZooKerasLocallyConnected1D
    """
    def __init__(self, nb_filter, filter_length, activation=None, border_mode="valid",
                 subsample_length=1, W_regularizer=None, b_regularizer=None,
                 bias=True, input_shape=None, **kwargs):
        if border_mode != "valid":
            raise ValueError("For LocallyConnected1D, "
                             "only border_mode='valid' is supported for now")
        super(LocallyConnected1D, self).__init__(None,
                                                 nb_filter,
                                                 filter_length,
                                                 activation,
                                                 subsample_length,
                                                 W_regularizer,
                                                 b_regularizer,
                                                 bias,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)


class LocallyConnected2D(ZooKerasLayer):
    """
    Locally-connected layer for 2D inputs that works similarly to the SpatialConvolution
    layer, except that weights are unshared, that is, a different set of filters is applied
    at each different patch of the input.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of cols in the convolution kernel.
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

    >>> locallyconnected2d = LocallyConnected2D(12, 3, 4, input_shape=(3, 128, 128))
    creating: createZooKerasLocallyConnected2D
    """
    def __init__(self, nb_filter, nb_row, nb_col, activation=None,
                 border_mode="valid", subsample=(1, 1), dim_ordering="th",
                 W_regularizer=None, b_regularizer=None, bias=True,
                 input_shape=None, **kwargs):
        super(LocallyConnected2D, self).__init__(None,
                                                 nb_filter,
                                                 nb_row,
                                                 nb_col,
                                                 activation,
                                                 border_mode,
                                                 subsample,
                                                 dim_ordering,
                                                 W_regularizer,
                                                 b_regularizer,
                                                 bias,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)
