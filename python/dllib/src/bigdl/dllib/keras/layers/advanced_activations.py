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

if sys.version >= '3':
    long = int
    unicode = str


class LeakyReLU(ZooKerasLayer):
    """
    Leaky version of a Rectified Linear Unit.
    It allows a small gradient when the unit is not active:
    f(x) = alpha * x for x < 0,
    f(x) = x for x >= 0.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    alpha: Float >= 0. Negative slope coefficient. Default is 0.3.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> leakyrelu = LeakyReLU(0.02, input_shape=(4, 5))
    creating: createZooKerasLeakyReLU
    """
    def __init__(self, alpha=0.01, input_shape=None, **kwargs):
        super(LeakyReLU, self).__init__(None,
                                        float(alpha),
                                        list(input_shape) if input_shape else None,
                                        **kwargs)


class ELU(ZooKerasLayer):
    """
    Exponential Linear Unit.
    It follows:
    f(x) =  alpha * (exp(x) - 1.) for x < 0,
    f(x) = x for x >= 0.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    alpha: Float, scale for the negative factor. Default is 1.0.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> elu = ELU(1.2, input_shape=(4, 5))
    creating: createZooKerasELU
    """
    def __init__(self, alpha=1.0, input_shape=None, **kwargs):
        super(ELU, self).__init__(None,
                                  float(alpha),
                                  list(input_shape) if input_shape else None,
                                  **kwargs)


class ThresholdedReLU(ZooKerasLayer):
    """
    Thresholded Rectified Linear Unit.
    It follows:
    f(x) = x for x > theta,
    f(x) = 0 otherwise.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    theta: Float >= 0. Threshold location of activation. Default is 1.0.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> thresholdedrelu = ThresholdedReLU(input_shape=(10, 12))
    creating: createZooKerasThresholdedReLU
    """
    def __init__(self, theta=1.0, input_shape=None, **kwargs):
        super(ThresholdedReLU, self).__init__(None,
                                              float(theta),
                                              list(input_shape) if input_shape else None,
                                              **kwargs)


class SReLU(ZooKerasLayer):
    """
    S-shaped Rectified Linear Unit.
    It follows:
    f(x) = t^r + a^r(x - t^r) for x >= t^r
    f(x) = x for t^r > x > t^l,
    f(x) = t^l + a^l(x - t^l) for x <= t^l

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    t_left_init: String representation of the initialization method for the left part intercept.
                 Default is 'zero'.
    a_left_init: String representation of the initialization method for the left part slope.
                 Default is 'glorot_uniform'.
    t_right_init: String representation of the nitialization method for the right part intercept.
                  Default is 'glorot_uniform'.
    a_right_init: String representation of the initialization method for the right part slope.
                  Default is 'one'.
    shared_axes: Int tuple. The axes along which to share learnable parameters for the
                 activation function. Default is None.
                 For example, if the incoming feature maps are from a 2D convolution with output
                 shape (batch, height, width, channels), and you wish to share parameters across
                 space so that each filter only has one set of parameters, set 'shared_axes=(1,2)'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> srelu = SReLU(input_shape=(4, 5))
    creating: createZooKerasSReLU
    """
    def __init__(self, t_left_init="zero", a_left_init="glorot_uniform",
                 t_right_init="glorot_uniform", a_right_init="one",
                 shared_axes=None, input_shape=None, **kwargs):
        super(SReLU, self).__init__(None,
                                    t_left_init,
                                    a_left_init,
                                    t_right_init,
                                    a_right_init,
                                    shared_axes,
                                    list(input_shape) if input_shape else None,
                                    **kwargs)
