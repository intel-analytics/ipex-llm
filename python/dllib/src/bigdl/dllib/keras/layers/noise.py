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


class GaussianNoise(ZooKerasLayer):
    """
    Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting (you could see it as a form of random data augmentation).
    Gaussian Noise is a natural choice as corruption process for real valued inputs.
    As it is a regularization layer, it is only active at training time.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    sigma: Float, standard deviation of the noise distribution.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> gaussiannoise = GaussianNoise(0.45, input_shape=(3, 4, 5), name="gaussiannoise1")
    creating: createZooKerasGaussianNoise
    """
    def __init__(self, sigma, input_shape=None, **kwargs):
        super(GaussianNoise, self).__init__(None,
                                            float(sigma),
                                            list(input_shape) if input_shape else None,
                                            **kwargs)


class GaussianDropout(ZooKerasLayer):
    """
    Apply multiplicative 1-centered Gaussian noise.
    As it is a regularization layer, it is only active at training time.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Drop probability. Float between 0 and 1.
       The multiplicative noise will have standard deviation 'sqrt(p/(1-p))'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> gaussiandropout = GaussianDropout(0.45, input_shape=(4, 8))
    creating: createZooKerasGaussianDropout
    """
    def __init__(self, p, input_shape=None, **kwargs):
        super(GaussianDropout, self).__init__(None,
                                              float(p),
                                              list(input_shape) if input_shape else None,
                                              **kwargs)
