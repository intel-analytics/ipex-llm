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

from bigdl.keras.ToBigDLHelper import to_bigdl_reg, to_bigdl_init
from bigdl.nn.layer import Layer, Container
from bigdl.util.common import get_activation_by_name


if sys.version >= '3':
    long = int
    unicode = str


class KerasLayer(Layer):
    def jvm_class_constructor(self):
        name = "createKeras" + self.__class__.__name__
        print("creating: " + name)
        return name


class Sequential(Container):
    def __init__(self, bigdl_type="float"):
        super(Sequential, self).__init__(None, bigdl_type, is_keras=True)


class Dense(Layer):
    """Just your regular densely-connected NN layer.

        # Example

        # Arguments
            output_dim: int > 0.
            init: name of initialization function for the weights of the layer
            activation: name of activation function to use
            W_regularizer: instance of regularizer (eg. L1 or L2 regularization),
                applied to the main weights matrix.
            b_regularizer: nstance of regularizer (eg. L1 or L2 regularization),
                applied to bias
            bias: whether to include a bias
                (i.e. make the layer affine rather than linear).
            input_shape: is required when using this layer as the first layer in a model.

        # Input shape
            nD tensor with shape: `(nb_samples, ..., input_dim)`.
            The most common situation would be
            a 2D input with shape `(nb_samples, input_dim)`.

        # Output shape
            nD tensor with shape: `(nb_samples, ..., output_dim)`.
            For instance, for a 2D input with shape `(nb_samples, input_dim)`,
            the output would have shape `(nb_samples, output_dim)`.
        >>> dense = Dense(10, input_shape=(3, 4))
        creating: createXavier
        creating: createKerasDense
        """
    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None,
                 W_regularizer=None, b_regularizer=None,
                 bias=True, input_shape=None, bigdl_type="float"):
        super(Dense, self).__init__(None, bigdl_type,
                                    output_dim,
                                    to_bigdl_init(init),
                                    get_activation_by_name(activation) if activation else None,  # noqa
                                    to_bigdl_reg(W_regularizer),
                                    to_bigdl_reg(b_regularizer),
                                    bias,
                                    list(input_shape) if input_shape else None)


class Embedding(Layer):
    def __init__(self, input_dim, output_dim, init='uniform',
                 W_regularizer=None, input_shape=None, bigdl_type="float"):
        super(Embedding, self).__init__(None, bigdl_type,
                                        input_dim,
                                        output_dim,
                                        to_bigdl_init(init),
                                        to_bigdl_reg(W_regularizer),
                                        list(input_shape) if input_shape else None)