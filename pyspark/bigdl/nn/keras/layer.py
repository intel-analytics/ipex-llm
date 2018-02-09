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

from bigdl.nn.layer import Layer, Container
from bigdl.util.common import callBigDlFunc, JTensor

if sys.version >= '3':
    long = int
    unicode = str


class KerasLayer(Layer):
    def jvm_class_constructor(self):
        name = "createKeras" + self.__class__.__name__
        print("creating: " + name)
        return name


class Sequential(Container):
    """
    >>> sequential = Sequential()
    creating: createSequential
    """
    def __init__(self, bigdl_type="float"):
        super(Sequential, self).__init__(None, bigdl_type, True)


class InputLayer(KerasLayer):
    """
    >>> inputLayer = InputLayer(input_shape=(3, 5))
    creating: createKerasInputLayer
    """
    def __init__(self, input_shape=None, bigdl_type="float"):
        super(InputLayer, self).__init__(None, bigdl_type,
                                         list(input_shape) if input_shape else None)


class Dense(KerasLayer):
    """
    A densely-connected NN layer.
    When you use this layer as the first layer of a model, you need to provide the argument
    inputShape (a Single Shape, does not include the batch dimension).
    The most common input is 2D.

    # Arguments
    output_dim: The size of output dimension.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear). Default is True.

    >>> dense = Dense(10, input_shape=(3, 4))
    creating: createKerasDense
    """
    def __init__(self, output_dim, init='glorot_uniform', activation=None,
                 W_regularizer=None, b_regularizer=None,
                 bias=True, input_shape=None, bigdl_type="float"):
        super(Dense, self).__init__(None, bigdl_type,
                                    output_dim,
                                    init,
                                    activation,
                                    W_regularizer,
                                    b_regularizer,
                                    bias,
                                    list(input_shape) if input_shape else None)


class Embedding(KerasLayer):
    """
    >>> embedding = Embedding(1000, 32, input_shape=(10, ))
    creating: createKerasEmbedding
    """
    def __init__(self, input_dim, output_dim, init='uniform',
                 W_regularizer=None, input_shape=None, bigdl_type="float"):
        super(Embedding, self).__init__(None, bigdl_type,
                                        input_dim,
                                        output_dim,
                                        init,
                                        W_regularizer,
                                        list(input_shape) if input_shape else None)


class BatchNormalization(KerasLayer):
    """
    >>> batchNormalization = BatchNormalization(input_shape=(3, 12, 12))
    creating: createKerasBatchNormalization
    """
    def __init__(self, epsilon=0.001, momentum=0.99, beta_init='zero', gamma_init='one',
                 dim_ordering="th", input_shape=None, bigdl_type="float"):
        super(BatchNormalization, self).__init__(None, bigdl_type,
                                                 epsilon,
                                                 momentum,
                                                 beta_init,
                                                 gamma_init,
                                                 dim_ordering,
                                                 list(input_shape) if input_shape else None)

    def set_running_mean(self, running_mean):
        callBigDlFunc(self.bigdl_type, "setKerasRunningMean",
                      self.value, JTensor.from_ndarray(running_mean))
        return self

    def set_running_std(self, running_std):
        callBigDlFunc(self.bigdl_type, "setKerasRunningStd",
                      self.value, JTensor.from_ndarray(running_std))
        return self

    def get_running_mean(self):
        return callBigDlFunc(self.bigdl_type, "getKerasRunningMean",
                             self.value).to_ndarray()

    def get_running_std(self):
        return callBigDlFunc(self.bigdl_type, "getKerasRunningStd",
                      self.value).to_ndarray()


class Merge(KerasLayer):
    """
    >>> l1 = InputLayer(input_shape=(3, 5))
    creating: createKerasInputLayer
    >>> l2 = InputLayer(input_shape=(3, 5))
    creating: createKerasInputLayer
    >>> merge = Merge(layers=[l1, l2], mode='sum')
    creating: createKerasMerge
    """
    def __init__(self, layers=None, mode='sum', concat_axis=-1,
                 input_shape=None, bigdl_type="float"):
        super(Merge, self).__init__(None, bigdl_type,
                                    list(layers) if layers else None,
                                    mode,
                                    concat_axis,
                                    input_shape)
