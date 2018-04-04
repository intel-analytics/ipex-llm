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

from bigdl.nn.keras.layer import *

if sys.version >= '3':
    long = int
    unicode = str


class ZooKerasCreator(JavaValue):
    def jvm_class_constructor(self):
        name = "createZooKeras" + self.__class__.__name__
        print("creating: " + name)
        return name


class ZooKerasLayer(ZooKerasCreator, KerasLayer):
    pass


class Input(ZooKerasCreator, Node):
    """
    Used to instantiate an input node.

    # Arguments
    shape: A shape tuple, not including batch.
    name: String to set the name of the input node. If not specified, its name will by default to be a generated string.

    >>> input = Input(name="input1", shape=(3, 5))
    creating: createZooKerasInput
    """
    def __init__(self, shape=None, name=None, bigdl_type="float"):
        super(Input, self).__init__(None, bigdl_type,
                                    name,
                                    list(shape) if shape else None)


class InputLayer(ZooKerasLayer):
    """
    Used as an entry point into a model.

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the input layer. If not specified, its name will by default to be a generated string.

    >>> inputlayer = InputLayer(input_shape=(3, 5), name="input1")
    creating: createZooKerasInputLayer
    """
    def __init__(self, input_shape=None, **kwargs):
        super(InputLayer, self).__init__(None,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)


class Dense(ZooKerasLayer):
    """
    A densely-connected NN layer.
    The most common input is 2D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: The size of output dimension.
    init: String representation of the initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representation of the activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear). Default is True.
    input_dim: Dimensionality of the input for 2D input. For nD input, you can alternatively specify
               'input_shape' when using this layer as the first layer.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer. If not specified, its name will by default to be a generated string.

    >>> dense = Dense(10, input_dim=8, name="dense1")
    creating: createZooKerasDense
    """
    def __init__(self, output_dim, init="glorot_uniform", activation=None,
                 W_regularizer=None, b_regularizer=None,
                 bias=True, input_dim=None, input_shape=None, **kwargs):
        if input_dim:
            input_shape = (input_dim, )
        super(Dense, self).__init__(None,
                                    output_dim,
                                    init,
                                    activation,
                                    W_regularizer,
                                    b_regularizer,
                                    bias,
                                    list(input_shape) if input_shape else None,
                                    **kwargs)
