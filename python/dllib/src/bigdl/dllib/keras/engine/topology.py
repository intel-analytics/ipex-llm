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

from bigdl.nn.keras.topology import KerasModel as BigDLKerasModel
from bigdl.nn.keras.layer import KerasLayer
from bigdl.nn.layer import Node
from bigdl.util.common import JavaValue

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


class ZooKerasModel(ZooKerasCreator, BigDLKerasModel):
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


class Merge(ZooKerasLayer):
    """
    Used to merge a list of inputs into a single output, following some merge mode.
    Merge must have at least two input layers.

    When using this layer as the first layer in a model, you need to provide the argument
    input_shape for input layers (a list of shape tuples, does not include the batch dimension).

    # Arguments
    layers: A list of layer instances. Must be more than one layer.
    mode: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos',
          'dot', 'max', 'sub', 'div', 'min'. Default is 'sum'.
    concat_axis: Int, axis to use when concatenating layers. Only specify this when merge mode is 'concat'.
                 Default is -1, meaning the last axis of the input.
    input_shape: A list of shape tuples, each not including batch.
    name: String to set the name of the layer. If not specified, its name will by default to be a generated string.

    >>> l1 = InputLayer(input_shape=(3, 5))
    creating: createZooKerasInputLayer
    >>> l2 = InputLayer(input_shape=(3, 5))
    creating: createZooKerasInputLayer
    >>> merge = Merge(layers=[l1, l2], mode='sum', name="merge1")
    creating: createZooKerasMerge
    """
    def __init__(self, layers=None, mode="sum", concat_axis=-1,
                 input_shape=None, **kwargs):
        super(Merge, self).__init__(None,
                                    list(layers) if layers else None,
                                    mode,
                                    concat_axis,
                                    input_shape,
                                    **kwargs)


def merge(inputs, mode="sum", concat_axis=-1, name=None):
    """
    Functional merge. Only use this method if you are defining a graph model.
    Used to merge a list of input nodes into a single output node (NOT layers!),
    following some merge mode.

    # Arguments
    inputs: A list of node instances. Must be more than one node.
    mode: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos',
          'dot', 'max', 'sub', 'div', 'min'. Default is 'sum'.
    concat_axis: Int, axis to use when concatenating nodes. Only specify this when merge mode is 'concat'.
                 Default is -1, meaning the last axis of the input.
    name: String to set the name of the merge. If not specified, its name will by default to be a generated string.
    """
    return Merge(mode=mode, concat_axis=concat_axis, name=name)(list(inputs))
