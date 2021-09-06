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


class TimeDistributed(ZooKerasLayer):
    """
    TimeDistributed wrapper.
    Apply a layer to every temporal slice of an input.
    The input should be at least 3D.
    The dimension of index one will be considered as the temporal dimension.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    name: String to specify the name of the wrapper. Default is None.

    # Arguments
    layer: A layer instance.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the wrapper.
          If not specified, its name will by default to be a generated string.

    >>> from zoo.pipeline.api.keras.layers import Dense
    >>> timedistributed = TimeDistributed(Dense(8), input_shape=(10, 12))
    creating: createZooKerasDense
    creating: createZooKerasTimeDistributed
    """
    def __init__(self, layer, input_shape=None, **kwargs):
        super(TimeDistributed, self).__init__(None,
                                              layer,
                                              list(input_shape) if input_shape else None,
                                              **kwargs)


class Bidirectional(ZooKerasLayer):
    """
    Bidirectional wrapper for RNNs.
    Bidirectional currently requires RNNs to return the full sequence, i.e. return_sequences = True.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).
    Example of creating a bidirectional LSTM:
    Bidirectiona(LSTM(12, return_sequences=True), merge_mode="sum", input_shape=(32, 32))

    # Arguments
    layer: An instance of a recurrent layer.
    merge_mode: Mode by which outputs of the forward and backward RNNs will be combined.
                Must be one of: 'sum', 'mul', 'concat', 'ave'. Default is 'concat'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the wrapper.
          If not specified, its name will by default to be a generated string.

    >>> from zoo.pipeline.api.keras.layers import LSTM
    >>> bidiretional = Bidirectional(LSTM(10, return_sequences=True), input_shape=(12, 16))
    creating: createZooKerasLSTM
    creating: createZooKerasBidirectional
    """
    def __init__(self, layer, merge_mode="concat", input_shape=None, **kwargs):
        super(Bidirectional, self).__init__(None,
                                            layer,
                                            merge_mode,
                                            list(input_shape) if input_shape else None,
                                            **kwargs)


class KerasLayerWrapper(ZooKerasLayer):
    """
    Wrap a torch style layer to keras style layer.
    This layer can be built multiple times.
    This layer will return a keras compatible layer

    # Arguments
    torch_layer: a torch style layer.
    input_shape: A shape tuple, not including batch.
    i.e If the input data is (2, 3, 4) and 2 is the batch size, you should input: (3, 4) here.
    >>> from zoo.pipeline.api.keras.layers import KerasLayerWrapper
    >>> from bigdl.nn.layer import Linear
    >>> linear = Linear(100, 10, with_bias=True)
    creating: createLinear
    >>> kerasLayer = KerasLayerWrapper(linear, input_shape=(100, ))
    creating: createZooKerasKerasLayerWrapper
    """
    def __init__(self, torch_layer, input_shape=None, **kwargs):
        super(KerasLayerWrapper, self).__init__(None,
                                                torch_layer,
                                                list(input_shape) if input_shape else None,
                                                **kwargs)
