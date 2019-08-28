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
import numpy as np
from bigdl.nn.layer import Layer

if sys.version >= '3':
    long = int
    unicode = str


class AveragePool(Layer):
    """
    >>> avgPool = AveragePool([2, 2])
    creating: createAveragePool
    """
    def __init__(self,
        kernel_shape,
        auto_pad="NOTSET",
        ceil_mode=0,
        count_include_pad=0,
        pads=[],
        strides=[],
        bigdl_type="float"):
        super(AveragePool, self).__init__(None, bigdl_type,
            kernel_shape, auto_pad, ceil_mode, count_include_pad, pads, strides)


class BatchNormalization(Layer):
    """
    >>> batch_norm = BatchNormalization(3)
    creating: createBatchNormalization
    """
    def __init__(self, n_output, epsilon=float(1e-05), momentum=float(0.9), bigdl_type="float"):
        super(BatchNormalization, self).__init__(None, bigdl_type,
            n_output, epsilon, momentum)


class Concat(Layer):
    """
    >>> input_dims = 3
    >>> concat = Concat(nInputDims)
    creating: createConcat
    """
    def __init__(self, input_dims, axis=0, bigdl_type="float"):
        super(Concat, self).__init__(None, bigdl_type, input_dims, axis)


class Conv(Layer):
    """
    >>> n_input_plane = 3
    >>> n_output_plane = 7
    >>> kernel_shape = [2, 2]
    >>> weight = np.array([2, 0.5, 1, 3])
    >>> bias = np.array([2, 0.5, 1, 3])
    >>> conv = Conv(n_input_plane, n_output_plane, kernel_shape, weight, bias)
    creating: createConv
    """
    def __init__(self,
        n_input_plane,
        n_output_plane,
        kernel_shape,
        weight,
        bias,
        auto_pad='NOTSET',
        dilations=[],
        group = 1,
        pads=[],
        strides=[],
        bigdl_type="float"):
        super(Conv, self).__init__(None, bigdl_type,
            n_input_plane, n_output_plane, kernel_shape, weight, bias,
            auto_pad, dilations, group, pads, strides)


class Gather(Layer):
    """
    >>> axis = 1
    >>> gather = Gather(axis)
    creating: createGather
    """
    def __init__(self, axis=0, bigdl_type="float"):
        super(Gather, self).__init__(None, bigdl_type, axis)


class Gemm(Layer):
    """
    >>> gemm = Gemm()
    creating: createGemm
    """
    def __init__(self, alpha=1.0, beta=1.0, trans_a=False, trans_b=False, bigdl_type="float"):
        super(Gemm, self).__init__(None, bigdl_type,
           alpha, beta, trans_a, trans_b)


class MaxPool(Layer):
    """
    >>> max_pool = MaxPool([2, 2])
    creating: createMaxPool
    """
    def __init__(self,
        kernel_shape,
        auto_pad="NOTSET",
        ceil_mode = 0,
        dilations=[],
        pads=[],
        storage_order=0,
        strides=[],
        bigdl_type="float"):
        super(MaxPool, self).__init__(None, bigdl_type, kernel_shape,
            auto_pad, ceil_mode, dilations, pads, storage_order, strides)


class Relu(Layer):
    """
    >>> relu = Relu()
    creating: createRelu
    """
    def __init__(self, bigdl_type="float"):
        super(Relu, self).__init__(None, bigdl_type)


class Reshape(Layer):
    """
    >>> reshape = Reshape((2, 2))
    creating: createGemm
    """
    def __init__(self, size, bigdl_type="float"):
        super(Reshape, self).__init__(None, bigdl_type, size)


class Shape(Layer):
    """
    A layer which takes a tensor as input and outputs an 1D tensor containing the shape of the input.

    >>> shape = Shape()
    creating: createShape
    """
    def __init__(self, bigdl_type="float"):
        super(Shape, self).__init__(None, bigdl_type)


class Softmax(Layer):
    """
    >>> softmax = Softmax()
    creating: createSoftmax
    """
    def __init__(self, alpha=float(1.0), beta=float(1.0), trans_a=False, trans_b=False, bigdl_type="float"):
        super(Softmax, self).__init__(None, bigdl_type, alpha, beta, trans_a, trans_b)


class OnnxSum(Layer):
    """
    >>> gemm = OnnxSum(False)
    creating: createOnnxSum
    """
    def __init__(self, inplace=False, bigdl_type="float"):
        super(OnnxSum, self).__init__(None, bigdl_type, inplace)


class Unsqueeze(Layer):
    """
    >>> unsqueeze = Unsqueeze()
    creating: createUnsqueeze
    """
    def __init__(self, axes=[0], numInputDims=1, bigdl_type="float"):
        super(Unsqueeze, self).__init__(None, bigdl_type, axes, numInputDims)