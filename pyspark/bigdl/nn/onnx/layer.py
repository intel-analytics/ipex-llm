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
from bigdl.util.common import JTensor

if sys.version >= '3':
    long = int
    unicode = str


class AveragePool(Layer):
    """
    AveragePool consumes an input tensor X and applies average pooling across the tensor
    according to kernel sizes, stride sizes, and pad lengths. average pooling consisting
    of computing the average on all values of a subset of the input tensor according to
    the kernel size and downsampling the data into the output tensor Y for further processing.

    >>> kernel_shape = (2, 2)
    >>> avgPool = AveragePool(kernel_shape=kernel_shape)
    creating: createAveragePool
    """
    def __init__(self,
        kernel_shape, # required
        auto_pad="NOTSET",
        ceil_mode=0,
        count_include_pad=0,
        pads=None,
        strides=None,
        bigdl_type="float"):
        super(AveragePool, self).__init__(None, bigdl_type,
            kernel_shape, auto_pad, ceil_mode, count_include_pad, pads, strides)


class BatchNormalization(Layer):
    """
    Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
    as described in the paper: https://arxiv.org/abs/1502.03167

    num_features – C from an expected input of size (N, C, H, W)(N,C,H,W)
    eps – a value added to the denominator for numerical stability. Default: 1e-5
    momentum – the value used for the running_mean and running_var computation.
     Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1

    Input:  (N, C, H, W)
    Output: (N, C, H, W) (same shape as input)

    >>> num_features = 3
    >>> batch_norm = BatchNormalization(num_features=num_features)
    creating: createBatchNormalization
    """
    def __init__(self, num_features, epsilon=float(1e-05), momentum=float(0.9), bigdl_type="float"):
        super(BatchNormalization, self).__init__(None, bigdl_type,
            num_features, epsilon, momentum)


class Concat(Layer):
    """
    Concatenate a list of tensors into a single tensor

    input_dims - dimension number of input
    axis - which axis to concat on

    Input: a BigDL table contains multiple tensors
    Output: a concatenated tensor

    >>> input_dims = 3
    >>> concat = Concat(input_dims)
    creating: createConcat
    """
    def __init__(self, input_dims, axis=0, bigdl_type="float"):
        super(Concat, self).__init__(None, bigdl_type, input_dims, axis)


class Constant(Layer):
    def __init__(self, value, bigdl_type="float"):
        super(Constant, self).__init__(None, bigdl_type, JTensor.from_ndarray(value))


class Conv(Layer):
    """
    The convolution operator consumes an input tensor and a filter, and computes the output.

    >>> n_input_plane = 3
    >>> n_output_plane = 7
    >>> kernel_shape = [2, 2]
    >>> weight = np.random.random(84)
    >>> bias = np.random.random(7)
    >>> conv = Conv(n_input_plane, n_output_plane, kernel_shape, weight, bias)
    creating: createConv
    """
    def __init__(self,
        n_input_plane, n_output_plane, kernel_shape, weight, bias,
        auto_pad='NOTSET', dilations=None, group=1, pads=None, strides=None,
        bigdl_type="float"):
        super(Conv, self).__init__(None, bigdl_type,
            n_input_plane, n_output_plane, kernel_shape,
            JTensor.from_ndarray(weight), JTensor.from_ndarray(bias),
            auto_pad, dilations, group,
            pads, strides)


class Gather(Layer):
    """
    Given data tensor of rank r >= 1, and indices tensor of rank q,
    gather entries of the axis dimension of data (by default outer-most one as axis=0) indexed by indices,
    and concatenates them in an output tensor of rank q + (r - 1).

    >>> gather = Gather()
    creating: createGather
    """
    def __init__(self, axis=0, bigdl_type="float"):
        super(Gather, self).__init__(None, bigdl_type, axis)


class Gemm(Layer):
    """
    General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

    A' = transpose(A) if transA else A
    B' = transpose(B) if transB else B

    Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
    input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
    and output tensor Y has shape (M, N). A will be transposed before doing the computation if
    attribute transA is non-zero, same for B and transB.

    >>> gemm = Gemm()
    creating: createGemm
    """
    def __init__(self, alpha=float(1.0), beta=float(1.0), trans_a=0, trans_b=0, bigdl_type="float"):
        super(Gemm, self).__init__(None, bigdl_type, alpha, beta, trans_a, trans_b)


class MaxPool(Layer):
    """
    MaxPool consumes an input tensor X and applies max pooling across the tensor according to kernel sizes,
    stride sizes, and pad lengths. max pooling consisting of computing the max on all values of a subset
    of the input tensor according to the kernel size and downsampling the data into the output tensor Y for
    further processing.

    >>> max_pool = MaxPool([2, 2])
    creating: createMaxPool
    """
    def __init__(self,
        kernel_shape,
        auto_pad="NOTSET",
        ceil_mode=0,
        dilations=None,
        pads=None,
        storage_order=0,
        strides=None,
        bigdl_type="float"):
        super(MaxPool, self).__init__(None, bigdl_type,
            kernel_shape, auto_pad, ceil_mode,
            dilations, pads, storage_order, strides)


class Relu(Layer):
    """
    Relu takes one input data (Tensor) and produces one output data (Tensor) where the
    rectified linear function, y = max(0, x), is applied to the tensor elementwise.

    >>> relu = Relu()
    creating: createRelu
    """
    def __init__(self, bigdl_type="float"):
        super(Relu, self).__init__(None, bigdl_type)


class Reshape(Layer):
    """
    Reshape the input tensor similar to numpy.reshape.

    >>> reshape = Reshape()
    creating: createGemm
    """
    def __init__(self, bigdl_type="float"):
        super(Reshape, self).__init__(None, bigdl_type)


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
    The operator computes the softmax (normalized exponential) values for each layer in the batch
    of the given input. The input is a 2-D tensor (Tensor) of size (batch_size x input_feature_dimensions).
    The output tensor has the same shape and contains the softmax values of the corresponding input.

    >>> softmax = Softmax(1)
    creating: createSoftmax
    """
    def __init__(self, axis=1, bigdl_type="float"):
        super(Softmax, self).__init__(None, bigdl_type, axis)


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
    def __init__(self, axes, num_input_dims, bigdl_type="float"):
        super(Unsqueeze, self).__init__(None, bigdl_type, axes, num_input_dims)
