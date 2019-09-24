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


class Shape(Layer):
    """
    A layer which takes a tensor as input and outputs an 1D tensor containing the shape of the input.

    >>> shape = Shape()
    creating: createShape
    """
    def __init__(self, bigdl_type="float"):
        super(Shape, self).__init__(None, bigdl_type)
