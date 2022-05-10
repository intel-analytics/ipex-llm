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
from bigdl.dllib.utils.log4Error import *

if sys.version >= '3':
    long = int
    unicode = str


class MaxPooling1D(ZooKerasLayer):
    """
    Applies max pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_length: Size of the region to which max pooling is applied. Integer. Default is 2.
    strides: Factor by which to downscale. 2 will halve the input.
             Default is None, and in this case it will be equal to pool_length..
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> maxpooling1d = MaxPooling1D(3, input_shape=(3, 24))
    creating: createZooKerasMaxPooling1D
    """
    def __init__(self, pool_length=2, stride=None, border_mode="valid",
                 input_shape=None, pad=0, **kwargs):
        if not stride:
            stride = -1
        super(MaxPooling1D, self).__init__(None,
                                           pool_length,
                                           stride,
                                           border_mode,
                                           list(input_shape) if input_shape else None,
                                           pad,
                                           **kwargs)


class AveragePooling1D(ZooKerasLayer):
    """
    Applies average pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_length: Size of the region to which max pooling is applied.
    strides: Factor by which to downscale. 2 will halve the input.
             Default is None, and in this case it will be equal to pool_length..
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> averagepooling1d = AveragePooling1D(input_shape=(3, 24))
    creating: createZooKerasAveragePooling1D
    """
    def __init__(self, pool_length=2, stride=None, border_mode="valid",
                 input_shape=None, **kwargs):
        if not stride:
            stride = -1
        super(AveragePooling1D, self).__init__(None,
                                               pool_length,
                                               stride,
                                               border_mode,
                                               list(input_shape) if input_shape else None,
                                               **kwargs)


class MaxPooling2D(ZooKerasLayer):
    """
    Applies max pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_size: Int tuple of length 2 corresponding to the downscale vertically and horizontally.
               Default is (2, 2), which will halve the image in each dimension.
    strides: Int tuple of length 2. Stride values.
             Default is None, and in this case it will be equal to pool_size.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> maxpooling2d = MaxPooling2D((2, 2), input_shape=(3, 32, 32), name="maxpooling2d_1")
    creating: createZooKerasMaxPooling2D
    """
    def __init__(self, pool_size=(2, 2), strides=None,
                 border_mode="valid", dim_ordering="th",
                 input_shape=None, pads=None, **kwargs):
        super(MaxPooling2D, self).__init__(None,
                                           pool_size,
                                           strides,
                                           border_mode,
                                           dim_ordering,
                                           list(input_shape) if input_shape else None,
                                           pads,
                                           **kwargs)


class AveragePooling2D(ZooKerasLayer):
    """
    Applies average pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_size: Int tuple of length 2 corresponding to the downscale vertically and horizontally.
               Default is (2, 2), which will halve the image in each dimension.
    strides: Int tuple of length 2. Stride values.
             Default is None, and in this case it will be equal to pool_size.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> averagepooling2d = AveragePooling2D((1, 2), input_shape=(2, 28, 32))
    creating: createZooKerasAveragePooling2D
    """
    def __init__(self, pool_size=(2, 2), strides=None, border_mode="valid",
                 dim_ordering="th", input_shape=None, pads=None, count_include_pad=False, **kwargs):
        super(AveragePooling2D, self).__init__(None,
                                               pool_size,
                                               strides,
                                               border_mode,
                                               dim_ordering,
                                               list(input_shape) if input_shape else None,
                                               pads,
                                               count_include_pad,
                                               **kwargs)


class MaxPooling3D(ZooKerasLayer):
    """
    Applies max pooling operation for 3D data (spatial or spatio-temporal).
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_size: Int tuple of length 3. Factors by which to downscale (dim1, dim2, dim3).
               Default is (2, 2, 2), which will halve the image in each dimension.
    strides: Int tuple of length 3. Stride values.
             Default is None, and in this case it will be equal to pool_size.
    border_mode: Only 'valid' is supported for now.
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> maxpooling3d = MaxPooling3D((2, 1, 3), input_shape=(3, 32, 32, 32))
    creating: createZooKerasMaxPooling3D
    """
    def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode="valid",
                 dim_ordering="th", input_shape=None, **kwargs):
        if border_mode != "valid":
            invalidInputError(False,
                              "For MaxPooling3D, only border_mode='valid' is supported for now")
        super(MaxPooling3D, self).__init__(None,
                                           pool_size,
                                           strides,
                                           dim_ordering,
                                           list(input_shape) if input_shape else None,
                                           **kwargs)


class AveragePooling3D(ZooKerasLayer):
    """
    Applies average pooling operation for 3D data (spatial or spatio-temporal).
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    pool_size: Int tuple of length 3. Factors by which to downscale (dim1, dim2, dim3).
               Default is (2, 2, 2), which will halve the image in each dimension.
    strides: Int tuple of length 3. Stride values.
             Default is None, and in this case it will be equal to pool_size.
    border_mode: Only 'valid' is supported for now.
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> averagepooling3d = AveragePooling3D((1, 1, 2), input_shape=(3, 28, 32, 36))
    creating: createZooKerasAveragePooling3D
    """
    def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode="valid",
                 dim_ordering="th", input_shape=None, **kwargs):
        if border_mode != "valid":
            invalidInputError(False,
                              "For AveragePooling3D, only border_mode='valid' is supported for now")
        super(AveragePooling3D, self).__init__(None,
                                               pool_size,
                                               strides,
                                               dim_ordering,
                                               list(input_shape) if input_shape else None,
                                               **kwargs)


class GlobalAveragePooling1D(ZooKerasLayer):
    """
    Applies global average pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> globalaveragepooling1d = GlobalAveragePooling1D(input_shape=(12, 12))
    creating: createZooKerasGlobalAveragePooling1D
    """
    def __init__(self, input_shape=None, **kwargs):
        super(GlobalAveragePooling1D, self).__init__(None,
                                                     list(input_shape) if input_shape else None,
                                                     **kwargs)


class GlobalMaxPooling1D(ZooKerasLayer):
    """
    Applies global max pooling operation for temporal data.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> globalmaxpooling1d = GlobalMaxPooling1D(input_shape=(4, 8))
    creating: createZooKerasGlobalMaxPooling1D
    """
    def __init__(self, input_shape=None, **kwargs):
        super(GlobalMaxPooling1D, self).__init__(None,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)


class GlobalAveragePooling2D(ZooKerasLayer):
    """
    Applies global average pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> globalaveragepooling2d = GlobalAveragePooling2D(input_shape=(4, 32, 32))
    creating: createZooKerasGlobalAveragePooling2D
    """
    def __init__(self, dim_ordering="th", input_shape=None, **kwargs):
        super(GlobalAveragePooling2D, self).__init__(None,
                                                     dim_ordering,
                                                     list(input_shape) if input_shape else None,
                                                     **kwargs)


class GlobalMaxPooling2D(ZooKerasLayer):
    """
    Applies global max pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> globalmaxpooling2d = GlobalMaxPooling2D(input_shape=(4, 32, 32))
    creating: createZooKerasGlobalMaxPooling2D
    """
    def __init__(self, dim_ordering="th", input_shape=None, **kwargs):
        super(GlobalMaxPooling2D, self).__init__(None,
                                                 dim_ordering,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)


class GlobalAveragePooling3D(ZooKerasLayer):
    """
    Applies global average pooling operation for 3D data.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> globalaveragepooling3d = GlobalAveragePooling3D(input_shape=(4, 16, 16, 20))
    creating: createZooKerasGlobalAveragePooling3D
    """
    def __init__(self, dim_ordering="th", input_shape=None, **kwargs):
        super(GlobalAveragePooling3D, self).__init__(None,
                                                     dim_ordering,
                                                     list(input_shape) if input_shape else None,
                                                     **kwargs)


class GlobalMaxPooling3D(ZooKerasLayer):
    """
    Applies global max pooling operation for 3D data.
    Data format currently supported for this layer is dim_ordering='th' (Channel First).
    Border mode currently supported for this layer is 'valid'.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dim_ordering: Format of input data. Only 'th' (Channel First) is supported for now.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> globalmaxpooling3d = GlobalMaxPooling3D(input_shape=(4, 32, 32, 32))
    creating: createZooKerasGlobalMaxPooling3D
    """
    def __init__(self, dim_ordering="th", input_shape=None, **kwargs):
        super(GlobalMaxPooling3D, self).__init__(None,
                                                 dim_ordering,
                                                 list(input_shape) if input_shape else None,
                                                 **kwargs)
