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

from bigdl.dllib.utils.common import INTMIN
from ..engine.topology import ZooKerasLayer

if sys.version >= '3':
    long = int
    unicode = str


class Masking(ZooKerasLayer):
    """
    Use a mask value to skip timesteps for a sequence.
    Masks a sequence by using a mask value to skip timesteps.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    mask_value: Float, mask value. For each timestep in the input (the second dimension),
                if all values in the input at that timestep are equal to 'mask_value',
                then the timestep will masked (skipped) in all downstream layers.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> masking = Masking(0.3, input_shape=(6, 8))
    creating: createZooKerasMasking
    """
    def __init__(self, mask_value=0.0, input_shape=None, **kwargs):
        super(Masking, self).__init__(None,
                                      float(mask_value),
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


class Dropout(ZooKerasLayer):
    """
    Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each
    update during training time in order to prevent overfitting.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Fraction of the input units to drop. Float between 0 and 1.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> dropout = Dropout(0.25, input_shape=(2, 3))
    creating: createZooKerasDropout
    """
    def __init__(self, p, input_shape=None, **kwargs):
        super(Dropout, self).__init__(None,
                                      float(p),
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


class SpatialDropout1D(ZooKerasLayer):
    """
    Spatial 1D version of Dropout.
    This version performs the same function as Dropout, however it drops entire 1D feature maps
    instead of individual elements. If adjacent frames within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then regular dropout will not regularize
    the activations and will otherwise just result in an effective learning rate decrease.
    In this case, SpatialDropout1D will help promote independence between feature maps and
    should be used instead.
    The input of this layer should be 3D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Fraction of the input units to drop. Float between 0 and 1.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> spatialdropout1d = SpatialDropout1D(0.4, input_shape=(10, 12))
    creating: createZooKerasSpatialDropout1D
    """
    def __init__(self, p=0.5, input_shape=None, **kwargs):
        super(SpatialDropout1D, self).__init__(None,
                                               float(p),
                                               list(input_shape) if input_shape else None,
                                               **kwargs)


class SpatialDropout2D(ZooKerasLayer):
    """
    Spatial 2D version of Dropout.
    This version performs the same function as Dropout, however it drops entire 2D feature maps
    instead of individual elements. If adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then regular dropout will not regularize
    the activations and will otherwise just result in an effective learning rate decrease.
    In this case, SpatialDropout2D will help promote independence between feature maps and
    should be used instead.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Fraction of the input units to drop. Float between 0 and 1.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> spatialdropout2d = SpatialDropout2D(0.25, input_shape=(5, 12, 12))
    creating: createZooKerasSpatialDropout2D
    """
    def __init__(self, p=0.5, dim_ordering="th", input_shape=None, **kwargs):
        super(SpatialDropout2D, self).__init__(None,
                                               float(p),
                                               dim_ordering,
                                               list(input_shape) if input_shape else None,
                                               **kwargs)


class SpatialDropout3D(ZooKerasLayer):
    """
    Spatial 3D version of Dropout.
    This version performs the same function as Dropout, however it drops entire 3D feature maps
    instead of individual elements. If adjacent voxels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then regular dropout will not regularize
    the activations and will otherwise just result in an effective learning rate decrease.
    In this case, SpatialDropout3D will help promote independence between feature maps and
    should be used instead.
    The input of this layer should be 5D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Fraction of the input units to drop. Float between 0 and 1.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last).
                  Default is 'th'.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> spatialdropout3d = SpatialDropout3D(0.6, input_shape=(4, 12, 12, 16))
    creating: createZooKerasSpatialDropout3D
    """
    def __init__(self, p=0.5, dim_ordering="th", input_shape=None, **kwargs):
        super(SpatialDropout3D, self).__init__(None,
                                               float(p),
                                               dim_ordering,
                                               list(input_shape) if input_shape else None,
                                               **kwargs)


class Activation(ZooKerasLayer):
    """
    Simple activation function to be applied to the output.
    Available activations: 'tanh', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign',
                           'hard_sigmoid', 'linear', 'relu6', 'tanh_shrink', 'softmin',
                           'log_sigmoid' and 'log_softmax'.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    activation: Name of the activation function as string.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> activation = Activation("relu", input_shape=(3, 4))
    creating: createZooKerasActivation
    """
    def __init__(self, activation, input_shape=None, **kwargs):
        super(Activation, self).__init__(None,
                                         activation,
                                         list(input_shape) if input_shape else None,
                                         **kwargs)


class Reshape(ZooKerasLayer):
    """
    Reshapes an output to a certain shape.
    Supports shape inference by allowing one -1 in the target shape.
    For example, if input_shape = (2, 3, 4), target_shape = (3, -1),
    then output_shape will be (3, 8).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    target_shape: A shape tuple. The target shape that you desire to have.
                  Batch dimension should be excluded.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> reshape = Reshape((2, 10), input_shape=(5, 4))
    creating: createZooKerasReshape
    """
    def __init__(self, target_shape, input_shape=None, **kwargs):
        super(Reshape, self).__init__(None,
                                      target_shape,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


class Permute(ZooKerasLayer):
    """
    Permutes the dimensions of the input according to a given pattern.
    Useful for connecting RNNs and convnets together.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    dims: Tuple of int. Permutation pattern, does not include the batch dimension.
          Indexing starts at 1.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> permute = Permute((2, 1, 3), input_shape=(3, 4, 5))
    creating: createZooKerasPermute
    """
    def __init__(self, dims, input_shape=None, **kwargs):
        super(Permute, self).__init__(None,
                                      dims,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


class Flatten(ZooKerasLayer):
    """
    Flattens the input without affecting the batch size.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> flatten = Flatten(input_shape=(3, 10, 2))
    creating: createZooKerasFlatten
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Flatten, self).__init__(None,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


class RepeatVector(ZooKerasLayer):
    """
    Repeats the input n times.
    The input of this layer should be 2D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    n: Repetition factor. Int.
    input_dim: Dimensionality of the input. Alternatively, you can specify 'input_shape'
               when using this layer as the first layer.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> repeatvector = RepeatVector(5, input_shape=(3, ))
    creating: createZooKerasRepeatVector
    """
    def __init__(self, n, input_dim=None, input_shape=None, **kwargs):
        if input_dim:
            input_shape = (input_dim, )
        super(RepeatVector, self).__init__(None,
                                           n,
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
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_dim: Dimensionality of the input for 2D input. For nD input, you can alternatively
               specify 'input_shape' when using this layer as the first layer.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> dense = Dense(10, input_dim=8, name="dense1")
    creating: createZooKerasDense
    """
    def __init__(self, output_dim, init="glorot_uniform", limits=None, activation=None,
                 W_regularizer=None, b_regularizer=None,
                 bias=True, input_dim=None, input_shape=None, **kwargs):
        if input_dim:
            input_shape = (input_dim, )
        super(Dense, self).__init__(None,
                                    output_dim,
                                    init,
                                    list(limits) if limits else None,
                                    activation,
                                    W_regularizer,
                                    b_regularizer,
                                    bias,
                                    list(input_shape) if input_shape else None,
                                    **kwargs)


class GetShape(ZooKerasLayer):
    """
    GetShape gets the value of input_shape.
    For example, if input_shape = (2, 3, 4),
    then output will be (2, 3, 4).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.
    >>> getShape = GetShape(input_shape=(3, 4, 5))
    creating: createZooKerasGetShape
    """
    def __init__(self, input_shape=None, **kwargs):
        super(GetShape, self).__init__(None,
                                       list(input_shape) if input_shape else None,
                                       **kwargs)


class SparseDense(ZooKerasLayer):
    """
    SparseDense is the sparse version of layer Dense. SparseDense has two different from Dense:
    firstly, SparseDense's input Tensor is a SparseTensor. Secondly, SparseDense doesn't backward
    gradient to next layer in the backpropagation by default, as the gradInput of SparseDense is
    useless and very big in most cases.

    But, considering model like Wide&Deep, we provide backwardStart and backwardLength to backward
    part of the gradient to next layer.

    The most common input is 2D.

    When you use this layer as the first layer of a model, you need to provide the argument
    inputShape (a Single Shape, does not include the batch dimension).

    # Arguments
    output_dim: The size of output dimension.
    init: String representation of the initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    backward_start: backward start index, counting from 1.
    backward_length: backward length.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> sparseDense = SparseDense(10, input_shape=(10, 4), name="sparseDense")
    creating: createZooKerasSparseDense
    """

    def __init__(self, output_dim, init="glorot_uniform", activation=None,
                 W_regularizer=None, b_regularizer=None, backward_start=-1,
                 backward_length=-1, init_weight=None, init_bias=None,
                 init_grad_weight=None, init_grad_bias=None,
                 bias=True, input_shape=None, **kwargs):
        super(SparseDense, self).__init__(None,
                                          output_dim,
                                          init,
                                          activation,
                                          W_regularizer,
                                          b_regularizer,
                                          backward_start,
                                          backward_length,
                                          init_weight,
                                          init_bias,
                                          init_grad_weight,
                                          init_grad_bias,
                                          bias,
                                          list(input_shape) if input_shape else None,
                                          **kwargs)


class MaxoutDense(ZooKerasLayer):
    """
    A dense maxout layer that takes the element-wise maximum of linear layers.
    This allows the layer to learn a convex, piecewise linear activation function over the inputs.
    The input of this layer should be 2D.

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    output_dim: The size of output dimension.
    nb_feature: Number of Dense layers to use internally. Int. Default is 4.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_dim: Dimensionality of the input. Alternatively, you can specify 'input_shape'
               when using this layer as the first layer.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> maxoutdense = MaxoutDense(6, input_shape=(10, ))
    creating: createZooKerasMaxoutDense
    """
    def __init__(self, output_dim, nb_feature=4, W_regularizer=None, b_regularizer=None,
                 bias=True, input_dim=None, input_shape=None, **kwargs):
        if input_dim:
            input_shape = (input_dim, )
        super(MaxoutDense, self).__init__(None,
                                          output_dim,
                                          nb_feature,
                                          W_regularizer,
                                          b_regularizer,
                                          bias,
                                          list(input_shape) if input_shape else None,
                                          **kwargs)


class Highway(ZooKerasLayer):
    """
    Densely connected highway network. Highway layers are a natural extension of LSTMs
    to feedforward networks.
    The input of this layer should be 2D, i.e. (batch, input dim).

    When you use this layer as the first layer of a model, you need to provide the argument
    input_shape (a shape tuple, does not include the batch dimension).

    # Arguments
    activation: String representation of the activation function to use
                (such as 'relu' or 'sigmoid'). Default is None.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_dim: Dimensionality of the input. Alternatively, you can specify 'input_shape'
               when using this layer as the first layer.
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> highway = Highway(activation='relu', input_shape=(8, ))
    creating: createZooKerasHighway
    """
    def __init__(self, activation=None, W_regularizer=None, b_regularizer=None,
                 bias=True, input_dim=None, input_shape=None, **kwargs):
        if input_dim:
            input_shape = (input_dim, )
        super(Highway, self).__init__(None,
                                      activation,
                                      W_regularizer,
                                      b_regularizer,
                                      bias,
                                      list(input_shape) if input_shape else None,
                                      **kwargs)


class Max(ZooKerasLayer):
    """
    Applies a max operation over dimension `dim`

    # Arguments
    dim: max along this dimension
    num_input_dims: Optional. If in a batch model, set to the inputDims.
    return_value: Optional. Config whether return value or indices
    input_shape: A shape tuple, not including batch.

    >>> max = Max(dim=1, input_shape=(3, 5))
    creating: createZooKerasMax
    """
    def __init__(self, dim, num_input_dims=INTMIN, return_value=True, input_shape=None, **kwargs):
        super(Max, self).__init__(None,
                                  dim,
                                  num_input_dims,
                                  return_value,
                                  list(input_shape) if input_shape else None,
                                  **kwargs)


class ExpandDim(ZooKerasLayer):
    """
    Expand_dim is an improved layer to suuport 1D input.
    For example, if we get an 1D input with shape(3),
    we will return the shape(1, 3) after we use expand_dim(0, input).
    # Arguments
    dim: The specified axis to expand dimension on.
    input_shape: A shape tuple, not including batch.

    >>> expandDim = ExpandDim(dim=0, input_shape=(3, 2))
    creating: createZooKerasExpandDim
    """
    def __init__(self, dim, input_shape=None, **kwargs):
        super(ExpandDim, self).__init__(None,
                                        dim,
                                        list(input_shape) if input_shape else None,
                                        **kwargs)
