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

from bigdl.nn.layer import Layer, Sequential as TSequential
from bigdl.util.common import callBigDlFunc, JTensor, JavaValue

if sys.version >= '3':
    long = int
    unicode = str


class InferShape(JavaValue):
    def __init__(self, bigdl_type="float"):
        self.bigdl_type = bigdl_type

    def __process_shape(self, output_shape):
        return tuple([None] + output_shape[1:])

    def get_input_shape(self):
        """
        Return a list of shape tuples if merge layer is the first layer.
        Return one shape tuple otherwise.
        """
        input = callBigDlFunc(self.bigdl_type, "getInputShape",
                              self.value)
        if len(input) == 1:
            return self.__process_shape(input(0))
        else:
            res = []
            for i in input:
                res.append(self.__process_shape(i))
            return res

    def get_output_shape(self):
        return self.__process_shape(callBigDlFunc(self.bigdl_type, "getOutputShape",
                                                  self.value))


class KerasLayer(Layer):
    def jvm_class_constructor(self):
        name = "createKeras" + self.__class__.__name__
        print("creating: " + name)
        return name


class Sequential(TSequential, InferShape):
    """
    Container for a Sequential model.

    >>> sequential = Sequential()
    creating: createSequential
    """
    def __init__(self, bigdl_type="float"):
        super(Sequential, self).__init__(bigdl_type, True)


class InputLayer(KerasLayer):
    """
    Layer to be used as an entry point into a model.

    # Arguments
    input_shape: Shape tuple, not including the batch axis.

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
    inputShape (a shape tuple, does not include the batch dimension).
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
    input_shape: A shape tuple, not including batch.

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
    Turn positive integers (indexes) into dense vectors of fixed size.
    The input of this layer should be 2D.

    This layer can only be used as the first layer in a model, you need to provide the argument
    inputShape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_dim: Int > 0. Size of the vocabulary.
    output_dim: Int >= 0. Dimension of the dense embedding.
    init: String representations of initialization method for the weights of the layer.
          Default is 'uniform'.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the embedding matrix. Default is None.
    input_shape: A shape tuple, not including batch.

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
    Batch normalization layer.
    Normalize the activations of the previous layer at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation standard deviation close to 1.
    It is a feature-wise normalization, each feature map in the input will be normalized separately.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    inputShape (a shape tuple, does not include the batch dimension).

    # Arguments
    epsilon: Small float > 0. Fuzz parameter. Default is 0.001.
    momentum: Float. Momentum in the computation of the exponential average of the mean and
              standard deviation of the data, for feature-wise normalization. Default is 0.99.
    beta_init: Name of initialization function for shift parameter. Default is 'zero'.
    gamma_init: Name of initialization function for scale parameter. Default is 'one'.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
                  For 'th', axis along which to normalize is 1. For 'tf', axis is 3.
    input_shape: A shape tuple, not including batch.

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
    Used to merge a list of tensors into a single tensor, following some merge mode.
    Merge must have at least two input layers.

    When using this layer as the first layer in a model, you need to provide the argument
    inputShape for input layers (shape tuples, does not include the batch dimension).

    # Arguments
    layers: A list of layer instances. Must be more than one layer.
    mode: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos',
          'dot', 'max'. Default is 'sum'.
    concat_axis: Integer, axis to use in mode concat. Only specify this when mode is 'concat'.
                 Default is -1, meaning the last axis of the input.
    input_shape: A list of shape tuples, each not including batch.

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


class Dropout(KerasLayer):
    """
    Applies Dropout to the input by randomly setting a fraction 'p' of input units to 0 at each
    update during training time in order to prevent overfitting.

    When you use this layer as the first layer of a model, you need to provide the argument
    inputShape (a shape tuple, does not include the batch dimension).

    # Arguments
    p: Fraction of the input units to drop. Float between 0 and 1.
    input_shape: A shape tuple, not including batch.

    >>> dropout = Dropout(0.25, input_shape=(2, 3))
    creating: createKerasDropout
    """
    def __init__(self, p, input_shape=None, bigdl_type="float"):
        super(Dropout, self).__init__(None, bigdl_type,
                                      p,
                                      list(input_shape) if input_shape else None)


class Flatten(KerasLayer):
    """
    Flattens the input without affecting the batch size.

    When you use this layer as the first layer of a model, you need to provide the argument
    inputShape (a shape tuple, does not include the batch dimension).

    # Arguments
    input_shape: A shape tuple, not including batch.

    >>> flatten = Flatten(input_shape=(3, 10, 2))
    creating: createKerasFlatten
    """
    def __init__(self, input_shape=None, bigdl_type="float"):
        super(Flatten, self).__init__(None, bigdl_type,
                                      list(input_shape) if input_shape else None)


class Reshape(KerasLayer):
    """
    Reshapes an output to a certain shape.
    Supports shape inference by allowing one -1 in the target shape.
    For example, if input_shape = (2, 3, 4), target_shape = (3, -1),
    then output_shape will be (3, 8).

    When you use this layer as the first layer of a model, you need to provide the argument
    inputShape (a shape tuple, does not include the batch dimension).

    # Arguments
    target_shape: A shape tuple. The target shape that you desire to have. Batch dimension should be excluded.
    input_shape: A shape tuple, not including batch.

    >>> reshape = Reshape((2, 10), input_shape=(5, 4))
    creating: createKerasReshape
    """
    def __init__(self, target_shape, input_shape=None, bigdl_type="float"):
        super(Reshape, self).__init__(None, bigdl_type,
                                      target_shape,
                                      list(input_shape) if input_shape else None)


class Activation(KerasLayer):
    """
    Simple activation function to be applied to the output.
    Available activations: 'tanh', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'hard_sigmoid'.

    When you use this layer as the first layer of a model, you need to provide the argument
    inputShape (a shape tuple, does not include the batch dimension).

    # Arguments
    activation: Name of the activation function as string.
    input_shape: A shape tuple, not including batch.

    >>> activation = Activation("relu", input_shape=(3, 4))
    creating: createKerasActivation
    """
    def __init__(self, activation, input_shape=None, bigdl_type="float"):
        super(Activation, self).__init__(None, bigdl_type,
                                         activation,
                                         list(input_shape) if input_shape else None)


class Convolution2D(KerasLayer):
    """
    Applies a 2D convolution over an input image composed of several input planes.
    You can also use Conv2D as an alias of this layer.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    inputShape (a shape tuple, does not include the batch dimension).
    e.g. input_shape=(3, 128, 128) for 128x128 RGB pictures.

    # Arguments
    nb_filter: Number of convolution filters to use.
    nb_row: Number of rows in the convolution kernel.
    nb_col: Number of rows in the convolution kernel.
    init: String representations of initialization method for the weights of the layer.
          Default is 'glorot_uniform'.
    activation: String representations of activation function to use (such as 'relu' or 'sigmoid').
                Default is None.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    subsample: Tuple of length 2 corresponding to the step of the convolution in the
               height and width dimension. Also called strides elsewhere. Default is (1, 1).
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    W_regularizer: An instance of [[Regularizer]], (eg. L1 or L2 regularization),
                   applied to the input weights matrices. Default is None.
    b_regularizer: An instance of [[Regularizer]], applied to the bias. Default is None.
    bias: Whether to include a bias (i.e. make the layer affine rather than linear).
          Default is True.
    input_shape: A shape tuple, not including batch.

    >>> conv2d = Convolution2D(32, 3, 3, input_shape=(3, 128, 128))
    creating: createKerasConvolution2D
    """
    def __init__(self, nb_filter, nb_row, nb_col,
                 init="glorot_uniform", activation=None,
                 border_mode="valid", subsample=(1, 1), dim_ordering="th",
                 W_regularizer=None, b_regularizer=None, bias=True,
                 input_shape=None, bigdl_type="float"):
        super(Convolution2D, self).__init__(None, bigdl_type,
                                            nb_filter,
                                            nb_row,
                                            nb_col,
                                            init,
                                            activation,
                                            border_mode,
                                            subsample,
                                            dim_ordering,
                                            W_regularizer,
                                            b_regularizer,
                                            bias,
                                            list(input_shape) if input_shape else None)


Conv2D = Convolution2D


class MaxPooling2D(KerasLayer):
    """
    Applies max pooling operation for spatial data.
    The input of this layer should be 4D.

    When you use this layer as the first layer of a model, you need to provide the argument
    inputShape (a shape tuple, does not include the batch dimension).

    # Arguments
    poolSize Tuple of length 2 corresponding to the downscale vertically and horizontally.
             Default is (2, 2), which will halve the image in each dimension.
    strides: Tuple of length 2. Stride values. Default is None, and in this case it will be equal to pool_size.
    border_mode: Either 'valid' or 'same'. Default is 'valid'.
    dim_ordering: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
    input_shape: A shape tuple, not including batch.

    >>> maxpooling2d = MaxPooling2D((2, 2), input_shape=(3, 32, 32))
    creating: createKerasMaxPooling2D
    """
    def __init__(self, pool_size=(2, 2), strides=None,
                 border_mode='valid', dim_ordering='th',
                 input_shape=None, bigdl_type="float"):
        super(MaxPooling2D, self).__init__(None, bigdl_type,
                                           pool_size,
                                           strides,
                                           border_mode,
                                           dim_ordering,
                                           list(input_shape) if input_shape else None)