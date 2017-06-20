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

from bigdl.util.common import JTensor
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import callJavaFunc
from bigdl.util.common import get_spark_context
from bigdl.util.common import to_list

if sys.version >= '3':
    long = int
    unicode = str

INTMAX = 2147483647
INTMIN = -2147483648
DOUBLEMAX = 1.7976931348623157E308


class Node(JavaValue):
    """
    Represent a node in a graph. The connections between nodes are directed.
    """
    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)
        self.bigdl_type = bigdl_type

    @classmethod
    def of(cls, jvalue, bigdl_type="float"):
        return Node(jvalue, bigdl_type)

    def element(self):
        return Layer.of(self.value.element())


class Layer(JavaValue):
    """
    Layer is the basic component of a neural network
    and it's also the base class of layers.
    Layer can connect to others to construct a complex neural network.
    """

    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, JavaValue.jvm_class_constructor(self), *args)
        self.bigdl_type = bigdl_type

    def __str__(self):
        """
        >>> conv2 = SpatialConvolution(6, 12, 5, 5).set_name("conv2")
        creating: createSpatialConvolution
        >>> print(conv2)
        SpatialConvolution[conv2](6 -> 12, 5 x 5, 1, 1, 0, 0)
        """
        return self.value.toString()

    def __call__(self, x=None):
        """
        Some other modules point to current module
        :param x: upstream module nodes. x is either a Node or list of Node.
        :return: node containing current module
        """

        x = x if x else []
        return Node.of(callBigDlFunc(self.bigdl_type,
                                     "createNode",
                                     self,
                                     to_list(x)))

    @classmethod
    def of(cls, jvalue, bigdl_type="float"):
        """
        Create a Python Layer base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Layer
        """
        model = Layer(jvalue, bigdl_type)
        return model

    def set_name(self, name):
        """
        Give this model a name. There would be a generated name
        consist of class name and UUID if user doesn't set it.
        """
        callJavaFunc(get_spark_context(), self.value.setName, name)
        return self

    def name(self):
        """
        Name of this layer
        """
        return callJavaFunc(get_spark_context(), self.value.getName)

    def set_seed(self, seed=123):
        """
        You can control the random seed which used to init weights for this model.

        :param seed: random seed
        :return: Model itself.
        """
        callBigDlFunc(self.bigdl_type, "setModelSeed", seed)
        return self

    def get_dtype(self):
        if "float" == self.bigdl_type:
            return "float32"
        else:
            return "float64"

    @staticmethod
    def check_input(input):
        if type(input) is list:
            if len(input) == 0:
                raise Exception('Error when checking: empty input')
            if not hasattr(input[0], 'shape'):
                raise Exception(
                    'Error when checking: expecting list of ndarray')
            return [JTensor.from_ndarray(i) for i in input]
        else:
            if not hasattr(input, 'shape'):
                raise Exception(
                    'Error when checking: expecting list of ndarray')
            return [JTensor.from_ndarray(input)]

    @staticmethod
    def convert_output(output):
        if type(output) is JTensor:
            return output.to_ndarray()
        else:
            return [x.to_ndarray() for x in output]

    def forward(self, input):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        Takes an input object, and computes the corresponding output of the module

        :param input: ndarray or list of ndarray
        :return: ndarray or list of ndarray
        """
        output = callBigDlFunc(self.bigdl_type,
                               "modelForward",
                               self.value,
                               self.check_input(input))
        return self.convert_output(output)

    def backward(self, input, grad_output):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        Performs a back-propagation step through the module, with respect to the given input. In
        general this method makes the assumption forward(input) has been called before, with the same
        input. This is necessary for optimization reasons. If you do not respect this rule, backward()
        will compute incorrect gradients.

        :param input: ndarray or list of ndarray
        :param grad_output: ndarray or list of ndarray
        :return: ndarray or list of ndarray
        """
        output = callBigDlFunc(self.bigdl_type,
                               "modelBackward",
                               self.value,
                               self.check_input(input),
                               self.check_input(grad_output))
        return self.convert_output(output)

    def zero_grad_parameters(self):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        If the module has parameters, this will zero the accumulation of the gradients with respect
        to these parameters. Otherwise, it does nothing.
        """
        callJavaFunc(get_spark_context(), self.value.zeroGradParameters)

    def update_parameters(self, learning_rate):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        """
        callBigDlFunc(self.bigdl_type,
                      "updateParameters",
                      self.value,
                      learning_rate)

    def reset(self):
        """
        Initialize the model weights.
        """
        callJavaFunc(get_spark_context(), self.value.reset)
        return self

    def parameters(self):
        """
        Get the model parameters which containing: weight, bias, gradBias, gradWeight

        :return: dict(layername -> dict(parametername -> ndarray))
        """
        name_to_params = callBigDlFunc(self.bigdl_type,
                                       "modelGetParameters",
                                       self.value)

        def to_ndarray(params):
            return dict((param_name,
                         np.array(values[0], dtype=self.get_dtype()).reshape(
                             values[1])) for param_name, values in
                        params.items())

        return dict((layer_name, to_ndarray(params)) for layer_name, params in
                name_to_params.items())

    def predict(self, data_rdd):
        """
        Model inference base on the given data.
        You need to invoke collect() to trigger those action \
        as the returning result is an RDD.

        :param data_rdd: the data to be predict.
        :return: An RDD represent the predict result.
        """
        result = callBigDlFunc(self.bigdl_type,
                             "modelPredictRDD", self.value, data_rdd)
        return result.map(lambda data: data.to_ndarray())

    def test(self, val_rdd, batch_size, val_methods):
        """
        A method to benchmark the model quality.

        :param val_rdd: the input data
        :param batch_size: batch size
        :param val_methods: a list of validation methods. i.e: Top1Accuracy,Top5Accuracy and Loss.
        :return:
        """
        return callBigDlFunc(self.bigdl_type,
                             "modelTest",
                             self.value,
                             val_rdd, batch_size, val_methods)

    def set_weights(self, weights):
        """
        Set weights for this layer

        :param weights: a list of numpy arrays which represent weight and bias
        :return:

        >>> linear = Linear(3,2)
        creating: createLinear
        >>> linear.set_weights([np.array([[1,2,3],[4,5,6]]), np.array([7,8])])
        >>> weights = linear.get_weights()
        >>> weights[0].shape == (2,3)
        True
        >>> weights[0][0]
        array([ 1.,  2.,  3.], dtype=float32)
        >>> weights[1]
        array([ 7.,  8.], dtype=float32)
        >>> relu = ReLU()
        creating: createReLU
        >>> from py4j.protocol import Py4JJavaError
        >>> try:
        ...     relu.set_weights([np.array([[1,2,3],[4,5,6]]), np.array([7,8])])
        ... except Py4JJavaError as err:
        ...     print(err.java_exception)
        ...
        java.lang.IllegalArgumentException: requirement failed: this layer does not have weight/bias
        >>> relu.get_weights()
        The layer does not have weight/bias
        >>> add = Add(2)
        creating: createAdd
        >>> try:
        ...     add.set_weights([np.array([7,8]), np.array([1,2])])
        ... except Py4JJavaError as err:
        ...     print(err.java_exception)
        ...
        java.lang.IllegalArgumentException: requirement failed: the number of input weight/bias is not consistant with number of weight/bias of this layer
        >>> cAdd = CAdd([4, 1])
        creating: createCAdd
        >>> cAdd.set_weights(np.ones([4, 1]))
        >>> (cAdd.get_weights()[0] == np.ones([4, 1])).all()
        True
        """
        tensors = [JTensor.from_ndarray(param, self.bigdl_type) for param in to_list(weights)]
        callBigDlFunc(self.bigdl_type, "setWeights", self.value, tensors)

    def get_weights(self):
        """
        Get weights for this layer

        :return: list of numpy arrays which represent weight and bias
        """
        tensorWeights = callBigDlFunc(self.bigdl_type,
                              "getWeights", self.value)
        if tensorWeights is not None:
            return [tensor.to_ndarray() for tensor in tensorWeights]
        else:
            print("The layer does not have weight/bias")
            return None

    def save(self, path, over_write = False):
        callBigDlFunc(self.bigdl_type, "modelSave", self.value, path,
                      over_write)


class Container(Layer):
    '''
     [[Container]] is a sub-class of Model that declares methods defined in all containers.
     A container usually contain some other modules which can be added through the "add" method
    '''

    def __init__(self, jvalue, bigdl_type, *args):
        super(Container, self).__init__(jvalue, bigdl_type, *args)

    def add(self, model):
        self.value.add(model.value)
        return self


class Model(Container):
    """
    A graph container. Each node can have multiple inputs. The output of the node should be a
    tensor. The output tensor can be connected to multiple nodes. So the module in each node can
    have a tensor or table input, and should have a tensor output.

    The graph container can have multiple inputs and multiple outputs. If there's one input,
    the input data fed to the graph module should be a tensor. If there're multiple inputs,
    the input data fed to the graph module should be a table, which is actually an sequence of
    tensor. The order of the input tensors should be same with the order of the input nodes.
    This is also applied to the gradient from the module in the back propagation.

    If there's one output, the module output is a tensor. If there're multiple outputs, the module
    output is a table, which is actually an sequence of tensor. The order of the output tensors is
    same with the order of the output modules. This is also applied to the gradient passed to the
    module in the back propagation.

    All inputs should be able to connect to outputs through some paths in the graph.
    It is allowed that some successors of the inputs node are not connect to outputs.
    If so, these nodes will be excluded in the computation.
    """

    def __init__(self,
                 inputs,
                 outputs,
                 bigdl_type="float"):
        super(Model, self).__init__(None, bigdl_type,
                                    to_list(inputs),
                                    to_list(outputs))

    @staticmethod
    def load(path, bigdl_type="float"):
        """
        Load a pre-trained Bigdl model.

        :param path: The path containing the pre-trained model.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadBigDL", path)
        return Layer.of(jmodel)

    @staticmethod
    def load_torch(path, bigdl_type="float"):
        """
        Load a pre-trained Torch model.

        :param path: The path containing the pre-trained model.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadTorch", path)
        return Layer.of(jmodel)

    @staticmethod
    def load_caffe(model, defPath, modelPath, match_all=True, bigdl_type="float"):
        """
        Load a pre-trained Caffe model.


        :param model: A bigdl model definition \which equivalent to the pre-trained caffe model.
        :param defPath: The path containing the caffe model definition.
        :param modelPath: The path containing the pre-trained caffe model.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadCaffe", model, defPath, modelPath, match_all)
        return Layer.of(jmodel)


class Linear(Layer):

    '''
    The [[Linear]] module applies a linear transformation to the input data,
    i.e. `y = Wx + b`. The input given in `forward(input)` must be either
    a vector (1D tensor) or matrix (2D tensor). If the input is a vector, it must
    have the size of `inputSize`. If it is a matrix, then each row is assumed to be
    an input sample of given batch (the number of rows means the batch size and
    the number of columns should be equal to the `inputSize`).


    :param input_size: the size the each input sample
    :param output_size: the size of the module output of each sample
    :param init_method: two initialized methods are supported here, which are [[Default]]and [[Xavier]], where [[Xavier]] set bias to zero here. For moredetailed information about `initMethod`, please refer to[[InitializationMethod]]


    >>> linear = Linear(100, 10, "Xavier")
    creating: createLinear
    '''

    def __init__(self, input_size, output_size, init_method="default", with_bias=True,
                 bigdl_type="float"):
        super(Linear, self).__init__(None, bigdl_type, input_size, output_size,
                                     init_method, with_bias)

    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                                   weight_init_method, bias_init_method)


class ReLU(Layer):

    '''
    Applies the rectified linear unit (ReLU) function element-wise to the input Tensor,
     thus outputting a Tensor of the same dimension.


    ReLU is defined as: f(x) = max(0, x)
    Can optionally do its operation in-place without using extra state memory


    >>> relu = ReLU()
    creating: createReLU
    '''

    def __init__(self, ip=False, bigdl_type="float"):
        super(ReLU, self).__init__(None, bigdl_type, ip)


class Tanh(Layer):

    '''
    Applies the Tanh function element-wise to the input Tensor, thus outputting a Tensor of the same
    dimension. Tanh is defined as f(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x)).


    >>> tanh = Tanh()
    creating: createTanh
    '''

    def __init__(self, bigdl_type="float"):
        super(Tanh, self).__init__(None, bigdl_type)


class Echo(Layer):

    '''
    This module is for debug purpose, which can print activation and gradient in your model
    topology


    >>> echo = Echo()
    creating: createEcho
    '''

    def __init__(self, bigdl_type="float"):
        super(Echo, self).__init__(None, bigdl_type)


class LogSoftMax(Layer):

    '''
    Applies the LogSoftMax function to an n-dimensional input Tensor.
    LogSoftmax is defined as: f_i(x) = log(1 / a exp(x_i))
    where a = sum_j[exp(x_j)].


    >>> logSoftMax = LogSoftMax()
    creating: createLogSoftMax
    '''

    def __init__(self, bigdl_type="float"):
        super(LogSoftMax, self).__init__(None, bigdl_type)


class Sequential(Container):

    '''
    Sequential provides a means to plug layers together
    in a feed-forward fully connected manner.


    >>> echo = Echo()
    creating: createEcho
    >>> s = Sequential()
    creating: createSequential
    >>> s = s.add(echo)
    >>> s = s.add(s)
    >>> s = s.add(echo)


    '''

    def __init__(self, bigdl_type="float"):
        super(Sequential, self).__init__(None, bigdl_type)


class SpatialConvolution(Layer):

    '''
    Applies a 2D convolution over an input image composed of several input planes.
    The input tensor in forward(input) is expected to be
    a 3D tensor (nInputPlane x height x width).


    :param n_input_plane: The number of expected input planes in the image given into forward()
    :param n_output_plane: The number of output planes the convolution layer will produce.
    :param kernel_w: The kernel width of the convolution
    :param kernel_h: The kernel height of the convolution
    :param stride_w: The step of the convolution in the width dimension.
    :param stride_h: The step of the convolution in the height dimension
    :param pad_w: The additional zeros added per width to the input planes.
    :param pad_h: The additional zeros added per height to the input planes.
    :param n_group: Kernel group number
    :param propagate_back: Propagate gradient back
    :param init_method: Initialization method to initialize bias and weight


    >>> spatialConvolution = SpatialConvolution(6, 12, 5, 5)
    creating: createSpatialConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 kernel_w,
                 kernel_h,
                 stride_w=1,
                 stride_h=1,
                 pad_w=0,
                 pad_h=0,
                 n_group=1,
                 propagate_back=True,
                 init_method="default",
                 bigdl_type="float"):
        super(SpatialConvolution, self).__init__(None, bigdl_type,
                                                 n_input_plane,
                                                 n_output_plane,
                                                 kernel_w,
                                                 kernel_h,
                                                 stride_w,
                                                 stride_h,
                                                 pad_w,
                                                 pad_h,
                                                 n_group,
                                                 propagate_back,
                                                 init_method)
    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                  weight_init_method, bias_init_method)


class SpatialMaxPooling(Layer):

    '''
    Applies 2D max-pooling operation in kWxkH regions by step size dWxdH steps.
    The number of output features is equal to the number of input planes.
    If the input image is a 3D tensor nInputPlane x height x width,
    the output image size will be nOutputPlane x oheight x owidth where
    owidth  = op((width  + 2*padW - kW) / dW + 1)
    oheight = op((height + 2*padH - kH) / dH + 1)
    op is a rounding operator. By default, it is floor.
    It can be changed by calling :ceil() or :floor() methods.

    :param kW:              kernel width
    :param kH:              kernel height
    :param dW:              step size in width
    :param dH:              step size in height
    :param padW:            padding in width
    :param padH:            padding in height

    >>> spatialMaxPooling = SpatialMaxPooling(2, 2, 2, 2)
    creating: createSpatialMaxPooling
    '''
    # to_ceil: call floor() when False; call ceil() when True

    def __init__(self, kw,
                 kh,
                 dw,
                 dh,
                 pad_w=0,
                 pad_h=0,
                 to_ceil=False,
                 bigdl_type="float"):
        super(SpatialMaxPooling, self).__init__(None, bigdl_type, kw,
                                                kh,
                                                dw,
                                                dh,
                                                pad_w,
                                                pad_h,
                                                to_ceil)


class Select(Layer):

    '''
    A Simple layer selecting an index of the input tensor in the given dimension


    :param dimension: the dimension to select
    :param index: the index of the dimension to be selected


    >>> select = Select(1, 1)
    creating: createSelect
    '''

    def __init__(self, dim, index, bigdl_type="float"):
        super(Select, self).__init__(None, bigdl_type, dim, index)

class Recurrent(Container):
    '''
    Recurrent module is a container of rnn cells
    Different types of rnn cells can be added using add() function


    >>> recurrent = Recurrent()
    creating: createRecurrent
    '''

    def __init__(self, bigdl_type="float"):
        super(Recurrent, self).__init__(None, bigdl_type)


class LSTM(Layer):
    '''
|   Long Short Term Memory architecture.
|   Ref.
|   A.: http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
|   B. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
|   C. http://arxiv.org/pdf/1503.04069v1.pdf
|   D. https://github.com/wojzaremba/lstm
|   E. https://github.com/Element-Research/rnn/blob/master/FastLSTM.lua


    :param inputSize: the size of each input vector
    :param hiddenSize: Hidden unit size in the LSTM
    :param  p: is used for [[Dropout]] probability. For more details aboutRNN dropouts, please refer to[RnnDrop: A Novel Dropout for RNNs in ASR](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)


    >>> lstm = LSTM(4, 3, 0.5)
    creating: createLSTM
    '''

    def __init__(self, input_size, hidden_size, p=0.0, bigdl_type="float"):
        super(LSTM, self).__init__(None, bigdl_type, input_size, hidden_size, p)


class LSTMPeephole(Layer):
    '''
|   Long Short Term Memory architecture with peephole.
|   Ref. A.: http://arxiv.org/pdf/1303.5778v1 (blueprint for this module)
|   B. http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf
|   C. http://arxiv.org/pdf/1503.04069v1.pdf
|   D. https://github.com/wojzaremba/lstm
|   E. https://github.com/Element-Research/rnn/blob/master/LSTM.lua


    :param input_size: the size of each input vector
    :param hidden_size: Hidden unit size in the LSTM
    :param  p: is used for [[Dropout]] probability. For more details aboutRNN dropouts, please refer to[RnnDrop: A Novel Dropout for RNNs in ASR](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)


    >>> lstm = LSTMPeephole(4, 3, 0.5)
    creating: createLSTMPeephole
    '''

    def __init__(self, input_size, hidden_size, p=0.0, bigdl_type="float"):
        super(LSTMPeephole, self).__init__(None, bigdl_type, input_size, hidden_size, p)


class GRU(Layer):
    '''
    Gated Recurrent Units architecture.
    The first input in sequence uses zero value for cell and hidden state


|   Ref.
|   http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
|   https://github.com/Element-Research/rnn/blob/master/GRU.lua


    :param input_size: the size of each input vector
    :param hidden_size: Hidden unit size in GRU
    :param  p: is used for [[Dropout]] probability. For more details aboutRNN dropouts, please refer to[RnnDrop: A Novel Dropout for RNNs in ASR](http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)[A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)


    >>> gru = GRU(4, 3, 0.5)
    creating: createGRU
    '''

    def __init__(self,  input_size, hidden_size, p=0.0, bigdl_type="float"):
        super(GRU, self).__init__(None, bigdl_type, input_size, hidden_size, p)


class RnnCell(Layer):
    '''
    It is a simple RNN. User can pass an activation function to the RNN.


    :param input_size: the size of each input vector
    :param hidden_size: Hidden unit size in simple RNN
    :param activation: activation function


    >>> reshape = RnnCell(4, 3, Tanh())
    creating: createTanh
    creating: createRnnCell
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 activation,
                 bigdl_type="float"):
        super(RnnCell, self).__init__(None, bigdl_type, input_size, hidden_size, activation)


class TimeDistributed(Layer):
    '''
    This layer is intended to apply contained layer to each temporal time slice
    of input tensor.


    For instance, The TimeDistributed Layer can feed each time slice of input tensor
    to the Linear layer.


    >>> td = TimeDistributed(Linear(2, 3))
    creating: createLinear
    creating: createTimeDistributed
    '''

    def __init__(self, model, bigdl_type="float"):
        super(TimeDistributed, self).__init__(None, bigdl_type, model)


class Concat(Container):

    '''
    Concat concatenates the output of one layer of "parallel"
    modules along the provided {@code dimension}: they take the
    same inputs, and their output is concatenated.
```
                    +-----------+
               +---->  module1  -----+
               |    |           |    |
    input -----+---->  module2  -----+----> output
               |    |           |    |
               +---->  module3  -----+
                    +-----------+
```

    :param dimension: dimension


    >>> concat = Concat(2)
    creating: createConcat
    '''

    def __init__(self,
                 dimension,
                 bigdl_type="float"):
        super(Concat, self).__init__(None, bigdl_type,
                                     dimension)


class SpatialAveragePooling(Layer):

    '''
    Applies 2D average-pooling operation in kWxkH regions by step size dWxdH steps.
    The number of output features is equal to the number of input planes.


    :param kW: kernel width
    :param kH: kernel height
    :param dW: step width
    :param dH: step height
    :param padW: padding width
    :param padH: padding height
    :param ceilMode: whether the output size is to be ceiled or floored
    :param countIncludePad: whether to include padding when dividing thenumber of elements in pooling region
    :param divide: whether to do the averaging


    >>> spatialAveragePooling = SpatialAveragePooling(7,7)
    creating: createSpatialAveragePooling
    '''

    def __init__(self,
                 kw,
                 kh,
                 dw=1,
                 dh=1,
                 pad_w=0,
                 pad_h=0,
                 ceil_mode=False,
                 count_include_pad=True,
                 divide=True,
                 bigdl_type="float"):
        super(SpatialAveragePooling, self).__init__(None, bigdl_type,
                                                    kw,
                                                    kh,
                                                    dw,
                                                    dh,
                                                    pad_w,
                                                    pad_h,
                                                    ceil_mode,
                                                    count_include_pad,
                                                    divide)


class SpatialBatchNormalization(Layer):

    '''
    This file implements Batch Normalization as described in the paper:
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    by Sergey Ioffe, Christian Szegedy
    This implementation is useful for inputs coming from convolution layers.
    For non-convolutional layers, see [[BatchNormalization]]
    The operation implemented is:

```
          ( x - mean(x) )
    y = -------------------- * gamma + beta
       standard-deviation(x)
```

    where gamma and beta are learnable parameters.
    The learning of gamma and beta is optional.


    >>> spatialBatchNormalization = SpatialBatchNormalization(1)
    creating: createSpatialBatchNormalization
    '''

    def __init__(self,
                 n_output,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 bigdl_type="float"):
        super(SpatialBatchNormalization, self).__init__(None, bigdl_type,
                                                        n_output,
                                                        eps,
                                                        momentum,
                                                        affine)

    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class SpatialCrossMapLRN(Layer):

    '''
    Applies Spatial Local Response Normalization between different feature maps.
    The operation implemented is:
```
                                 x_f
    y_f =  -------------------------------------------------
            (k+(alpha/size)* sum_{l=l1 to l2} (x_l^2^))^beta^
```

    where x_f is the input at spatial locations h,w (not shown for simplicity) and feature map f,
    l1 corresponds to max(0,f-ceil(size/2)) and l2 to min(F, f-ceil(size/2) + size).
    Here, F is the number of feature maps.

    :param size:  the number of channels to sum over (for cross channel LRN) or the side length ofthe square region to sum over (for within channel LRN)
    :param alpha:  the scaling parameter
    :param beta:   the exponent
    :param k: a constant


    >>> spatialCrossMapLRN = SpatialCrossMapLRN()
    creating: createSpatialCrossMapLRN
    '''

    def __init__(self,
                 size=5,
                 alpha=1.0,
                 beta=0.75,
                 k=1.0,
                 bigdl_type="float"):
        super(SpatialCrossMapLRN, self).__init__(None, bigdl_type,
                                                 size,
                                                 alpha,
                                                 beta,
                                                 k)


class Dropout(Layer):

    '''
    Dropout masks(set to zero) parts of input using a bernoulli distribution.
    Each input element has a probability initP of being dropped. If scale is
    set, the outputs are scaled by a factor of 1/(1-initP) during training.
    During evaluating, output is the same as input.


    :param initP: probability to be dropped
    :param inplace: inplace model
    :param scale: if scale by a factor of 1/(1-initP)


    >>> dropout = Dropout(0.4)
    creating: createDropout
    '''

    def __init__(self,
                 init_p=0.5,
                 inplace=False,
                 scale=True,
                 bigdl_type="float"):
        super(Dropout, self).__init__(None, bigdl_type,
                                      init_p,
                                      inplace,
                                      scale)


class View(Layer):

    '''
    This module creates a new view of the input tensor using the sizes passed to the constructor.
    The method setNumInputDims() allows to specify the expected number of dimensions of the
    inputs of the modules. This makes it possible to use minibatch inputs when using a size -1
    for one of the dimensions.


    :param size: sizes use for creates a new view


    >>> view = View([1024,2])
    creating: createView
    '''

    def __init__(self,
                 sizes,
                 num_input_dims=0,
                 bigdl_type="float"):
        super(View, self).__init__(None, bigdl_type,
                                   sizes,
                                   num_input_dims)


class Abs(Layer):

    '''
    an element-wise abs operation


    >>> abs = Abs()
    creating: createAbs
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Abs, self).__init__(None, bigdl_type)


class Add(Layer):

    '''
    adds a bias term to input data ;

    :param input_size: size of input data

    >>> add = Add(1)
    creating: createAdd
    '''

    def __init__(self,
                 input_size,
                 bigdl_type="float"):
        super(Add, self).__init__(None, bigdl_type,
                                  input_size)
    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class AddConstant(Layer):

    '''
    adding a constant


    :param constant_scalar: constant value
    :param inplace: Can optionally do its operation in-place without using extra state memory


    >>> addConstant = AddConstant(1e-5, True)
    creating: createAddConstant
    '''

    def __init__(self,
                 constant_scalar,
                 inplace=False,
                 bigdl_type="float"):
        super(AddConstant, self).__init__(None, bigdl_type,
                                          constant_scalar,
                                          inplace)

class BatchNormalization(Layer):

    '''
    This layer implements Batch Normalization as described in the paper:
             "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
             Covariate Shift"
    by Sergey Ioffe, Christian Szegedy https://arxiv.org/abs/1502.03167


    This implementation is useful for inputs NOT coming from convolution layers. For convolution
    layers, use nn.SpatialBatchNormalization.


    The operation implemented is:
```
                ( x - mean(x) )
         y = -------------------- * gamma + beta
             standard-deviation(x)
```
    where gamma and beta are learnable parameters.The learning of gamma and beta is optional.


    :param n_output: output feature map number
    :param eps: avoid divide zero
    :param momentum: momentum for weight update
    :param affine: affine operation on output or not


    >>> batchNormalization = BatchNormalization(1, 1e-5, 1e-5, True)
    creating: createBatchNormalization
    '''
    def __init__(self,
                 n_output,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 bigdl_type="float"):
        super(BatchNormalization, self).__init__(None, bigdl_type,
                                                 n_output,
                                                 eps,
                                                 momentum,
                                                 affine)
    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class Bilinear(Layer):

    '''
    a bilinear transformation with sparse inputs,
    The input tensor given in forward(input) is a table containing both inputs x_1 and x_2,
    which are tensors of size N x inputDimension1 and N x inputDimension2, respectively.


    :param input_size1: input dimension of x_1
    :param input_size2: input dimension of x_2
    :param output_size: output dimension
    :param bias_res: whether use bias


    >>> bilinear = Bilinear(1, 1, 1, True)
    creating: createBilinear
    '''

    def __init__(self,
                 input_size1,
                 input_size2,
                 output_size,
                 bias_res=True,
                 bigdl_type="float"):
        super(Bilinear, self).__init__(None, bigdl_type,
                                       input_size1,
                                       input_size2,
                                       output_size,
                                       bias_res)
    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class Bottle(Container):

    '''
    Bottle allows varying dimensionality input to be forwarded through any module
    that accepts input of nInputDim dimensions, and generates output of nOutputDim dimensions.

    :param module: transform module
    :param n_input_dim: nInputDim dimensions of module
    :param n_output_dim1: output of nOutputDim dimensions


    >>> bottle = Bottle(Linear(100,10), 1, 1)
    creating: createLinear
    creating: createBottle
    '''

    def __init__(self,
                 module,
                 n_input_dim=2,
                 n_output_dim1=INTMAX,
                 bigdl_type="float"):
        super(Bottle, self).__init__(None, bigdl_type,
                                     module,
                                     n_input_dim,
                                     n_output_dim1)


class CAdd(Layer):

    '''
    This layer has a bias tensor with given size. The bias will be added element wise to the input
    tensor. If the element number of the bias tensor match the input tensor, a simply element wise
    will be done. Or the bias will be expanded to the same size of the input. The expand means
    repeat on unmatched singleton dimension(if some unmatched dimension isn't singleton dimension,
    it will report an error). If the input is a batch, a singleton dimension will be add to the
    first dimension before the expand.


    :param size: the size of the bias


    >>> cAdd = CAdd([1,2])
    creating: createCAdd
    '''

    def __init__(self,
                 size,
                 bigdl_type="float"):
        super(CAdd, self).__init__(None, bigdl_type,
                                   size)

    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class CAddTable(Layer):

    '''
    Merge the input tensors in the input table by element wise adding them together. The input
    table is actually an array of tensor with same size.


    :param inplace: reuse the input memory


    >>> cAddTable = CAddTable(True)
    creating: createCAddTable
    '''

    def __init__(self,
                 inplace=False,
                 bigdl_type="float"):
        super(CAddTable, self).__init__(None, bigdl_type,
                                        inplace)


class CDivTable(Layer):

    '''
    Takes a table with two Tensor and returns the component-wise division between them.


    >>> cDivTable = CDivTable()
    creating: createCDivTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CDivTable, self).__init__(None, bigdl_type)


class CMaxTable(Layer):

    '''
    Takes a table of Tensors and outputs the max of all of them.


    >>> cMaxTable = CMaxTable()
    creating: createCMaxTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CMaxTable, self).__init__(None, bigdl_type)


class CMinTable(Layer):

    '''
    Takes a table of Tensors and outputs the min of all of them.

    >>> cMinTable = CMinTable()
    creating: createCMinTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CMinTable, self).__init__(None, bigdl_type)


class CMul(Layer):

    '''
    Applies a component-wise multiplication to the incoming data


    :param size: size of the data


    >>> cMul = CMul([1,2])
    creating: createCMul
    '''

    def __init__(self,
                 size,
                 bigdl_type="float"):
        super(CMul, self).__init__(None, bigdl_type,
                                   size)

    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class CMulTable(Layer):

    '''
    Takes a table of Tensors and outputs the multiplication of all of them.


    >>> cMulTable = CMulTable()
    creating: createCMulTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CMulTable, self).__init__(None, bigdl_type)


class CSubTable(Layer):

    '''
    Takes a table with two Tensor and returns the component-wise subtraction between them.


    >>> cSubTable = CSubTable()
    creating: createCSubTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CSubTable, self).__init__(None, bigdl_type)


class Clamp(Layer):

    '''
    Clamps all elements into the range [min_value, max_value].
    Output is identical to input in the range,
    otherwise elements less than min_value (or greater than max_value)
    are saturated to min_value (or max_value).


    :param min:
    :param max:


    >>> clamp = Clamp(1, 3)
    creating: createClamp
    '''

    def __init__(self,
                 min,
                 max,
                 bigdl_type="float"):
        super(Clamp, self).__init__(None, bigdl_type,
                                    min,
                                    max)


class Contiguous(Layer):

    '''
    used to make input, grad_output both contiguous


    >>> contiguous = Contiguous()
    creating: createContiguous
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Contiguous, self).__init__(None, bigdl_type)


class Cosine(Layer):

    '''
    Cosine calculates the cosine similarity of the input to k mean centers. The input given in
    forward(input) must be either a vector (1D tensor) or matrix (2D tensor). If the input is a
    vector, it must have the size of inputSize. If it is a matrix, then each row is assumed to be
    an input sample of given batch (the number of rows means the batch size and the number of
    columns should be equal to the inputSize).


    :param input_size: the size of each input sample
    :param output_size: the size of the module output of each sample


    >>> cosine = Cosine(2,3)
    creating: createCosine
    '''

    def __init__(self,
                 input_size,
                 output_size,
                 bigdl_type="float"):
        super(Cosine, self).__init__(None, bigdl_type,
                                     input_size,
                                     output_size)
    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class CosineDistance(Layer):

    '''
    Outputs the cosine distance between inputs


    >>> cosineDistance = CosineDistance()
    creating: createCosineDistance
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CosineDistance, self).__init__(None, bigdl_type)


class DotProduct(Layer):

    '''
    This is a simple table layer which takes a table of two tensors as input
    and calculate the dot product between them as outputs


    >>> dotProduct = DotProduct()
    creating: createDotProduct
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(DotProduct, self).__init__(None, bigdl_type)


class ELU(Layer):

    '''
    D-A Clevert, Thomas Unterthiner, Sepp Hochreiter
    Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
    [http://arxiv.org/pdf/1511.07289.pdf]


    >>> eLU = ELU(1e-5, True)
    creating: createELU
    '''

    def __init__(self,
                 alpha=1.0,
                 inplace=False,
                 bigdl_type="float"):
        super(ELU, self).__init__(None, bigdl_type,
                                  alpha,
                                  inplace)


class Euclidean(Layer):

    '''
    Outputs the Euclidean distance of the input to outputSize centers

    :param inputSize: inputSize
    :param outputSize: outputSize
    :param T: Numeric type. Only support float/double now


    >>> euclidean = Euclidean(1, 1, True)
    creating: createEuclidean
    '''



    def __init__(self,
                 input_size,
                 output_size,
                 fast_backward=True,
                 bigdl_type="float"):
        super(Euclidean, self).__init__(None, bigdl_type,
                                        input_size,
                                        output_size,
                                        fast_backward)

    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class Exp(Layer):

    '''
    Applies element-wise exp to input tensor.

    >>> exp = Exp()
    creating: createExp
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Exp, self).__init__(None, bigdl_type)


class FlattenTable(Layer):

    '''
    This is a table layer which takes an arbitrarily deep table of Tensors
    (potentially nested) as input and a table of Tensors without any nested
    table will be produced


    >>> flattenTable = FlattenTable()
    creating: createFlattenTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(FlattenTable, self).__init__(None, bigdl_type)


class GradientReversal(Layer):

    '''
    It is a simple module preserves the input, but takes the
    gradient from the subsequent layer, multiplies it by -lambda
    and passes it to the preceding layer. This can be used to maximise
    an objective function whilst using gradient descent, as described in
     ["Domain-Adversarial Training of Neural Networks"
     (http://arxiv.org/abs/1505.07818)]


    :param lambda: hyper-parameter lambda can be set dynamically during training


    >>> gradientReversal = GradientReversal(1e-5)
    creating: createGradientReversal
    >>> gradientReversal = GradientReversal()
    creating: createGradientReversal
    '''

    def __init__(self,
                 the_lambda=1.0,
                 bigdl_type="float"):
        super(GradientReversal, self).__init__(None, bigdl_type,
                                               the_lambda)


class HardShrink(Layer):

    '''
    This is a transfer layer which applies the hard shrinkage function
    element-wise to the input Tensor. The parameter lambda is set to 0.5
    by default
```
            x, if x >  lambda
    f(x) =  x, if x < -lambda
            0, otherwise
```

   :param the_lambda: a threshold value whose default value is 0.5


    >>> hardShrink = HardShrink(1e-5)
    creating: createHardShrink
    '''

    def __init__(self,
                 the_lambda=0.5,
                 bigdl_type="float"):
        super(HardShrink, self).__init__(None, bigdl_type,
                                         the_lambda)


class HardTanh(Layer):

    '''
    Applies HardTanh to each element of input, HardTanh is defined:
```
             |  maxValue, if x > maxValue
      f(x) = |  minValue, if x < minValue
             |  x, otherwise
```
    :param min_value: minValue in f(x), default is -1.
    :param max_value: maxValue in f(x), default is 1.
    :param inplace: whether enable inplace model.


    >>> hardTanh = HardTanh(1e-5, 1e5, True)
    creating: createHardTanh
    >>> hardTanh = HardTanh()
    creating: createHardTanh
    '''

    def __init__(self,
                 min_value=-1.0,
                 max_value=1.0,
                 inplace=False,
                 bigdl_type="float"):
        super(HardTanh, self).__init__(None, bigdl_type,
                                       min_value,
                                       max_value,
                                       inplace)


class Index(Layer):

    '''
    Applies the Tensor index operation along the given dimension.


    :param dimension: the dimension to be indexed


    >>> index = Index(1)
    creating: createIndex
    '''

    def __init__(self,
                 dimension,
                 bigdl_type="float"):
        super(Index, self).__init__(None, bigdl_type,
                                    dimension)


class InferReshape(Layer):

    '''
    Reshape with the support of infered size,
    Positive numbers are used directly, setting the corresponding dimension of the output tensor.
    In addition, two special values are accepted:
    0 means "copy the respective dimension of the input".
    i.e., if the input has 2 as its 1st dimension,
    the output will have 2 as its 1st dimension as well
    -1 stands for "infer this from the other dimensions"
    this dimension is calculated to keep the overall element count the same as in the input.
    At most one -1 can be used in a reshape operation.
    For example, (4, 5, 6, 7) -> InferReshape (4, 0, 3, -1) -> (4, 5, 3, 14)
    with 1st and 3rd dim same as given size, with 2nd dim same as input, and the infered dim is 14


    :param size:      the target tensor size
    :param batch_mode: whether in batch mode


    >>> inferReshape = InferReshape([4, 0, 3, -1], False)
    creating: createInferReshape
    '''

    def __init__(self,
                 size,
                 batch_mode=False,
                 bigdl_type="float"):
        super(InferReshape, self).__init__(None, bigdl_type,
                                           size,
                                           batch_mode)


class JoinTable(Layer):

    '''
    It is a table module which takes a table of Tensors as input and
    outputs a Tensor by joining them together along the dimension `dimension`.


    The input to this layer is expected to be a tensor, or a batch of tensors;
    when using mini-batch, a batch of sample tensors will be passed to the layer and
    the user need to specify the number of dimensions of each sample tensor in the
    batch using `nInputDims`.


    :param dimension: to be join in this dimension
    :param nInputDims: specify the number of dimensions that this module will receiveIf it is more than the dimension of input tensors, the first dimensionwould be considered as batch size


    >>> joinTable = JoinTable(1, 1)
    creating: createJoinTable
    '''

    def __init__(self,
                 dimension,
                 n_input_dims,
                 bigdl_type="float"):
        super(JoinTable, self).__init__(None, bigdl_type,
                                        dimension,
                                        n_input_dims)


class L1Penalty(Layer):

    '''
    adds an L1 penalty to an input (for sparsity).
    L1Penalty is an inline module that in its forward propagation copies the input Tensor
    directly to the output, and computes an L1 loss of the latent state (input) and stores
    it in the module's loss field. During backward propagation: gradInput = gradOutput + gradLoss.


    :param l1weight:
    :param sizeAverage:
    :param provideOutput:


    >>> l1Penalty = L1Penalty(1, True, True)
    creating: createL1Penalty
    '''

    def __init__(self,
                 l1weight,
                 size_average=False,
                 provide_output=True,
                 bigdl_type="float"):
        super(L1Penalty, self).__init__(None, bigdl_type,
                                        l1weight,
                                        size_average,
                                        provide_output)


class LeakyReLU(Layer):

    '''
    It is a transfer module that applies LeakyReLU, which parameter negval sets the slope of the
    negative part: LeakyReLU is defined as: f(x) = max(0, x) + negval * min(0, x)


    :param negval: sets the slope of the negative partl
    :param inplace: if it is true, doing the operation in-place without using extra state memory


    >>> leakyReLU = LeakyReLU(1e-5, True)
    creating: createLeakyReLU
    '''

    def __init__(self,
                 negval=0.01,
                 inplace=False,
                 bigdl_type="float"):
        super(LeakyReLU, self).__init__(None, bigdl_type,
                                        negval,
                                        inplace)


class Log(Layer):

    '''
    Applies the log function element-wise to the input Tensor,
     thus outputting a Tensor of the same dimension.


    >>> log = Log()
    creating: createLog
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Log, self).__init__(None, bigdl_type)


class LogSigmoid(Layer):

    '''
    This class is a transform layer corresponding to the sigmoid function:
    f(x) = Log(1 / (1 + e ^^ (-x)))


    >>> logSigmoid = LogSigmoid()
    creating: createLogSigmoid
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(LogSigmoid, self).__init__(None, bigdl_type)


class LookupTable(Layer):

    '''
    a convolution of width 1, commonly used for word embeddings


    >>> lookupTable = LookupTable(1, 1, 1e-5, 1e-5, 1e-5, True)
    creating: createLookupTable
    '''

    def __init__(self,
                 n_index,
                 n_output,
                 padding_value=0.0,
                 max_norm=DOUBLEMAX,
                 norm_type=2.0,
                 should_scale_grad_by_freq=False,
                 bigdl_type="float"):
        super(LookupTable, self).__init__(None, bigdl_type,
                                          n_index,
                                          n_output,
                                          padding_value,
                                          max_norm,
                                          norm_type,
                                          should_scale_grad_by_freq)
    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class MM(Layer):

    '''
    Module to perform matrix multiplication on two mini-batch inputs, producing a mini-batch.


    :param trans_a: specifying whether or not transpose the first input matrix
    :param trans_b: specifying whether or not transpose the second input matrix


    >>> mM = MM(True, True)
    creating: createMM
    '''
    def __init__(self,
                 trans_a=False,
                 trans_b=False,
                 bigdl_type="float"):
        super(MM, self).__init__(None, bigdl_type,
                                 trans_a,
                                 trans_b)


class MV(Layer):

    '''
    It is a module to perform matrix vector multiplication on two mini-batch inputs,
    producing a mini-batch.


    :param trans: whether make matrix transpose before multiplication


    >>> mV = MV(True)
    creating: createMV
    '''

    def __init__(self,
                 trans=False,
                 bigdl_type="float"):
        super(MV, self).__init__(None, bigdl_type,
                                 trans)


class MapTable(Container):

    '''
    This class is a container for a single module which will be applied
    to all input elements. The member module is cloned as necessary to
    process all input elements.


    >>> mapTable = MapTable(Linear(100,10))
    creating: createLinear
    creating: createMapTable
    '''

    def __init__(self,
                 module,
                 bigdl_type="float"):
        super(MapTable, self).__init__(None, bigdl_type,
                                       module)

class MaskedSelect(Layer):

    '''
    Performs a torch.MaskedSelect on a Tensor. The mask is supplied as a tabular argument with
    the input on the forward and backward passes.

    >>> maskedSelect = MaskedSelect()
    creating: createMaskedSelect
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(MaskedSelect, self).__init__(None, bigdl_type)


class Max(Layer):

    '''
    Applies a max operation over dimension `dim`


   :param dim: max along this dimension
   :param num_input_dims: Optional. If in a batch model, set to the inputDims.


    >>> max = Max(1)
    creating: createMax
    '''

    def __init__(self,
                 dim,
                 num_input_dims=INTMIN,
                 bigdl_type="float"):
        super(Max, self).__init__(None, bigdl_type,
                                  dim,
                                  num_input_dims)


class Mean(Layer):

    '''
    It is a simple layer which applies a mean operation over the given dimension. When nInputDims
    is provided, the input will be considered as batches. Then the mean operation will be applied
    in (dimension + 1). The input to this layer is expected to be a tensor, or a batch of
    tensors; when using mini-batch, a batch of sample tensors will be passed to the layer and the
    user need to specify the number of dimensions of each sample tensor in the batch using
    nInputDims.


    :param dimension: the dimension to be applied mean operation
    :param n_input_dims: specify the number of dimensions that this module will receiveIf it is more than the dimension of input tensors, the first dimension would be consideredas batch size


    >>> mean = Mean(1, 1)
    creating: createMean
    '''

    def __init__(self,
                 dimension=1,
                 n_input_dims=-1,
                 bigdl_type="float"):
        super(Mean, self).__init__(None, bigdl_type,
                                   dimension,
                                   n_input_dims)


class Min(Layer):

    '''
    Applies a min operation over dimension `dim`.


    :param dim: min along this dimension
    :param num_input_dims: Optional. If in a batch model, set to the input_dim.


    >>> min = Min(1)
    creating: createMin
    '''

    def __init__(self,
                 dim,
                 num_input_dims=INTMIN,
                 bigdl_type="float"):
        super(Min, self).__init__(None, bigdl_type,
                                  dim,
                                  num_input_dims)


class MixtureTable(Layer):

    '''
    Creates a module that takes a table {gater, experts} as input and outputs the mixture of experts
    (a Tensor or table of Tensors) using a gater Tensor. When dim is provided, it specifies the
    dimension of the experts Tensor that will be interpolated (or mixed). Otherwise, the experts
    should take the form of a table of Tensors. This Module works for experts of dimension 1D or
    more, and for a 1D or 2D gater, i.e. for single examples or mini-batches.


    >>> mixtureTable = MixtureTable()
    creating: createMixtureTable
    >>> mixtureTable = MixtureTable(10)
    creating: createMixtureTable
    '''

    def __init__(self,
                 dim=INTMAX,
                 bigdl_type="float"):
        super(MixtureTable, self).__init__(None, bigdl_type, dim)


class Mul(Layer):

    '''
    Multiply a single scalar factor to the incoming data


    >>> mul = Mul()
    creating: createMul
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Mul, self).__init__(None, bigdl_type)

    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class MulConstant(Layer):

    '''
    Multiplies input Tensor by a (non-learnable) scalar constant.
    This module is sometimes useful for debugging purposes.


    :param scalar: scalar constant
    :param inplace: Can optionally do its operation in-place without using extra state memory


    >>> mulConstant = MulConstant(2.5)
    creating: createMulConstant
    '''

    def __init__(self,
                 scalar,
                 inplace=False,
                 bigdl_type="float"):
        super(MulConstant, self).__init__(None, bigdl_type,
                                          scalar,
                                          inplace)


class Narrow(Layer):

    '''
    Narrow is application of narrow operation in a module.
    The module further supports a negative length in order to handle inputs with an unknown size.

    >>> narrow = Narrow(1, 1, 1)
    creating: createNarrow
    '''

    def __init__(self,
                 dimension,
                 offset,
                 length=1,
                 bigdl_type="float"):
        super(Narrow, self).__init__(None, bigdl_type,
                                     dimension,
                                     offset,
                                     length)


class NarrowTable(Layer):

    '''
    Creates a module that takes a table as input and outputs the subtable starting at index
    offset having length elements (defaults to 1 element). The elements can be either
    a table or a Tensor. If `length` is negative, it means selecting the elements from the
    offset to element which located at the abs(`length`) to the last element of the input.


    :param offset: the start index of table
    :param length: the length want to select


    >>> narrowTable = NarrowTable(1, 1)
    creating: createNarrowTable
    '''

    def __init__(self,
                 offset,
                 length=1,
                 bigdl_type="float"):
        super(NarrowTable, self).__init__(None, bigdl_type,
                                          offset,
                                          length)


class Normalize(Layer):

    '''
    Normalizes the input Tensor to have unit L_p norm. The smoothing parameter eps prevents
    division by zero when the input contains all zero elements (default = 1e-10).
    p can be the max value of double


    >>> normalize = Normalize(1e-5, 1e-5)
    creating: createNormalize
    '''

    def __init__(self,
                 p,
                 eps=1e-10,
                 bigdl_type="float"):
        super(Normalize, self).__init__(None, bigdl_type,
                                        p,
                                        eps)


class PReLU(Layer):

    '''
    Applies parametric ReLU, which parameter varies the slope of the negative part.


    PReLU: f(x) = max(0, x) + a * min(0, x)


    nOutputPlane's default value is 0, that means using PReLU in shared version and has
    only one parameters.


    Notice: Please don't use weight decay on this.


    :param n_output_plane: input map number. Default is 0.


    >>> pReLU = PReLU(1)
    creating: createPReLU
    '''

    def __init__(self,
                 n_output_plane=0,
                 bigdl_type="float"):
        super(PReLU, self).__init__(None, bigdl_type,
                                    n_output_plane)

    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class Padding(Layer):

    '''
    This module adds pad units of padding to dimension dim of the input. If pad is negative,
    padding is added to the left, otherwise, it is added to the right of the dimension.


    The input to this layer is expected to be a tensor, or a batch of tensors;
    when using mini-batch, a batch of sample tensors will be passed to the layer and
    the user need to specify the number of dimensions of each sample tensor in the
    batch using n_input_dim.


    :param dim: the dimension to be applied padding operation
    :param pad: num of the pad units
    :param n_input_dim: specify the number of dimensions that this module will receiveIf it is more than the dimension of input tensors, the first dimensionwould be considered as batch size
    :param value: padding value


    >>> padding = Padding(1, 1, 1, 1e-5, 1)
    creating: createPadding
    '''

    def __init__(self,
                 dim,
                 pad,
                 n_input_dim,
                 value=0.0,
                 n_index=1,
                 bigdl_type="float"):
        super(Padding, self).__init__(None, bigdl_type,
                                      dim,
                                      pad,
                                      n_input_dim,
                                      value,
                                      n_index)


class PairwiseDistance(Layer):

    '''
    It is a module that takes a table of two vectors as input and outputs
    the distance between them using the p-norm.
    The input given in `forward(input)` is a [[Table]] that contains two tensors which
    must be either a vector (1D tensor) or matrix (2D tensor). If the input is a vector,
    it must have the size of `inputSize`. If it is a matrix, then each row is assumed to be
    an input sample of the given batch (the number of rows means the batch size and
    the number of columns should be equal to the `inputSize`).

    :param norm: the norm of distance


    >>> pairwiseDistance = PairwiseDistance(2)
    creating: createPairwiseDistance
    '''

    def __init__(self,
                 norm=2,
                 bigdl_type="float"):
        super(PairwiseDistance, self).__init__(None, bigdl_type,
                                               norm)


class ParallelTable(Container):

    '''
    It is a container module that applies the i-th member module to the i-th
    input, and outputs an output in the form of Table


    >>> parallelTable = ParallelTable()
    creating: createParallelTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(ParallelTable, self).__init__(None, bigdl_type)


class Power(Layer):

    '''
    Apply an element-wise power operation with scale and shift.
    f(x) = (shift + scale * x)^power^

    :param power: the exponent.
    :param scale: Default is 1.
    :param shift: Default is 0.


    >>> power = Power(1e-5)
    creating: createPower
    '''

    def __init__(self,
                 power,
                 scale=1.0,
                 shift=0.0,
                 bigdl_type="float"):
        super(Power, self).__init__(None, bigdl_type,
                                    power,
                                    scale,
                                    shift)


class RReLU(Layer):

    '''
    Applies the randomized leaky rectified linear unit (RReLU) element-wise to the input Tensor,
    thus outputting a Tensor of the same dimension. Informally the RReLU is also known as
    'insanity' layer. RReLU is defined as:
```
        f(x) = max(0,x) + a * min(0, x) where a ~ U(l, u).
```

    In training mode negative inputs are multiplied by a factor a drawn from a uniform random
    distribution U(l, u).


    In evaluation mode a RReLU behaves like a LeakyReLU with a constant mean factor
        a = (l + u) / 2.


    By default, l = 1/8 and u = 1/3. If l == u a RReLU effectively becomes a LeakyReLU.


    Regardless of operating in in-place mode a RReLU will internally allocate an input-sized
    noise tensor to store random factors for negative inputs.


    The backward() operation assumes that forward() has been called before.


    For reference see [Empirical Evaluation of Rectified Activations in Convolutional Network](
    http://arxiv.org/abs/1505.00853).


    :param lower: lower boundary of uniform random distribution
    :param upper: upper boundary of uniform random distribution
    :param inplace: optionally do its operation in-place without using extra state memory


    >>> rReLU = RReLU(1e-5, 1e5, True)
    creating: createRReLU
    '''

    def __init__(self,
                 lower=1.0/8,
                 upper=1.0/3,
                 inplace=False,
                 bigdl_type="float"):
        super(RReLU, self).__init__(None, bigdl_type,
                                    lower,
                                    upper,
                                    inplace)


class ReLU6(Layer):

    '''
    Same as ReLU except that the rectifying function f(x) saturates at x = 6


    :param inplace: either True = in-place or False = keeping separate state


    >>> reLU6 = ReLU6(True)
    creating: createReLU6
    '''

    def __init__(self,
                 inplace=False,
                 bigdl_type="float"):
        super(ReLU6, self).__init__(None, bigdl_type,
                                    inplace)


class Replicate(Layer):

    '''
    Replicate repeats input `nFeatures` times along its `dim` dimension.
    Notice: No memory copy, it set the stride along the `dim`-th dimension to zero.


    :param n_features: replicate times.
    :param dim: dimension to be replicated.
    :param n_dim: specify the number of non-batch dimensions.


    >>> replicate = Replicate(2)
    creating: createReplicate
    '''
    def __init__(self,
                 n_features,
                 dim=1,
                 n_dim=INTMAX,
                 bigdl_type="float"):
        super(Replicate, self).__init__(None, bigdl_type,
                                        n_features,
                                        dim,
                                        n_dim)


class RoiPooling(Layer):

    '''
    Region of interest pooling
    The RoIPooling uses max pooling to convert the features inside any valid region of interest
    into a small feature map with a fixed spatial extent of pooledH * pooledW (e.g., 7 * 7)
    an RoI is a rectangular window into a conv feature map.
    Each RoI is defined by a four-tuple (x1, y1, x2, y2) that specifies its
    top-left corner (x1, y1) and its bottom-right corner (x2, y2).
    RoI max pooling works by dividing the h * w RoI window into an pooledH * pooledW grid of
    sub-windows of approximate size h/H * w/W and then max-pooling the values in each sub-window
    into the corresponding output grid cell.
    Pooling is applied independently to each feature map channel


    :param pooled_w:      spatial extent in width
    :param pooled_h:      spatial extent in height
    :param spatial_scale: spatial scale


    >>> roiPooling = RoiPooling(1, 1, 1e-5)
    creating: createRoiPooling
    '''

    def __init__(self,
                 pooled_w,
                 pooled_h,
                 spatial_scale,
                 bigdl_type="float"):
        super(RoiPooling, self).__init__(None, bigdl_type,
                                         pooled_w,
                                         pooled_h,
                                         spatial_scale)


class Scale(Layer):

    '''
    Scale is the combination of CMul and CAdd
    Computes the elementwise product of input and weight, with the shape of the weight "expand" to
    match the shape of the input.
    Similarly, perform a expand cdd bias and perform an elementwise add


    :param size: size of weight and bias


    >>> scale = Scale([1,2])
    creating: createScale
    '''

    def __init__(self,
                 size,
                 bigdl_type="float"):
        super(Scale, self).__init__(None, bigdl_type,
                                    size)


class SelectTable(Layer):

    '''
    Creates a module that takes a table as input and outputs the element at index `index`
    (positive or negative). This can be either a table or a Tensor.
    The gradients of the non-index elements are zeroed Tensors of the same size.
    This is true regardless of the depth of the encapsulated Tensor as the function used
    internally to do so is recursive.


    :param dimension: the dimension to be selected


    >>> selectTable = SelectTable(1)
    creating: createSelectTable
    '''

    def __init__(self,
                 dimension,
                 bigdl_type="float"):
        super(SelectTable, self).__init__(None, bigdl_type,
                                          dimension)


class Sigmoid(Layer):

    '''
    Applies the Sigmoid function element-wise to the input Tensor,
    thus outputting a Tensor of the same dimension.

    >>> sigmoid = Sigmoid()
    creating: createSigmoid
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Sigmoid, self).__init__(None, bigdl_type)


class SoftMax(Layer):

    '''
    Applies the SoftMax function to an n-dimensional input Tensor, rescaling them so that the
    elements of the n-dimensional output Tensor lie in the range (0, 1) and sum to 1.
    Softmax is defined as: f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
    where shift = max_i(x_i).


    >>> softMax = SoftMax()
    creating: createSoftMax
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(SoftMax, self).__init__(None, bigdl_type)


class SoftMin(Layer):

    '''
    Applies the SoftMin function to an n-dimensional input Tensor, rescaling them so that the
    elements of the n-dimensional output Tensor lie in the range (0,1) and sum to 1.
    Softmin is defined as: f_i(x) = exp(-x_i - shift) / sum_j exp(-x_j - shift)
    where shift = max_i(-x_i).


    >>> softMin = SoftMin()
    creating: createSoftMin
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(SoftMin, self).__init__(None, bigdl_type)


class SoftPlus(Layer):

    '''
    Apply the SoftPlus function to an n-dimensional input tensor.
    SoftPlus function: f_i(x) = 1/beta * log(1 + exp(beta * x_i))


    :param beta: Controls sharpness of transfer function


    >>> softPlus = SoftPlus(1e-5)
    creating: createSoftPlus
    '''

    def __init__(self,
                 beta=1.0,
                 bigdl_type="float"):
        super(SoftPlus, self).__init__(None, bigdl_type,
                                       beta)


class SoftShrink(Layer):

    '''
    Apply the soft shrinkage function element-wise to the input Tensor


    SoftShrinkage operator:
```
           | x - lambda, if x >  lambda
    f(x) = | x + lambda, if x < -lambda
           | 0, otherwise
```

    :param the_lambda: lambda, default is 0.5


    >>> softShrink = SoftShrink(1e-5)
    creating: createSoftShrink
    '''

    def __init__(self,
                 the_lambda=0.5,
                 bigdl_type="float"):
        super(SoftShrink, self).__init__(None, bigdl_type,
                                         the_lambda)


class SoftSign(Layer):

    '''
    Apply SoftSign function to an n-dimensional input Tensor.


    SoftSign function: f_i(x) = x_i / (1+|x_i|)


    >>> softSign = SoftSign()
    creating: createSoftSign
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(SoftSign, self).__init__(None, bigdl_type)


class SpatialDilatedConvolution(Layer):

    '''
    Apply a 2D dilated convolution over an input image.


    The input tensor is expected to be a 3D or 4D(with batch) tensor.


    If input is a 3D tensor nInputPlane x height x width,
    owidth  = floor(width + 2 * padW - dilationW * (kW-1) - 1) / dW + 1
    oheight = floor(height + 2 * padH - dilationH * (kH-1) - 1) / dH + 1


    Reference Paper: Yu F, Koltun V. Multi-scale context aggregation by dilated convolutions[J].
    arXiv preprint arXiv:1511.07122, 2015.


    :param n_input_plane: The number of expected input planes in the image given into forward().
    :param n_output_plane: The number of output planes the convolution layer will produce.
    :param kw: The kernel width of the convolution.
    :param kh: The kernel height of the convolution.
    :param dw: The step of the convolution in the width dimension. Default is 1.
    :param dh: The step of the convolution in the height dimension. Default is 1.
    :param pad_w: The additional zeros added per width to the input planes. Default is 0.
    :param pad_h: The additional zeros added per height to the input planes. Default is 0.
    :param dilation_w: The number of pixels to skip. Default is 1.
    :param dilation_h: The number of pixels to skip. Default is 1.
    :param init_method: Init method, Default, Xavier.


    >>> spatialDilatedConvolution = SpatialDilatedConvolution(1, 1, 1, 1)
    creating: createSpatialDilatedConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 kw,
                 kh,
                 dw=1,
                 dh=1,
                 pad_w=0,
                 pad_h=0,
                 dilation_w=1,
                 dilation_h=1,
                 init_method='default',
                 bigdl_type="float"):
        super(SpatialDilatedConvolution, self).__init__(None, bigdl_type,
                                                        n_input_plane,
                                                        n_output_plane,
                                                        kw,
                                                        kh,
                                                        dw,
                                                        dh,
                                                        pad_w,
                                                        pad_h,
                                                        dilation_w,
                                                        dilation_h,
                                                        init_method)

    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class SpatialFullConvolution(Layer):
    '''
    Apply a 2D full convolution over an input image.
    The input tensor is expected to be a 3D or 4D(with batch) tensor. Note that instead
    of setting adjW and adjH, SpatialFullConvolution[Table, T] also accepts a table input
    with two tensors: T(convInput, sizeTensor) where convInput is the standard input tensor,
    and the size of sizeTensor is used to set the size of the output (will ignore the adjW and
    adjH values used to construct the module). This module can be used without a bias by setting
    parameter noBias = true while constructing the module.


    If input is a 3D tensor nInputPlane x height x width,
    owidth  = (width  - 1) * dW - 2*padW + kW + adjW
    oheight = (height - 1) * dH - 2*padH + kH + adjH


    Other frameworks call this operation "In-network Upsampling", "Fractionally-strided convolution",
    "Backwards Convolution," "Deconvolution", or "Upconvolution."


    Reference Paper: Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic
    segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
    2015: 3431-3440.


    :param nInputPlane: The number of expected input planes in the image given into forward()
    :param nOutputPlane: The number of output planes the convolution layer will produce.
    :param kW: The kernel width of the convolution.
    :param kH: The kernel height of the convolution.
    :param dW: The step of the convolution in the width dimension. Default is 1.
    :param dH: The step of the convolution in the height dimension. Default is 1.
    :param padW: The additional zeros added per width to the input planes. Default is 0.
    :param padH: The additional zeros added per height to the input planes. Default is 0.
    :param adjW: Extra width to add to the output image. Default is 0.
    :param adjH: Extra height to add to the output image. Default is 0.
    :param nGroup: Kernel group number.
    :param noBias: If bias is needed.
    :param initMethod: Init method, Default, Xavier, Bilinear.


    >>> spatialFullConvolution = SpatialFullConvolution(1, 1, 1, 1)
    creating: createSpatialFullConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 kw,
                 kh,
                 dw=1,
                 dh=1,
                 pad_w=0,
                 pad_h=0,
                 adj_w=0,
                 adj_h=0,
                 n_group=1,
                 no_bias=False,
                 init_method='default',
                 bigdl_type="float"):
        super(SpatialFullConvolution, self).__init__(None, bigdl_type,
                                                     n_input_plane,
                                                     n_output_plane,
                                                     kw,
                                                     kh,
                                                     dw,
                                                     dh,
                                                     pad_w,
                                                     pad_h,
                                                     adj_w,
                                                     adj_h,
                                                     n_group,
                                                     no_bias,
                                                     init_method)
    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class SpatialShareConvolution(Layer):

    '''

    >>> spatialShareConvolution = SpatialShareConvolution(1, 1, 1, 1)
    creating: createSpatialShareConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 kernel_w,
                 kernel_h,
                 stride_w=1,
                 stride_h=1,
                 pad_w=0,
                 pad_h=0,
                 n_group=1,
                 propagate_back=True,
                 init_method='default',
                 bigdl_type="float"):
        super(SpatialShareConvolution, self).__init__(None, bigdl_type,
                                                      n_input_plane,
                                                      n_output_plane,
                                                      kernel_w,
                                                      kernel_h,
                                                      stride_w,
                                                      stride_h,
                                                      pad_w,
                                                      pad_h,
                                                      n_group,
                                                      propagate_back,
                                                      init_method)
    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class VolumetricConvolution(Layer):

    '''
    Applies a 3D convolution over an input image composed of several input planes. The input tensor
    in forward(input) is expected to be a 4D tensor (nInputPlane x time x height x width).

    :param n_input_plane: The number of expected input planes in the image given into forward()
    :param n_output_plane: The number of output planes the convolution layer will produce.
    :param k_t: The kernel size of the convolution in time
    :param k_w: The kernel width of the convolution
    :param k_h: The kernel height of the convolution
    :param d_t: The step of the convolution in the time dimension. Default is 1
    :param d_w: The step of the convolution in the width dimension. Default is 1
    :param d_h: The step of the convolution in the height dimension. Default is 1
    :param pad_t: Additional zeros added to the input plane data on both sides of time axis.Default is 0. (kT-1)/2 is often used here.
    :param pad_w: The additional zeros added per width to the input planes.
    :param pad_h: The additional zeros added per height to the input planes.
    :param with_bias: whether with bias
    :param init_method: Init method, Default, Xavier, Bilinear.


    >>> volumetricConvolution = VolumetricConvolution(6, 12, 5, 5, 5, 1, 1, 1)
    creating: createVolumetricConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 k_t,
                 k_w,
                 k_h,
                 d_t=1,
                 d_w=1,
                 d_h=1,
                 pad_t=0,
                 pad_w=0,
                 pad_h=0,
                 with_bias=True,
                 init_method="default",
                 bigdl_type="float"):
        super(VolumetricConvolution, self).__init__(None, bigdl_type,
                                                    n_input_plane,
                                                    n_output_plane,
                                                    k_t,
                                                    k_w,
                                                    k_h,
                                                    d_t,
                                                    d_w,
                                                    d_h,
                                                    pad_t,
                                                    pad_w,
                                                    pad_h,
                                                    with_bias,
                                                    init_method)

    def set_init_method(self, weight_init_method = None, bias_init_method = None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)


class VolumetricMaxPooling(Layer):

    '''
    Applies 3D max-pooling operation in kTxkWxkH regions by step size dTxdWxdH steps.
    The number of output features is equal to the number of input planes / dT.
    The input can optionally be padded with zeros. Padding should be smaller than
    half of kernel size. That is, padT < kT/2, padW < kW/2 and padH < kH/2

    :param k_t: The kernel size
    :param k_w: The kernel width
    :param k_h: The kernel height
    :param d_t: The step in the time dimension
    :param d_w: The step in the width dimension
    :param d_h: The step in the height dimension
    :param pad_t: The padding in the time dimension
    :param pad_w: The padding in the width dimension
    :param pad_h: The padding in the height dimension


    >>> volumetricMaxPooling = VolumetricMaxPooling(5, 5, 5, 1, 1, 1)
    creating: createVolumetricMaxPooling
    '''

    def __init__(self,
                 k_t,
                 k_w,
                 k_h,
                 d_t,
                 d_w,
                 d_h,
                 pad_t=0,
                 pad_w=0,
                 pad_h=0,
                 bigdl_type="float"):
        super(VolumetricMaxPooling, self).__init__(None, bigdl_type,
                                                    k_t,
                                                    k_w,
                                                    k_h,
                                                    d_t,
                                                    d_w,
                                                    d_h,
                                                    pad_t,
                                                    pad_w,
                                                    pad_h)


class SpatialZeroPadding(Layer):

    '''
    Each feature map of a given input is padded with specified number of zeros.
    If padding values are negative, then input is cropped.

    :param padLeft: pad left position
    :param padRight: pad right position
    :param padTop: pad top position
    :param padBottom: pad bottom position


    >>> spatialZeroPadding = SpatialZeroPadding(1, 1, 1, 1)
    creating: createSpatialZeroPadding
    '''

    def __init__(self,
                 pad_left,
                 pad_right,
                 pad_top,
                 pad_bottom,
                 bigdl_type="float"):
        super(SpatialZeroPadding, self).__init__(None, bigdl_type,
                                                 pad_left,
                                                 pad_right,
                                                 pad_top,
                                                 pad_bottom)


class SplitTable(Layer):

    '''
    Creates a module that takes a Tensor as input and
    outputs several tables, splitting the Tensor along
    the specified dimension `dimension`.


    The input to this layer is expected to be a tensor, or a batch of tensors;
    when using mini-batch, a batch of sample tensors will be passed to the layer and
    the user need to specify the number of dimensions of each sample tensor in a
    batch using `nInputDims`.


    :param dimension: to be split along this dimension
    :param n_input_dims: specify the number of dimensions that this module will receiveIf it is more than the dimension of input tensors, the first dimensionwould be considered as batch size


    >>> splitTable = SplitTable(1, 1)
    creating: createSplitTable
    '''

    def __init__(self,
                 dimension,
                 n_input_dims=-1,
                 bigdl_type="float"):
        super(SplitTable, self).__init__(None, bigdl_type,
                                         dimension,
                                         n_input_dims)


class Sqrt(Layer):

    '''
    Apply an element-wise sqrt operation.


    >>> sqrt = Sqrt()
    creating: createSqrt
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Sqrt, self).__init__(None, bigdl_type)


class Square(Layer):

    '''
    Apply an element-wise square operation.

    >>> square = Square()
    creating: createSquare
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Square, self).__init__(None, bigdl_type)


class Squeeze(Layer):

    '''
    Delete singleton all dimensions or a specific dim.


    :param dim: Optional. The dimension to be delete. Default: delete all dimensions.
    :param num_input_dims: Optional. If in a batch model, set to the inputDims.




    >>> squeeze = Squeeze(1)
    creating: createSqueeze
    '''

    def __init__(self,
                 dim,
                 num_input_dims=INTMIN,
                 bigdl_type="float"):
        super(Squeeze, self).__init__(None, bigdl_type,
                                      dim,
                                      num_input_dims)


class Sum(Layer):

    '''
    It is a simple layer which applies a sum operation over the given dimension.
    When nInputDims is provided, the input will be considered as a batches.
    Then the sum operation will be applied in (dimension + 1)
    The input to this layer is expected to be a tensor, or a batch of tensors;
    when using mini-batch, a batch of sample tensors will be passed to the layer and
    the user need to specify the number of dimensions of each sample tensor in the
    batch using `nInputDims`.


    :param dimension: the dimension to be applied sum operation
    :param n_input_dims: specify the number of dimensions that this module will receiveIf it is more than the dimension of input tensors, the first dimensionwould be considered as batch size
    :param size_average: default is false, if it is true, it will return the mean instead


    >>> sum = Sum(1, 1, True)
    creating: createSum
    '''

    def __init__(self,
                 dimension=1,
                 n_input_dims=-1,
                 size_average=False,
                 bigdl_type="float"):
        super(Sum, self).__init__(None, bigdl_type,
                                  dimension,
                                  n_input_dims,
                                  size_average)


class TanhShrink(Layer):

    '''
    A simple layer for each element of the input tensor, do the following operation
    during the forward process:
    [f(x) = tanh(x) - 1]


    >>> tanhShrink = TanhShrink()
    creating: createTanhShrink
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(TanhShrink, self).__init__(None, bigdl_type)


class Threshold(Layer):

    '''
    Threshold input Tensor.
    If values in the Tensor smaller than th, then replace it with v


    :param th: the threshold to compare with
    :param v: the value to replace with
    :param ip: inplace mode


    >>> threshold = Threshold(1e-5, 1e-5, True)
    creating: createThreshold
    '''

    def __init__(self,
                 th=1e-6,
                 v=0.0,
                 ip=False,
                 bigdl_type="float"):
        super(Threshold, self).__init__(None, bigdl_type,
                                        th,
                                        v,
                                        ip)


class Unsqueeze(Layer):

    '''
    Create an Unsqueeze layer.  Insert singleton dim (i.e., dimension 1) at position pos.
    For an input with dim = input.dim(),
    there are dim + 1 possible positions to insert the singleton dimension.


    :param pos: The position will be insert singleton.
    :param num_input_dims: Optional. If in a batch model, set to the inputDim


    >>> unsqueeze = Unsqueeze(1, 1)
    creating: createUnsqueeze
    '''

    def __init__(self,
                 pos,
                 num_input_dims=INTMIN,
                 bigdl_type="float"):
        super(Unsqueeze, self).__init__(None, bigdl_type,
                                        pos,
                                        num_input_dims)


class Reshape(Layer):
    '''
    The forward(input) reshape the input tensor into a size(0) * size(1) * ... tensor, taking the
    elements row-wise.


    :param size: the reshape size


    >>> reshape = Reshape([1, 28, 28])
    creating: createReshape
    >>> reshape = Reshape([1, 28, 28], False)
    creating: createReshape
    '''

    def __init__(self, size, batch_mode=None, bigdl_type="float"):
        super(Reshape, self).__init__(None, bigdl_type, size, batch_mode)


class BiRecurrent(Container):
    '''
    Create a Bidirectional recurrent layer


    :param merge: merge layer


    >>> biRecurrent = BiRecurrent(CAddTable())
    creating: createCAddTable
    creating: createBiRecurrent
    >>> biRecurrent = BiRecurrent()
    creating: createBiRecurrent
    '''

    def __init__(self,
                 merge=None,
                 bigdl_type="float"):
        super(BiRecurrent, self).__init__(None, bigdl_type, merge)


class ConcatTable(Container):
    '''
    ConcateTable is a container module like Concate. Applies an input
    to each member module, input can be a tensor or a table.


    ConcateTable usually works with CAddTable and CMulTable to
    implement element wise add/multiply on outputs of two modules.


    >>> concatTable = ConcatTable()
    creating: createConcatTable
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(ConcatTable, self).__init__(None, bigdl_type)


class Identity(Layer):
    '''
    Identity just return the input to output.
    It's useful in same parallel container to get an origin input.


    >>> identity = Identity()
    creating: createIdentity
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Identity, self).__init__(None, bigdl_type)


class Reverse(Layer):
    '''
    Reverse the input w.r.t given dimension.
    The input can be a Tensor or Table.


    :param dim:


    >>> reverse = Reverse()
    creating: createReverse
    '''

    def __init__(self,
                 dimension=1,
                 bigdl_type="float"):
        super(Reverse, self).__init__(None, bigdl_type,
                                      dimension)


class Transpose(Layer):
    '''
    Transpose input along specified dimensions


    :param permutations: dimension pairs that need to swap


    >>> transpose = Transpose([(1,2)])
    creating: createTranspose
    '''

    def __init__(self,
                 permutations,
                 bigdl_type="float"):
        super(Transpose, self).__init__(None, bigdl_type,
                                        permutations)


class SpatialContrastiveNormalization(Layer):
    '''
    Subtractive + divisive contrast normalization.


    :param n_input_plane:
    :param kernel:
    :param threshold:
    :param thresval:


    >>> kernel = np.ones([9,9]).astype("float32")
    >>> spatialContrastiveNormalization = SpatialContrastiveNormalization(1, kernel)
    creating: createSpatialContrastiveNormalization
    >>> spatialContrastiveNormalization = SpatialContrastiveNormalization()
    creating: createSpatialContrastiveNormalization
    '''

    def __init__(self,
                 n_input_plane=1,
                 kernel=None,
                 threshold=1e-4,
                 thresval=1e-4,
                 bigdl_type="float"):
        super(SpatialContrastiveNormalization, self).__init__(None, bigdl_type,
                                                              n_input_plane,
                                                              JTensor.from_ndarray(kernel),
                                                              threshold,
                                                              thresval)


class SpatialConvolutionMap(Layer):
    '''
    This class is a generalization of SpatialConvolution.
    It uses a generic connection table between input and output features.
    The SpatialConvolution is equivalent to using a full connection table.


    >>> ct = np.ones([9,9]).astype("float32")
    >>> spatialConvolutionMap = SpatialConvolutionMap(ct, 9, 9)
    creating: createSpatialConvolutionMap
    '''

    def __init__(self,
                 conn_table,
                 kw,
                 kh,
                 dw=1,
                 dh=1,
                 pad_w=0,
                 pad_h=0,
                 bigdl_type="float"):
        super(SpatialConvolutionMap, self).__init__(None, bigdl_type,
                                                    JTensor.from_ndarray(conn_table),
                                                    kw,
                                                    kh,
                                                    dw,
                                                    dh,
                                                    pad_w,
                                                    pad_h)


class SpatialDivisiveNormalization(Layer):
    '''
    Applies a spatial division operation on a series of 2D inputs using kernel for
    computing the weighted average in a neighborhood. The neighborhood is defined for
    a local spatial region that is the size as kernel and across all features. For
    an input image, since there is only one feature, the region is only spatial. For
    an RGB image, the weighted average is taken over RGB channels and a spatial region.


    If the kernel is 1D, then it will be used for constructing and separable 2D kernel.
    The operations will be much more efficient in this case.


    The kernel is generally chosen as a gaussian when it is believed that the correlation
    of two pixel locations decrease with increasing distance. On the feature dimension,
    a uniform average is used since the weighting across features is not known.




    :param nInputPlane: number of input plane, default is 1.
    :param kernel: kernel tensor, default is a 9 x 9 tensor.
    :param threshold: threshold
    :param thresval: threshhold value to replace withif data is smaller than theshold


    >>> kernel = np.ones([9,9]).astype("float32")
    >>> spatialDivisiveNormalization = SpatialDivisiveNormalization(2,kernel)
    creating: createSpatialDivisiveNormalization
    >>> spatialDivisiveNormalization = SpatialDivisiveNormalization()
    creating: createSpatialDivisiveNormalization
    '''

    def __init__(self,
                 n_input_plane=1,
                 kernel=None,
                 threshold=1e-4,
                 thresval=1e-4,
                 bigdl_type="float"):
        super(SpatialDivisiveNormalization, self).__init__(None, bigdl_type,
                                                           n_input_plane,
                                                           JTensor.from_ndarray(kernel),
                                                           threshold,
                                                           thresval)


class SpatialSubtractiveNormalization(Layer):
    '''
    Applies a spatial subtraction operation on a series of 2D inputs using kernel for
    computing the weighted average in a neighborhood. The neighborhood is defined for
    a local spatial region that is the size as kernel and across all features. For a
    an input image, since there is only one feature, the region is only spatial. For
    an RGB image, the weighted average is taken over RGB channels and a spatial region.


    If the kernel is 1D, then it will be used for constructing and separable 2D kernel.
    The operations will be much more efficient in this case.


    The kernel is generally chosen as a gaussian when it is believed that the correlation
    of two pixel locations decrease with increasing distance. On the feature dimension,
    a uniform average is used since the weighting across features is not known.


    :param n_input_plane: number of input plane, default is 1.
    :param kernel: kernel tensor, default is a 9 x 9 tensor.


    >>> kernel = np.ones([9,9]).astype("float32")
    >>> spatialSubtractiveNormalization = SpatialSubtractiveNormalization(2,kernel)
    creating: createSpatialSubtractiveNormalization
    >>> spatialSubtractiveNormalization = SpatialSubtractiveNormalization()
    creating: createSpatialSubtractiveNormalization
    '''

    def __init__(self,
                 n_input_plane=1,
                 kernel=None,
                 bigdl_type="float"):
        super(SpatialSubtractiveNormalization, self).__init__(None, bigdl_type,
                                                              n_input_plane,
                                                              JTensor.from_ndarray(kernel))

class Const(Model):
    '''
    Return a constant tensor defined by value when forward
    '''

    def __init__(self, value, bigdl_type="float"):
        super(Const, self).__init__(None, bigdl_type, JTensor.from_ndarray(value))

class Fill(Model):
    '''
    Return a constant tensor defined by value when forward
    '''

    def __init__(self, value, bigdl_type="float"):
        super(Fill, self).__init__(None, bigdl_type, value)

class Pack(Model):
    '''
    Stacks a list of n-dimensional tensors into one (n+1)-dimensional tensor.
    '''

    def __init__(self, dimension, bigdl_type="float"):
        super(Pack, self).__init__(None, bigdl_type, dimension)

class Shape(Model):
    '''
    Given input, return the shape of this input as a 1-D tensor
    '''

    def __init__(self, bigdl_type="float"):
        super(Shape, self).__init__(None, bigdl_type)

class SplitAndSelect(Model):
    '''
    First split the tensor along the [[dimension]] into [[numSplit]] sub tensors,
    then select the [[index]]th one
    '''

    def __init__(self, dimension, index, num_split, bigdl_type="float"):
        super(SplitAndSelect, self).__init__(None, bigdl_type, dimension, index, num_split)

class StrideSlice(Model):
    '''
    Extracts a strided slice from a tensor.
    '''

    def __init__(self):
        raise Exception('StrideSlice is not supported in python yet')



def _test():
    import doctest
    from pyspark import SparkContext
    from bigdl.nn import layer
    from bigdl.util.common import init_engine
    from bigdl.util.common import create_spark_conf
    globs = layer.__dict__.copy()
    sc = SparkContext(master="local[4]", appName="test layer",
                      conf=create_spark_conf())
    globs['sc'] = sc
    init_engine()

    (failure_count, test_count) = doctest.testmod(globs=globs,
                                                  optionflags=doctest.ELLIPSIS)
    if failure_count:
        exit(-1)

if __name__ == "__main__":
    _test()
