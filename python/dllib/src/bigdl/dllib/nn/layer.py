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
import importlib

import numpy as np
import six

from bigdl.dllib.utils.common import JTensor
from bigdl.dllib.utils.common import JavaValue
from bigdl.dllib.utils.common import callBigDlFunc
from bigdl.dllib.utils.common import callJavaFunc
from bigdl.dllib.utils.common import get_spark_context
from bigdl.dllib.utils.common import to_list
from bigdl.dllib.utils.common import INTMAX, INTMIN, DOUBLEMAX
from bigdl.dllib.utils.common import get_activation_by_name
from bigdl.dllib.optim.optimizer import L1Regularizer, L2Regularizer, L1L2Regularizer
from py4j.java_gateway import JavaObject
from pyspark.rdd import RDD
from bigdl.dllib.feature.transform.vision.image import ImageFrame
from bigdl.dllib.feature.dataset.dataset import DataSet
from bigdl.dllib.utils.log4Error import *


if sys.version >= '3':
    long = int
    unicode = str


class Node(JavaValue):
    """
    Represent a node in a graph. The connections between nodes are directed.
    """

    def __init__(self, jvalue, bigdl_type, *args):
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), *args)
        self.bigdl_type = bigdl_type

    @classmethod
    def of(cls, jvalue, bigdl_type="float"):
        return Node(jvalue, bigdl_type)

    def element(self):
        return Layer.of(self.value.element())

    def remove_pre_edges(self):
        callJavaFunc(self.value.removePreEdges)

    def remove_next_edges(self):
        callJavaFunc(self.value.removeNextEdges)


class SharedStaticUtils():
    @staticmethod
    def of(jvalue, bigdl_type="float"):
        """
        Create a Python Layer base on the given java value and the real type.
        :param jvalue: Java object create by Py4j
        :return: A Python Layer
        """

        def get_py_name(jclass_name):
            if jclass_name == "StaticGraph" or jclass_name == "DynamicGraph":
                return "Model"
            elif jclass_name == "Input":
                return "Layer"
            else:
                return jclass_name

        jname = callBigDlFunc(bigdl_type,
                              "getRealClassNameOfJValue",
                              jvalue)

        jpackage_name = ".".join(jname.split(".")[:-1])
        pclass_name = get_py_name(jname.split(".")[-1])

        if "com.intel.analytics.bigdl.dllib.keras.Model" == jname or \
                "com.intel.analytics.bigdl.dllib.keras.Sequential" == jname:
            base_module = importlib.import_module('bigdl.dllib.keras.layers.topology')
        elif "com.intel.analytics.bigdl.dllib.keras" == jpackage_name:
            base_module = importlib.import_module('bigdl.dllib.keras.layers.layer')
        else:
            base_module = importlib.import_module('bigdl.dllib.nn.layer')

        realClassName = "Layer"  # The top base class
        if pclass_name in dir(base_module):
            realClassName = pclass_name
        module = getattr(base_module, realClassName)
        jvalue_creator = getattr(module, "from_jvalue")
        model = jvalue_creator(jvalue, bigdl_type)
        return model


class Layer(JavaValue, SharedStaticUtils):
    """
    Layer is the basic component of a neural network
    and it's also the base class of layers.
    Layer can connect to others to construct a complex neural network.
    """

    def __init__(self, jvalue, bigdl_type, *args):
        if (jvalue):
            invalidInputError((type(jvalue) == JavaObject),
                              f"jvalue type ${type(jvalue)} doesn't match"
                              f" JavaObject ${JavaObject}")
            self.value = jvalue
        else:
            self.value = callBigDlFunc(
                bigdl_type, self.jvm_class_constructor(), *args)
        self.bigdl_type = bigdl_type

    def set_running_mean(self, running_mean):
        """
        Set the running mean of the layer.
        Only use this method for a BatchNormalization layer.
        :param running_mean: a Numpy array.
        """
        callBigDlFunc(self.bigdl_type, "setRunningMean",
                      self.value, JTensor.from_ndarray(running_mean))
        return self

    def set_running_std(self, running_std):
        """
        Set the running variance of the layer.
        Only use this method for a BatchNormalization layer.
        :param running_std: a Numpy array.
        """
        callBigDlFunc(self.bigdl_type, "setRunningStd",
                      self.value, JTensor.from_ndarray(running_std))
        return self

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

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Layer(jvalue=jvalue, bigdl_type=bigdl_type)
        model.value = jvalue
        return model

    def set_name(self, name):
        """
        Give this model a name. There would be a generated name
        consist of class name and UUID if user doesn't set it.
        """
        callJavaFunc(self.value.setName, name)
        return self

    def name(self):
        """
        Name of this layer
        """
        return callJavaFunc(self.value.getName)

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
        """
        :param input: ndarray or list of ndarray or JTensor or list of JTensor.
        :return: (list of JTensor, isTable)
        """

        def to_jtensor(i):
            if isinstance(i, np.ndarray):
                return JTensor.from_ndarray(i)
            elif isinstance(i, JTensor):
                return i
            else:
                invalidInputError(False, "Error unknown input type %s" % type(i))

        def check_list(input):
            if type(input) is list:
                if len(input) == 0:
                    invalidInputError(False, 'Error when checking: empty input')
                return list(map(lambda i: check_list(i), input))
            else:
                return to_jtensor(input)

        if type(input) is list:
            if len(input) == 0:
                invalidInputError(False, 'Error when checking: empty input')
            return list(map(lambda i: check_list(i), input)), True
        else:
            return [to_jtensor(input)], False

    @staticmethod
    def convert_output(output):
        if type(output) is JTensor:
            return output.to_ndarray()
        elif (len(output) == 1):
            return output[0].to_ndarray()
        else:
            return [x.to_ndarray() for x in output]

    def forward(self, input):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        Takes an input object, and computes the corresponding output of the module

        :param input: ndarray or list of ndarray
        :param input: ndarray or list of ndarray or JTensor or list of JTensor.
        :return: ndarray or list of ndarray
        """
        jinput, input_is_table = self.check_input(input)
        output = callBigDlFunc(self.bigdl_type,
                               "modelForward",
                               self.value,
                               jinput,
                               input_is_table)
        return self.convert_output(output)

    def backward(self, input, grad_output):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        Performs a back-propagation step through the module, with respect to the given input. In
        general this method makes the assumption forward(input) has been called before, with the
        same input. This is necessary for optimization reasons. If you do not respect this rule,
        backward() will compute incorrect gradients.

        :param input: ndarray or list of ndarray or JTensor or list of JTensor.
        :param grad_output: ndarray or list of ndarray or JTensor or list of JTensor.
        :return: ndarray or list of ndarray
        """
        jinput, input_is_table = self.check_input(input)
        jgrad_output, grad_output_is_table = self.check_input(grad_output)
        output = callBigDlFunc(self.bigdl_type,
                               "modelBackward",
                               self.value,
                               jinput,
                               input_is_table,
                               jgrad_output,
                               grad_output_is_table)
        return self.convert_output(output)

    def zero_grad_parameters(self):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        If the module has parameters, this will zero the accumulation of the gradients with respect
        to these parameters. Otherwise, it does nothing.
        """
        callJavaFunc(self.value.zeroGradParameters)

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
        callJavaFunc(self.value.reset)
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

    def evaluate(self, *args):
        """
        No argument passed in:
        Evaluate the model to set train = false, useful when doing test/forward
        :return: layer itself

        Three arguments passed in:
        A method to benchmark the model quality.

        :param dataset: the input data
        :param batch_size: batch size
        :param val_methods: a list of validation methods. i.e: Top1Accuracy,Top5Accuracy and Loss.
        :return: a list of the metrics result
        """
        if len(args) == 0:
            callBigDlFunc(self.bigdl_type,
                          "evaluate", self.value)
            return self
        elif len(args) == 3:
            dataset, batch_size, val_methods = args
            if (isinstance(dataset, ImageFrame)):
                return callBigDlFunc(self.bigdl_type,
                                     "modelEvaluateImageFrame",
                                     self.value,
                                     dataset, batch_size, val_methods)
            else:
                return callBigDlFunc(self.bigdl_type,
                                     "modelEvaluate",
                                     self.value,
                                     dataset, batch_size, val_methods)
        else:
            invalidInputError(False,
                              "Error when calling evaluate(): it takes no argument or"
                              " exactly three arguments only")

    def _to_jtensors(self, x):
        x = to_list(x)
        if isinstance(x[0], np.ndarray):
            return [JTensor.from_ndarray(i) for i in x]
        elif isinstance(x[0], JTensor):
            return x
        else:
            invalidInputError(False, "Not supported type: %s" % type(x[0]))

    def predict_local(self, X, batch_size=-1):
        """
        :param X: X can be a ndarray or list of ndarray if the model has multiple inputs.
                  The first dimension of X should be batch.
        :param batch_size: total batch size of prediction.
        :return: a ndarray as the prediction result.
        """

        jresults = callBigDlFunc(self.bigdl_type,
                                 "predictLocal",
                                 self.value,
                                 self._to_jtensors(X),
                                 batch_size)

        return np.stack([j.to_ndarray() for j in jresults])

    def predict_class_local(self, X):
        """

        :param X: X can be a ndarray or list of ndarray if the model has multiple inputs.
                  The first dimension of X should be batch.
        :return: a ndarray as the prediction result.
        """
        result = callBigDlFunc(self.bigdl_type,
                               "predictLocalClass",
                               self.value,
                               self._to_jtensors(X))
        return np.stack(result)

    def predict(self, features, batch_size=-1):
        """
        Model inference base on the given data.
        :param features: it can be a ndarray or list of ndarray for locally inference
                         or RDD[Sample] for running in distributed fashion
        :param batch_size: total batch size of prediction.
        :return: ndarray or RDD[Sample] depend on the the type of features.
        """
        if isinstance(features, RDD):
            return self.predict_distributed(features, batch_size)
        else:
            return self.predict_local(features, batch_size)

    def predict_class(self, features):
        """
        Model inference base on the given data which returning label
        :param features: it can be a ndarray or list of ndarray for locally inference
                         or RDD[Sample] for running in distributed fashion
        :return: ndarray or RDD[Sample] depend on the the type of features.
        """
        if isinstance(features, RDD):
            return self.predict_class_distributed(features)
        else:
            return self.predict_class_local(features)

    def predict_distributed(self, data_rdd, batch_size=-1):
        """
        Model inference base on the given data.
        You need to invoke collect() to trigger those action \
        as the returning result is an RDD.

        :param data_rdd: the data to be predict.
        :param batch_size: total batch size of prediction.
        :return: An RDD represent the predict result.
        """
        result = callBigDlFunc(self.bigdl_type,
                               "modelPredictRDD", self.value, data_rdd, batch_size)
        return result.map(lambda data: data.to_ndarray())

    def predict_class_distributed(self, data_rdd):
        """
        module predict, return the predict label

        :param data_rdd: the data to be predict.
        :return: An RDD represent the predict label.
        """
        result = callBigDlFunc(self.bigdl_type,
                               "modelPredictClass", self.value, data_rdd)
        return result

    def predict_image(self, image_frame, output_layer=None, share_buffer=False,
                      batch_per_partition=4, predict_key="predict"):
        """
        model predict images, return imageFrame with predicted tensor
        :param image_frame imageFrame that contains images
        :param output_layer if output_layer is not null, the output of layer that matches
        output_layer will be used as predicted output
        :param share_buffer whether to share same memory for each batch predict results
        :param batch_per_partition batch size per partition, default is 4
        :param predict_key key to store predicted results
        """

        image_frame = callBigDlFunc(self.bigdl_type, "modelPredictImage", self.value,
                                    image_frame,
                                    output_layer,
                                    share_buffer,
                                    batch_per_partition,
                                    predict_key)
        return ImageFrame(image_frame)

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
        >>> np.testing.assert_allclose(weights[0][0], np.array([1., 2., 3.]))
        >>> np.testing.assert_allclose(weights[1], np.array([7., 8.]))
        >>> relu = ReLU()
        creating: createReLU
        >>> from py4j.protocol import Py4JJavaError
        >>> try:
        ...     relu.set_weights([np.array([[1,2,3],[4,5,6]]), np.array([7,8])])
        ... except Py4JJavaError as err:
        ...     print(err.java_exception)
        ...
        java.lang.IllegalArgumentException: this layer does not have weight/bias
        >>> relu.get_weights()
        The layer does not have weight/bias
        >>> add = Add(2)
        creating: createAdd
        >>> try:
        ...     add.set_weights([np.array([7,8]), np.array([1,2])])
        ... except Py4JJavaError as err:
        ...     pass
        ...
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

    def is_with_weights(self):
        return callBigDlFunc(self.bigdl_type,
                             "isWithWeights", self.value)

    def saveModel(self, modelPath, weightPath=None, over_write=False):
        callBigDlFunc(self.bigdl_type, "saveBigDLModule", self.value, modelPath,
                      weightPath, over_write)

    def save_caffe(self, prototxt_path, model_path, use_v2=True, overwrite=False):
        callBigDlFunc(self.bigdl_type, "saveCaffe", self.value, prototxt_path,
                      model_path, use_v2, overwrite)

    def save_tensorflow(self, inputs, path, byte_order="little_endian", data_format="nhwc"):
        """
        Save a model to protobuf files so that it can be used in tensorflow inference.

        When saving the model, placeholders will be added to the tf model as input nodes. So
        you need to pass in the names and shapes of the placeholders. BigDL model doesn't have
        such information. The order of the placeholder information should be same as the inputs
        of the graph model.
        :param inputs: placeholder information, should be an array of tuples (input_name, shape)
                       where 'input_name' is a string and shape is an array of integer
        :param path: the path to be saved to
        :param byte_order: model byte order
        :param data_format: model data format, should be "nhwc" or "nchw"
        """
        callBigDlFunc(self.bigdl_type, "saveTF", self.value, inputs, path, byte_order, data_format)

    def setWRegularizer(self, wRegularizer):
        '''
        set weight regularizer
        :param wRegularizer: weight regularizer
        :return:
        '''
        self.value.wRegularizer = wRegularizer.value

    def setBRegularizer(self, bRegularizer):
        '''
        set bias regularizer
        :param wRegularizer: bias regularizer
        :return:
        '''
        self.value.bRegularizer = bRegularizer.value

    def freeze(self, names=None):
        """
        freeze module, if names is not None, set an array of layers that match given names
        to be freezed
        :param names: an array of layer names
        :return:
        """
        callBigDlFunc(self.bigdl_type, "freeze", self.value, names)
        return self

    def unfreeze(self, names=None):
        """
        unfreeze module, if names is not None, unfreeze layers that match given names
        :param names: an array of layer names
        :return:
        """
        callBigDlFunc(self.bigdl_type, "unFreeze", self.value, names)
        return self

    def training(self, is_training=True):
        '''
        Set this layer in the training mode or in predition mode if is_training=False
        '''
        if is_training:
            callJavaFunc(self.value.training)
        else:
            callJavaFunc(self.value.evaluate)
        return self

    def is_training(self):
        '''
        :return: Whether this layer is in the training mode

        >>> layer = Dropout()
        creating: createDropout
        >>> layer = layer.evaluate()
        >>> layer.is_training()
        False
        >>> layer = layer.training()
        >>> layer.is_training()
        True
        '''
        return callJavaFunc(self.value.isTraining)

    def quantize(self):
        '''
        Clone self and quantize it, at last return a new quantized model.
        :return: A new quantized model.

        >>> fc = Linear(4, 2)
        creating: createLinear
        >>> fc.set_weights([np.ones((2, 4)), np.ones((2,))])
        >>> input = np.ones((2, 4))
        >>> output = fc.forward(input)
        >>> expected_output = np.array([[5., 5.], [5., 5.]])
        >>> np.testing.assert_allclose(output, expected_output)
        >>> quantized_fc = fc.quantize()
        >>> quantized_output = quantized_fc.forward(input)
        >>> expected_quantized_output = np.array([[5., 5.], [5., 5.]])
        >>> np.testing.assert_allclose(quantized_output, expected_quantized_output)

        >>> assert("quantized.Linear" in quantized_fc.__str__())
        >>> conv = SpatialConvolution(1, 2, 3, 3)
        creating: createSpatialConvolution
        >>> conv.set_weights([np.ones((2, 1, 3, 3)), np.zeros((2,))])
        >>> input = np.ones((2, 1, 4, 4))
        >>> output = conv.forward(input)
        >>> expected_output = np.array([[[[9., 9.], [9., 9.]], [[9., 9.], [9., 9.]]], [[[9., 9.],
        ... [9., 9.]], [[9., 9.], [9., 9.]]]])
        >>> np.testing.assert_allclose(output, expected_output)
        >>> quantized_conv = conv.quantize()
        >>> quantized_output = quantized_conv.forward(input)
        >>> expected_quantized_output = np.array([[[[9., 9.], [9., 9.]], [[9., 9.], [9., 9.]]],
        ... [[[9., 9.], [9., 9.]], [[9., 9.], [9., 9.]]]])
        >>> np.testing.assert_allclose(quantized_output, expected_quantized_output)
        >>> assert("quantized.SpatialConvolution" in quantized_conv.__str__())
        >>> seq = Sequential()
        creating: createSequential
        >>> seq = seq.add(conv)
        >>> seq = seq.add(Reshape([8, 4], False))
        creating: createReshape
        >>> seq = seq.add(fc)
        >>> input = np.ones([1, 1, 6, 6])
        >>> output = seq.forward(input)
        >>> expected_output = np.array([[37., 37.], [37., 37.], [37., 37.], [37., 37.], [37., 37.],
        ... [37., 37.], [37., 37.], [37., 37.]])
        >>> np.testing.assert_allclose(output, expected_output)
        >>> quantized_seq = seq.quantize()
        >>> quantized_output = quantized_seq.forward(input)
        >>> expected_quantized_output = np.array([[37., 37.], [37., 37.], [37., 37.], [37., 37.],
        ... [37., 37.], [37., 37.], [37., 37.], [37., 37.]])
        >>> np.testing.assert_allclose(quantized_output, expected_quantized_output)
        >>> assert("quantized.Linear" in quantized_seq.__str__())
        >>> assert("quantized.SpatialConvolution" in quantized_seq.__str__())
        '''
        quantized_model = callBigDlFunc(self.bigdl_type, "quantize", self.value)
        return Layer.of(quantized_model)


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

    @property
    def layers(self):
        jlayers = callBigDlFunc(self.bigdl_type, "getContainerModules", self)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers

    def flattened_layers(self, include_container=False):
        jlayers = callBigDlFunc(self.bigdl_type, "getFlattenModules", self, include_container)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers


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

    We also support initializing a Graph directly from a tensorflow module. In this case, you should
    pass your tensorflow nodes as inputs and outputs and also specify the byte_order parameter
    ("little_endian"
     or "big_endian") and node_type parameter ("bigdl" or "tensorflow")
    node_type parameter.
    """

    def __init__(self,
                 inputs,
                 outputs,
                 jvalue=None,
                 bigdl_type="float", byte_order="little_endian", model_type="bigdl"):
        if jvalue:
            self.value = jvalue
            self.bigdl_type = bigdl_type
        elif model_type == "bigdl" and (isinstance(inputs, list) or isinstance(inputs, Node)):
            super(Model, self).__init__(None, bigdl_type,
                                        to_list(inputs),
                                        to_list(outputs))
        elif model_type == "bigdl" and isinstance(inputs, Layer):
            self.value = callBigDlFunc(
                bigdl_type, "createModelPreprocessor", inputs, outputs)
            self.bigdl_type = bigdl_type
        else:
            from bigdl.dllib.utils.tf_utils import convert
            model = convert(to_list(inputs), to_list(outputs), byte_order, bigdl_type)
            super(Model, self).__init__(model.value, bigdl_type)

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Model([], [], jvalue=jvalue)
        model.value = jvalue
        return model

    def __str__(self):
        return "->".join(self.layers())

    @staticmethod
    def loadModel(modelPath, weightPath=None, bigdl_type="float"):
        """
        Load a pre-trained Bigdl model.

        :param path: The path containing the pre-trained model.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadBigDLModule", modelPath, weightPath)
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
    def load_keras(json_path=None, hdf5_path=None, by_name=False):
        """
        Load a pre-trained Keras model.

        :param json_path: The json path containing the keras model definition.
        :param hdf5_path: The HDF5 path containing the pre-trained keras model weights with or
         without the model architecture.
        :return: A bigdl model.
        """
        import os
        try:
            import tensorflow
        except ImportError:
            os.environ['KERAS_BACKEND'] = "theano"
            try:
                # Make theano backend compatible with Python3
                from theano import ifelse
            except ImportError:
                invalidInputError(False,
                                  "No backend is found for Keras."
                                  " Please install either tensorflow or theano.")
        from bigdl.dllib.keras.converter import DefinitionLoader, WeightLoader
        if json_path and not hdf5_path:
            return DefinitionLoader.from_json_path(json_path)
        elif json_path and hdf5_path:
            return WeightLoader.load_weights_from_json_hdf5(json_path, hdf5_path, by_name=by_name)
        elif hdf5_path and not json_path:
            kmodel, bmodel = DefinitionLoader.from_hdf5_path(hdf5_path)
            WeightLoader.load_weights_from_kmodel(bmodel, kmodel)
            return bmodel

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

    @staticmethod
    def load_caffe_model(defPath, modelPath, bigdl_type="float"):
        """
        Load a pre-trained Caffe model.


        :param defPath: The path containing the caffe model definition.
        :param modelPath: The path containing the pre-trained caffe model.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadCaffeModel", defPath, modelPath)
        return Layer.of(jmodel)

    @staticmethod
    def load_tensorflow(path, inputs, outputs, byte_order="little_endian",
                        bin_file=None, generated_backward=True, bigdl_type="float"):
        """
        Load a pre-trained Tensorflow model.
        :param path: The path containing the pre-trained model.
        :param inputs: The input node of this graph
        :param outputs: The output node of this graph
        :param byte_order: byte_order of the file, `little_endian` or `big_endian`
        :param bin_file: the optional bin file produced by bigdl dump_model util function to store
                         the weights
        :param generated_backward: if generate backward graph
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "loadTF", path, inputs, outputs,
                               byte_order, bin_file, generated_backward)
        return Model.of(jmodel)

    @staticmethod
    def train(output, data, label, opt_method, criterion, batch_size, end_when, session=None,
              bigdl_type="float"):
        from bigdl.dllib.utils.tf_utils import get_path
        from bigdl.dllib.utils.common import Sample
        output_name = output.name.split(":")[0]
        path = get_path(output_name, session)
        sc = get_spark_context()
        rdd_train_images = sc.parallelize(data)
        rdd_train_labels = sc.parallelize(label)
        rdd_train_sample = rdd_train_images.zip(rdd_train_labels).map(lambda input:
                                                                      Sample.from_ndarray(input[0],
                                                                                          input[1]))
        jmodel = callBigDlFunc(bigdl_type, "trainTF", path, output_name, rdd_train_sample,
                               opt_method, criterion, batch_size, end_when)
        return Model.of(jmodel)

    def stop_gradient(self, stop_layers, bigdl_type="float"):
        """
        stop the input gradient of layers that match the given ```names```
        their input gradient are not computed.
        And they will not contributed to the input gradient computation of
        layers that depend on them.
        :param stop_layers:  an array of layer names
        :param bigdl_type:
        :return:
        """
        callBigDlFunc(bigdl_type, "setStopGradient", self.value, stop_layers)
        return self

    def node(self, name, bigdl_type="float"):
        """
        Return the corresponding node has the given name. If the given name doesn't match any node,
        an exception will be thrown
        :param name: node name
        :param bigdl_type:
        :return:
        """
        jnode = callBigDlFunc(bigdl_type, "findGraphNode", self.value, name)
        return Node.of(jnode)

    def save_graph_topology(self, log_path, bigdl_type="float"):
        """
        save current model graph to a folder, which can be display in tensorboard by running
            tensorboard --logdir logPath
        :param log_path: path to save the model graph
        :param bigdl_type:
        :return:
        """
        callBigDlFunc(bigdl_type, "saveGraphTopology", self.value, log_path)
        return self

    def set_input_formats(self, input_formats, bigdl_type="float"):
        """
        set input formats for graph.
        :param input_formats: list of input format numbers
        :param bigdl_type:
        :return:
        """
        jname = callBigDlFunc(bigdl_type,
                              "getRealClassNameOfJValue",
                              self.value)
        if jname.split(".")[-1] == "StaticGraph":
            callBigDlFunc(bigdl_type, "setInputFormats", self.value, input_formats)
        return self

    def set_output_formats(self, output_formats, bigdl_type="float"):
        """
        set output formats for graph.
        :param output_formats: list of output format numbers
        :param bigdl_type:
        :return:
        """
        jname = callBigDlFunc(bigdl_type,
                              "getRealClassNameOfJValue",
                              self.value)
        if jname.split(".")[-1] == "StaticGraph":
            callBigDlFunc(bigdl_type, "setOutputFormats", self.value, output_formats)
        return self


class Attention(Layer):
    '''
    Implementation of multiheaded attention and self-attention layers.

    >>> attention = Attention(8, 4, 1.0)
    creating: createAttention
    '''

    def __init__(self, hidden_size, num_heads, attention_dropout, bigdl_type="float"):
        super(Attention, self).__init__(None, bigdl_type,
                                        hidden_size, num_heads, attention_dropout)


class FeedForwardNetwork(Layer):
    '''
    Implementation FeedForwardNetwork constructed with fully connected network.
    Input with shape (batch_size, length, hidden_size)
    Output with shape (batch_size, length, hidden_size)

    >>> ffn = FeedForwardNetwork(8, 4, 1.0)
    creating: createFeedForwardNetwork
    '''

    def __init__(self, hidden_size, filter_size, relu_dropout, bigdl_type="float"):
        super(FeedForwardNetwork, self).__init__(None, bigdl_type,
                                                 hidden_size, filter_size, relu_dropout)


class LayerNormalization(Layer):
    '''
    Applies layer normalization.

    >>> norm = LayerNormalization(8)
    creating: createLayerNormalization
    '''

    def __init__(self, hidden_size, bigdl_type="float"):
        super(LayerNormalization, self).__init__(None, bigdl_type, hidden_size)


class TableOperation(Layer):
    '''
    When two tensors have different size, firstly expand small size tensor to large size tensor,
    and then do table operation.

    >>> norm = TableOperation(CMulTable())
    creating: createCMulTable
    creating: createTableOperation
    '''

    def __init__(self, operation_layer, bigdl_type="float"):
        super(TableOperation, self).__init__(None, bigdl_type, operation_layer)


class ExpandSize(Layer):
    '''
    Expand tensor to configured size

    >>> expand = ExpandSize([2, 3, 4])
    creating: createExpandSize
    '''

    def __init__(self, sizes, bigdl_type="float"):
        super(ExpandSize, self).__init__(None, bigdl_type, sizes)


class Transformer(Layer):
    '''
    Implementation for Transformer
    >>> layer = Transformer(20, 4, 2, 3, 1, 0.1, 0.1, 0.1)
    creating: createTransformer
    '''

    def __init__(self, vocab_size, hidden_size, num_heads, filter_size, num_hidden_layers,
                 postprocess_dropout, attention_dropout,
                 relu_dropout, bigdl_type="float"):
        super(Transformer, self).__init__(None, bigdl_type, vocab_size,
                                          hidden_size, num_heads, filter_size,
                                          num_hidden_layers, postprocess_dropout,
                                          attention_dropout, relu_dropout)


class Linear(Layer):
    '''
    The [[Linear]] module applies a linear transformation to the input data,
    i.e. `y = Wx + b`. The input given in `forward(input)` must be either
    a vector (1D tensor) or matrix (2D tensor). If the input is a vector, it must
    have the size of `inputSize`. If it is a matrix, then each row is assumed to be
    an input sample of given batch (the number of rows means the batch size and
    the number of columns should be equal to the `inputSize`).

    :param input_size the size the each input sample
    :param output_size the size of the module output of each sample
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.
    :param init_weight: the optional initial value for the weight
    :param init_bias: the optional initial value for the bias
    :param init_grad_weight: the optional initial value for the grad_weight
    :param init_grad_bias: the optional initial value for the grad_bias


    >>> linear = Linear(100, 10, True, L1Regularizer(0.5), L1Regularizer(0.5))
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createLinear
    >>> import numpy as np
    >>> init_weight = np.random.randn(10, 100)
    >>> init_bias = np.random.randn(10)
    >>> init_grad_weight = np.zeros([10, 100])
    >>> init_grad_bias = np.zeros([10])
    >>> linear = Linear(100, 10, True, L1Regularizer(0.5), L1Regularizer(0.5), init_weight,
    ... init_bias, init_grad_weight, init_grad_bias)
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createLinear
    '''

    def __init__(self, input_size, output_size, with_bias=True, wRegularizer=None,
                 bRegularizer=None,
                 init_weight=None, init_bias=None, init_grad_weight=None, init_grad_bias=None,
                 bigdl_type="float"):
        super(Linear, self).__init__(None, bigdl_type, input_size, output_size,
                                     with_bias, wRegularizer, bRegularizer,
                                     JTensor.from_ndarray(init_weight),
                                     JTensor.from_ndarray(init_bias),
                                     JTensor.from_ndarray(init_grad_weight),
                                     JTensor.from_ndarray(init_grad_bias))

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class SparseLinear(Layer):
    '''
    SparseLinear is the sparse version of module Linear. SparseLinear has two different from Linear:
    firstly, SparseLinear's input Tensor is a SparseTensor. Secondly, SparseLinear doesn't backward
    gradient to next layer in the backpropagation by default, as the gradInput of SparseLinear is
    useless and very big in most cases.

    But, considering model like Wide&Deep, we provide backwardStart and backwardLength to backward
    part of the gradient to next layer.

    :param input_size the size the each input sample
    :param output_size the size of the module output of each sample
    :param backwardStart backwardStart index, counting from 1
    :param backwardLength backward length
    :param withBias if has bias
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
     input weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.
    :param init_weight: the optional initial value for the weight
    :param init_bias: the optional initial value for the bias
    :param init_grad_weight: the optional initial value for the grad_weight
    :param init_grad_bias: the optional initial value for the grad_bias


    >>> sparselinear = SparseLinear(100, 10, True, wRegularizer=L1Regularizer(0.5),
    ... bRegularizer=L1Regularizer(0.5))
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createSparseLinear
    >>> import numpy as np
    >>> init_weight = np.random.randn(10, 100)
    >>> init_bias = np.random.randn(10)
    >>> init_grad_weight = np.zeros([10, 100])
    >>> init_grad_bias = np.zeros([10])
    >>> sparselinear = SparseLinear(100, 10, True, 1, 5, L1Regularizer(0.5), L1Regularizer(0.5),
    ... init_weight, init_bias, init_grad_weight, init_grad_bias)
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createSparseLinear
    >>> np.random.seed(123)
    >>> init_weight = np.random.randn(5, 1000)
    >>> init_bias = np.random.randn(5)
    >>> sparselinear = SparseLinear(1000, 5, init_weight=init_weight, init_bias=init_bias)
    creating: createSparseLinear
    >>> input = JTensor.sparse(np.array([1, 3, 5, 2, 4, 6]), np.array([0, 0, 0, 1, 1, 1, 1, 5, 300,
    ... 2, 100, 500]), np.array([2, 1000]))
    >>> output = sparselinear.forward(input)
    >>> expected_output = np.array([[10.09569263, -10.94844246, -4.1086688, 1.02527523, 11.80737209]
    ... , [7.9651413, 9.7131443, -10.22719955, 0.02345783, -3.74368906]])
    >>> np.testing.assert_allclose(output, expected_output, rtol=1e-6, atol=1e-6)
    '''

    def __init__(self, input_size, output_size, with_bias=True, backwardStart=-1, backwardLength=-1,
                 wRegularizer=None, bRegularizer=None, init_weight=None, init_bias=None,
                 init_grad_weight=None, init_grad_bias=None, bigdl_type="float"):
        super(SparseLinear, self).__init__(None, bigdl_type, input_size, output_size,
                                           with_bias, backwardStart, backwardLength,
                                           wRegularizer, bRegularizer,
                                           JTensor.from_ndarray(init_weight),
                                           JTensor.from_ndarray(init_bias),
                                           JTensor.from_ndarray(init_grad_weight),
                                           JTensor.from_ndarray(init_grad_bias))

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class DenseToSparse(Layer):
    '''
    Convert DenseTensor to SparseTensor.


    >>> DenseToSparse = DenseToSparse()
    creating: createDenseToSparse
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(DenseToSparse, self).__init__(None, bigdl_type)


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


    '''

    def __init__(self, jvalue=None, bigdl_type="float"):
        super(Sequential, self).__init__(jvalue, bigdl_type)

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Sequential(jvalue=jvalue)
        model.value = jvalue
        return model

    def to_graph(self):
        """
        Convert a sequential model (Sequential) to a graph model (Model)
        :return: A Python graph model
        """
        jvalue = callBigDlFunc(self.bigdl_type,
                               "toGraph",
                               self.value)
        model = Model.from_jvalue(jvalue)
        return model


class TemporalConvolution(Layer):
    '''
    Applies a 1D convolution over an input sequence composed of nInputFrame frames..
    The input tensor in `forward(input)` is expected to be a 2D tensor
    (`nInputFrame` x `inputFrameSize`) or a 3D tensor
    (`nBatchFrame` x `nInputFrame` x `inputFrameSize`).

    :param input_frame_size The input frame size expected in sequences given into `forward()`
    :param output_frame_size The output frame size the convolution layer will produce.
    :param kernel_w The kernel width of the convolution
    :param stride_w The step of the convolution in the width dimension.
    :param propagate_back Whether propagate gradient back, default is true.
    :param weight_regularizer instance of [[Regularizer]]
                        (eg. L1 or L2 regularization), applied to the input weights matrices.
    :param bias_regularizer instance of [[Regularizer]]
                         applied to the bias.
    :param init_weight Initial weight
    :param init_bias Initial bias
    :param init_grad_weight Initial gradient weight
    :param init_grad_bias Initial gradient bias

    >>> temporalConvolution = TemporalConvolution(6, 12, 5, 5)
    creating: createTemporalConvolution
    >>> temporalConvolution.setWRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    >>> temporalConvolution.setBRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    '''

    def __init__(self,
                 input_frame_size,
                 output_frame_size,
                 kernel_w,
                 stride_w=1,
                 propagate_back=True,
                 weight_regularizer=None,
                 bias_regularizer=None,
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 bigdl_type="float"):
        super(TemporalConvolution, self).__init__(None, bigdl_type,
                                                  input_frame_size,
                                                  output_frame_size,
                                                  kernel_w,
                                                  stride_w,
                                                  propagate_back,
                                                  weight_regularizer,
                                                  bias_regularizer,
                                                  JTensor.from_ndarray(init_weight),
                                                  JTensor.from_ndarray(init_bias),
                                                  JTensor.from_ndarray(init_grad_weight),
                                                  JTensor.from_ndarray(init_grad_bias))

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class LocallyConnected1D(Layer):
    '''
    The `LocallyConnected1D` layer works similarly to
    the `TemporalConvolution` layer, except that weights are unshared,
    that is, a different set of filters is applied at each different patch
    of the input.
    The input tensor in `forward(input)` is expected to be a 2D tensor
    (`nInputFrame` x `inputFrameSize`) or a 3D tensor
    (`nBatchFrame` x `nInputFrame` x `inputFrameSize`).
    :param nInputFrame the input frame channel
    :param input_frame_size The input frame size expected in sequences given into `forward()`
    :param output_frame_size The output frame size the convolution layer will produce.
    :param kernel_w The kernel width of the convolution
    :param stride_w The step of the convolution in the width dimension.
    :param propagate_back Whether propagate gradient back, default is true.
    :param weight_regularizer instance of [[Regularizer]]
                        (eg. L1 or L2 regularization), applied to the input weights matrices.
    :param bias_regularizer instance of [[Regularizer]]
                         applied to the bias.
    :param init_weight Initial weight
    :param init_bias Initial bias
    :param init_grad_weight Initial gradient weight
    :param init_grad_bias Initial gradient bias
    >>> locallyConnected1D = LocallyConnected1D(10, 6, 12, 5, 5)
    creating: createLocallyConnected1D
    >>> locallyConnected1D.setWRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    >>> locallyConnected1D.setBRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    '''

    def __init__(self,
                 n_input_frame,
                 input_frame_size,
                 output_frame_size,
                 kernel_w,
                 stride_w=1,
                 propagate_back=True,
                 weight_regularizer=None,
                 bias_regularizer=None,
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 bigdl_type="float"):
        super(LocallyConnected1D, self).__init__(None, bigdl_type,
                                                 n_input_frame,
                                                 input_frame_size,
                                                 output_frame_size,
                                                 kernel_w,
                                                 stride_w,
                                                 propagate_back,
                                                 weight_regularizer,
                                                 bias_regularizer,
                                                 JTensor.from_ndarray(init_weight),
                                                 JTensor.from_ndarray(init_bias),
                                                 JTensor.from_ndarray(init_grad_weight),
                                                 JTensor.from_ndarray(init_grad_bias))

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class BinaryTreeLSTM(Layer):
    '''
    This class is an implementation of Binary TreeLSTM (Constituency Tree LSTM).
    :param inputSize input units size
    :param hiddenSize hidden units size
    :param gateOutput whether gate output
    :param withGraph whether create lstms with [[Graph]], the default value is true.
    >>> treeLSTM = BinaryTreeLSTM(100, 200)
    creating: createBinaryTreeLSTM
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_output=True,
                 with_graph=True,
                 bigdl_type="float"):
        super(BinaryTreeLSTM, self).__init__(None,
                                             bigdl_type,
                                             input_size,
                                             hidden_size,
                                             gate_output,
                                             with_graph)


class LocallyConnected2D(Layer):
    '''
    The LocallyConnected2D layer works similarly to the [[SpatialConvolution]] layer,
    except that weights are unshared, that is, a different set of filters
    is applied at each different patch of the input.

    :param n_input_plane The number of expected input planes in the image given into forward()
    :param input_width The expected width of input
    :param input_height The expected height of input
    :param n_output_plane The number of output planes the convolution layer will produce.
    :param kernel_w The kernel width of the convolution
    :param kernel_h The kernel height of the convolution
    :param stride_w The step of the convolution in the width dimension.
    :param stride_h The step of the convolution in the height dimension
    :param pad_w The additional zeros added per width to the input planes.
    :param pad_h The additional zeros added per height to the input planes.
    :param propagate_back Propagate gradient back
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.
    :param init_weight: the optional initial value for the weight
    :param init_bias: the optional initial value for the bias
    :param init_grad_weight: the optional initial value for the grad_weight
    :param init_grad_bias: the optional initial value for the grad_bias
    :param with_bias: the optional initial value for if need bias
    :param data_format: a string value of "NHWC" or "NCHW" to specify the input data format of this
                        layer. In "NHWC" format data is stored in the order of
                        [batch_size, height, width, channels], in "NCHW" format data is stored
                        in the order of [batch_size, channels, height, width].

    >>> locallyConnected2D = LocallyConnected2D(6, 2, 4, 12, 5, 5)
    creating: createLocallyConnected2D
    >>> locallyConnected2D.setWRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    >>> locallyConnected2D.setBRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    '''

    def __init__(self,
                 n_input_plane,
                 input_width,
                 input_height,
                 n_output_plane,
                 kernel_w,
                 kernel_h,
                 stride_w=1,
                 stride_h=1,
                 pad_w=0,
                 pad_h=0,
                 propagate_back=True,
                 wRegularizer=None,
                 bRegularizer=None,
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 with_bias=True,
                 data_format="NCHW",
                 bigdl_type="float"):
        super(LocallyConnected2D, self).__init__(None, bigdl_type,
                                                 n_input_plane,
                                                 input_width,
                                                 input_height,
                                                 n_output_plane,
                                                 kernel_w,
                                                 kernel_h,
                                                 stride_w,
                                                 stride_h,
                                                 pad_w,
                                                 pad_h,
                                                 propagate_back,
                                                 wRegularizer,
                                                 bRegularizer,
                                                 JTensor.from_ndarray(init_weight),
                                                 JTensor.from_ndarray(init_bias),
                                                 JTensor.from_ndarray(init_grad_weight),
                                                 JTensor.from_ndarray(init_grad_bias),
                                                 with_bias,
                                                 data_format)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class SpatialConvolution(Layer):
    '''
    Applies a 2D convolution over an input image composed of several input planes.
    The input tensor in forward(input) is expected to be
    a 3D tensor (nInputPlane x height x width).

    :param n_input_plane The number of expected input planes in the image given into forward()
    :param n_output_plane The number of output planes the convolution layer will produce.
    :param kernel_w The kernel width of the convolution
    :param kernel_h The kernel height of the convolution
    :param stride_w The step of the convolution in the width dimension.
    :param stride_h The step of the convolution in the height dimension
    :param pad_w The additional zeros added per width to the input planes.
    :param pad_h The additional zeros added per height to the input planes.
    :param n_group Kernel group number
    :param propagate_back Propagate gradient back
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.
    :param init_weight: the optional initial value for the weight
    :param init_bias: the optional initial value for the bias
    :param init_grad_weight: the optional initial value for the grad_weight
    :param init_grad_bias: the optional initial value for the grad_bias
    :param with_bias: the optional initial value for if need bias
    :param data_format: a string value of "NHWC" or "NCHW" to specify the input data format of this
                       layer. In "NHWC" format
                       data is stored in the order of [batch_size, height, width, channels],
                       in "NCHW" format data is stored
                       in the order of [batch_size, channels, height, width].

    >>> spatialConvolution = SpatialConvolution(6, 12, 5, 5)
    creating: createSpatialConvolution
    >>> spatialConvolution.setWRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    >>> spatialConvolution.setBRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    >>> import numpy as np
    >>> init_weight = np.random.randn(1, 12, 6, 5, 5)
    >>> init_bias = np.random.randn(12)
    >>> init_grad_weight = np.zeros([1, 12, 6, 5, 5])
    >>> init_grad_bias = np.zeros([12])
    >>> spatialConvolution = SpatialConvolution(6, 12, 5, 5, 1, 1, 0, 0, 1, True,
    ... L1Regularizer(0.5), L1Regularizer(0.5), init_weight, init_bias, init_grad_weight,
    ... init_grad_bias, True, "NCHW")
    creating: createL1Regularizer
    creating: createL1Regularizer
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
                 wRegularizer=None,
                 bRegularizer=None,
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 with_bias=True,
                 data_format="NCHW",
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
                                                 wRegularizer,
                                                 bRegularizer,
                                                 JTensor.from_ndarray(init_weight),
                                                 JTensor.from_ndarray(init_bias),
                                                 JTensor.from_ndarray(init_grad_weight),
                                                 JTensor.from_ndarray(init_grad_bias),
                                                 with_bias,
                                                 data_format)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class TemporalMaxPooling(Layer):
    '''
    Applies 1D max-pooling operation in kW regions by step size dW steps.
    Input sequence composed of nInputFrame frames.
    The input tensor in forward(input) is expected to be a 2D tensor (nInputFrame x inputFrameSize)
     or a 3D tensor (nBatchFrame x nInputFrame x inputFrameSize).

    If the input sequence is a 2D tensor of dimension nInputFrame x inputFrameSize,
    the output sequence will be nOutputFrame x inputFrameSize where

    nOutputFrame = (nInputFrame - k_w) / d_w + 1

    :param k_w:              kernel width
    :param d_w:              step size in width, default is -1, means the `d_w` equals `k_w`

    >>> temporalMaxPooling = TemporalMaxPooling(2, 2)
    creating: createTemporalMaxPooling
    '''

    def __init__(self,
                 k_w,
                 d_w=-1,
                 bigdl_type="float"):
        super(TemporalMaxPooling, self).__init__(None, bigdl_type, k_w,
                                                 d_w)


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

    When padW and padH are both -1, we use a padding algorithm similar to the "SAME"
    padding of tensorflow. That is

     outHeight = Math.ceil(inHeight.toFloat/strideH.toFloat)
     outWidth = Math.ceil(inWidth.toFloat/strideW.toFloat)

     padAlongHeight = Math.max(0, (outHeight - 1) * strideH + kernelH - inHeight)
     padAlongWidth = Math.max(0, (outWidth - 1) * strideW + kernelW - inWidth)
     padTop = padAlongHeight / 2
     padLeft = padAlongWidth / 2

    :param kW:              kernel width
    :param kH:              kernel height
    :param dW:              step size in width
    :param dH:              step size in height
    :param padW:            padding in width
    :param padH:            padding in height
    :param format:          "NCHW" or "NHWC", indicating the input data format

    >>> spatialMaxPooling = SpatialMaxPooling(2, 2, 2, 2)
    creating: createSpatialMaxPooling
    >>> spatialMaxPooling = SpatialMaxPooling(2, 2, 2, 2, -1, -1, True, "NHWC")
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
                 format="NCHW",
                 bigdl_type="float"):
        super(SpatialMaxPooling, self).__init__(None, bigdl_type, kw,
                                                kh,
                                                dw,
                                                dh,
                                                pad_w,
                                                pad_h,
                                                to_ceil,
                                                format)


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

    def get_hidden_state(self):
        """
        get hidden state and cell at last time step.

        :return: list of hidden state and cell
        """
        states = callBigDlFunc(self.bigdl_type, "getHiddenState", self.value)
        return states


class RecurrentDecoder(Recurrent):
    '''
    RecurrentDecoder module is a container of rnn cells which used to make
    a prediction of the next timestep based on the prediction we made from
    the previous timestep. Input for RecurrentDecoder is dynamically composed
    during training. input at t(i) is output at t(i-1), input at t(0) is
    user input, and user input has to be batch x stepShape(shape of the input
    at a single time step).

    Different types of rnn cells can be added using add() function.

    >>> recurrent_decoder = RecurrentDecoder(output_length = 5)
    creating: createRecurrentDecoder
    '''

    def __init__(self, output_length, bigdl_type="float"):
        super(Recurrent, self).__init__(None, bigdl_type, output_length)


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
    :param p: is used for [[Dropout]] probability. For more details aboutRNN dropouts, please refer
              to[RnnDrop: A Novel Dropout for RNNs in ASR](http://www.stat.berkeley.edu/~tsmoon/
              files/Conference/asru2015.pdf)[A Theoretically Grounded Application of Dropout in
              Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)
    :param activation: activation function, by default to be Tanh if not specified.
                        It can also be the name of an existing activation as a string.
    :param inner_activation: activation function for the inner cells, by default to be Sigmoid if
                             not specified.
                            It can also be the name of an existing activation as a string.
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param uRegularizer: instance [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         recurrent weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.


    >>> lstm = LSTM(4, 3, 0.5, 'tanh', Sigmoid(), L1Regularizer(0.5), L1Regularizer(0.5),
    ... L1Regularizer(0.5))
    creating: createSigmoid
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createTanh
    creating: createLSTM
    '''

    def __init__(self, input_size, hidden_size, p=0.0, activation=None, inner_activation=None,
                 wRegularizer=None, uRegularizer=None, bRegularizer=None, bigdl_type="float"):
        if not activation:
            activation = Tanh()
        if not inner_activation:
            inner_activation = Sigmoid()
        if isinstance(activation, six.string_types):
            activation = get_activation_by_name(activation)
        if isinstance(inner_activation, six.string_types):
            inner_activation = get_activation_by_name(inner_activation)
        super(LSTM, self).__init__(None, bigdl_type, input_size, hidden_size, p,
                                   activation, inner_activation, wRegularizer, uRegularizer,
                                   bRegularizer)


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
    :param  p: is used for [[Dropout]] probability. For more details aboutRNN dropouts, please refer
               to[RnnDrop: A Novel Dropout for RNNs in ASR](http://www.stat.berkeley.edu/~tsmoon/
               files/Conference/asru2015.pdf)[A Theoretically Grounded Application of Dropout in
               Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param uRegularizer: instance [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         recurrent weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.

    >>> lstm = LSTMPeephole(4, 3, 0.5, L1Regularizer(0.5), L1Regularizer(0.5), L1Regularizer(0.5))
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createLSTMPeephole
    '''

    def __init__(self, input_size=4, hidden_size=3, p=0.0, wRegularizer=None, uRegularizer=None,
                 bRegularizer=None, bigdl_type="float"):
        super(LSTMPeephole, self).__init__(None, bigdl_type, input_size, hidden_size, p,
                                           wRegularizer, uRegularizer, bRegularizer)


class Gemm(Layer):
    def __init__(self, alpha=1.0, beta=1.0, trans_a=False, trans_b=False, bigdl_type="float"):
        super(Gemm, self).__init__(None, bigdl_type, alpha, beta, trans_a, trans_b)


class GRU(Layer):
    '''
    Gated Recurrent Units architecture.
    The first input in sequence uses zero value for cell and hidden state


|   Ref.
|   http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-
|   rnn-with-python-and-theano/
|   https://github.com/Element-Research/rnn/blob/master/GRU.lua


    :param input_size: the size of each input vector
    :param hidden_size: Hidden unit size in GRU
    :param p: is used for [[Dropout]] probability. For more details aboutRNN dropouts, please refer
              to[RnnDrop: A Novel Dropout for RNNs in ASR](http://www.stat.berkeley.edu/~tsmoon/
              files/Conference/asru2015.pdf)[A Theoretically Grounded Application of Dropout in
              Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf)
    :param activation: activation function, by default to be Tanh if not specified.
                        It can also be the name of an existing activation as a string.
    :param inner_activation: activation function for the inner cells, by default to be Sigmoid if
                             not specified.
                             It can also be the name of an existing activation as a string.
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param uRegularizer: instance [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         recurrent weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.



    >>> gru = GRU(4, 3, 0.5, Tanh(), Sigmoid(), L1Regularizer(0.5), L1Regularizer(0.5),
    ... L1Regularizer(0.5))
    creating: createTanh
    creating: createSigmoid
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createGRU
    '''

    def __init__(self, input_size, hidden_size, p=0.0, activation=None, inner_activation=None,
                 wRegularizer=None, uRegularizer=None, bRegularizer=None, bigdl_type="float"):
        if not activation:
            activation = Tanh()
        if not inner_activation:
            inner_activation = Sigmoid()
        if isinstance(activation, six.string_types):
            activation = get_activation_by_name(activation)
        if isinstance(inner_activation, six.string_types):
            inner_activation = get_activation_by_name(inner_activation)
        super(GRU, self).__init__(None, bigdl_type, input_size, hidden_size, p, activation,
                                  inner_activation,
                                  wRegularizer, uRegularizer, bRegularizer)


class RnnCell(Layer):
    '''
    It is a simple RNN. User can pass an activation function to the RNN.


    :param input_size: the size of each input vector
    :param hidden_size: Hidden unit size in simple RNN
    :param activation: activation function. It can also be the name of an existing activation as a
                       string.
    :param isInputWithBias: boolean
    :param isHiddenWithBias: boolean
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param uRegularizer: instance [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         recurrent weights matrices.
    :param bRegularizer: instance of [[Regularizer]](../regularizers.md),applied to the bias.


    >>> rnn = RnnCell(4, 3, Tanh(), True, True, L1Regularizer(0.5), L1Regularizer(0.5),
    ... L1Regularizer(0.5))
    creating: createTanh
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createRnnCell
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 activation,
                 isInputWithBias=True,
                 isHiddenWithBias=True,
                 wRegularizer=None,
                 uRegularizer=None,
                 bRegularizer=None,
                 bigdl_type="float"):
        if isinstance(activation, six.string_types):
            activation = get_activation_by_name(activation)
        super(RnnCell, self).__init__(None, bigdl_type, input_size, hidden_size, activation,
                                      isInputWithBias, isHiddenWithBias, wRegularizer, uRegularizer,
                                      bRegularizer)


class TimeDistributed(Layer):
    '''
    This layer is intended to apply contained layer to each temporal time slice
    of input tensor.


    For instance, The TimeDistributed Layer can feed each time slice of input tensor
    to the Linear layer.

    The input data format is [Batch, Time, Other dims]. For the contained layer, it must not change
    the Other dims length.


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

    When padW and padH are both -1, we use a padding algorithm similar to the "SAME"
    padding of tensorflow. That is

     outHeight = Math.ceil(inHeight.toFloat/strideH.toFloat)
     outWidth = Math.ceil(inWidth.toFloat/strideW.toFloat)

     padAlongHeight = Math.max(0, (outHeight - 1) * strideH + kernelH - inHeight)
     padAlongWidth = Math.max(0, (outWidth - 1) * strideW + kernelW - inWidth)

     padTop = padAlongHeight / 2
     padLeft = padAlongWidth / 2

    :param kW: kernel width
    :param kH: kernel height
    :param dW: step width
    :param dH: step height
    :param padW: padding width
    :param padH: padding height
    :param global_pooling: If globalPooling then it will pool over the size of the input by doing
                         kH = input->height and kW = input->width
    :param ceilMode: whether the output size is to be ceiled or floored
    :param countIncludePad: whether to include padding when dividing thenumber of elements in
                            pooling region
    :param divide: whether to do the averaging
    :param format:          "NCHW" or "NHWC", indicating the input data format


    >>> spatialAveragePooling = SpatialAveragePooling(7,7)
    creating: createSpatialAveragePooling
    >>> spatialAveragePooling = SpatialAveragePooling(2, 2, 2, 2, -1, -1, True, format="NHWC")
    creating: createSpatialAveragePooling
    '''

    def __init__(self,
                 kw,
                 kh,
                 dw=1,
                 dh=1,
                 pad_w=0,
                 pad_h=0,
                 global_pooling=False,
                 ceil_mode=False,
                 count_include_pad=True,
                 divide=True,
                 format="NCHW",
                 bigdl_type="float"):
        super(SpatialAveragePooling, self).__init__(None, bigdl_type,
                                                    kw,
                                                    kh,
                                                    dw,
                                                    dh,
                                                    pad_w,
                                                    pad_h,
                                                    global_pooling,
                                                    ceil_mode,
                                                    count_include_pad,
                                                    divide,
                                                    format)

    def set_weights(self, weights):
        super(SpatialAveragePooling, self).set_weights(weights)


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

    :param n_output: output feature map number
    :param eps: avoid divide zero
    :param momentum: momentum for weight update
    :param affine: affine operation on output or not
    :param data_format a string value (or DataFormat Object in Scala) of "NHWC" or "NCHW" to specify
                        the input data format of this layer. In "NHWC" format
                        data is stored in the order of [batch_size, height, width, channels],
                        in "NCHW" format data is stored
                        in the order of [batch_size, channels, height, width].


    >>> spatialBatchNormalization = SpatialBatchNormalization(1)
    creating: createSpatialBatchNormalization
    >>> import numpy as np
    >>> init_weight = np.array([1.0])
    >>> init_grad_weight = np.array([0.0])
    >>> init_bias = np.array([0.0])
    >>> init_grad_bias = np.array([0.0])
    >>> spatialBatchNormalization = SpatialBatchNormalization(1, 1e-5, 0.1, True, init_weight,
    ... init_bias, init_grad_weight, init_grad_bias)
    creating: createSpatialBatchNormalization
    >>> spatialBatchNormalization = SpatialBatchNormalization(1, 1e-5, 0.1, True, init_weight,
    ... init_bias, init_grad_weight, init_grad_bias, "NHWC")
    creating: createSpatialBatchNormalization
    '''

    def __init__(self,
                 n_output,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 data_format="NCHW",
                 bigdl_type="float"):
        super(SpatialBatchNormalization, self).__init__(None, bigdl_type,
                                                        n_output,
                                                        eps,
                                                        momentum,
                                                        affine,
                                                        JTensor.from_ndarray(init_weight),
                                                        JTensor.from_ndarray(init_bias),
                                                        JTensor.from_ndarray(init_grad_weight),
                                                        JTensor.from_ndarray(init_grad_bias),
                                                        data_format)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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

    :param size:  the number of channels to sum over
    :param alpha:  the scaling parameter
    :param beta:   the exponent
    :param k: a constant
    :param data_format a string value (or DataFormat Object in Scala) of "NHWC" or "NCHW" to specify
                        the input data format of this layer. In "NHWC" format
                        data is stored in the order of [batch_size, height, width, channels], in
                        "NCHW" format data is stored
                        in the order of [batch_size, channels, height, width]


    >>> spatialCrossMapLRN = SpatialCrossMapLRN()
    creating: createSpatialCrossMapLRN
    >>> spatialCrossMapLRN = SpatialCrossMapLRN(5, 1.0, 0.75, 1.0, "NHWC")
    creating: createSpatialCrossMapLRN
    '''

    def __init__(self,
                 size=5,
                 alpha=1.0,
                 beta=0.75,
                 k=1.0,
                 data_format="NCHW",
                 bigdl_type="float"):
        super(SpatialCrossMapLRN, self).__init__(None, bigdl_type,
                                                 size,
                                                 alpha,
                                                 beta,
                                                 k, data_format)


class SpatialDropout3D(Layer):
    '''
    This version performs the same function as Dropout, however it drops
    entire 3D feature maps instead of individual elements. If adjacent voxels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout3D will help promote independence
    between feature maps and should be used instead.

    :param initP the probability p
    :param format  'NCHW' or 'NHWC'.
        In 'NCHW' mode, the channels dimension (the depth)
        is at index 1, in 'NHWC' mode is it at index 4.

    >>> dropout = SpatialDropout3D(0.5, "NHWC")
    creating: createSpatialDropout3D
    '''

    def __init__(self,
                 init_p=0.5,
                 data_format="NCHW",
                 bigdl_type="float"):
        super(SpatialDropout3D, self).__init__(None, bigdl_type,
                                               init_p, data_format)


class SpatialDropout2D(Layer):
    '''
    This version performs the same function as Dropout, however it drops
    entire 2D feature maps instead of individual elements. If adjacent pixels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout2D will help promote independence
    between feature maps and should be used instead.

    :param initP the probability p
    :param format  'NCHW' or 'NHWC'.
        In 'NCHW' mode, the channels dimension (the depth)
        is at index 1, in 'NHWC' mode is it at index 4.

    >>> dropout = SpatialDropout2D(0.4, "NHWC")
    creating: createSpatialDropout2D
    '''

    def __init__(self,
                 init_p=0.5,
                 data_format="NCHW",
                 bigdl_type="float"):
        super(SpatialDropout2D, self).__init__(None, bigdl_type,
                                               init_p, data_format)


class SpatialDropout1D(Layer):
    '''
    This version performs the same function as Dropout, however it drops
    entire 1D feature maps instead of individual elements. If adjacent frames
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout1D will help promote independence
    between feature maps and should be used instead.

    :param initP the probability p

    >>> dropout = SpatialDropout1D(0.4)
    creating: createSpatialDropout1D
    '''

    def __init__(self,
                 init_p=0.5,
                 bigdl_type="float"):
        super(SpatialDropout1D, self).__init__(None, bigdl_type,
                                               init_p)


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


class GaussianDropout(Layer):
    '''
    Apply multiplicative 1-centered Gaussian noise.
    The multiplicative noise will have standard deviation `sqrt(rate / (1 - rate)).

    As it is a regularization layer, it is only active at training time.

    :param rate: drop probability (as with `Dropout`).


    >>> GaussianDropout = GaussianDropout(0.5)
    creating: createGaussianDropout
    '''

    def __init__(self,
                 rate,
                 bigdl_type="float"):
        super(GaussianDropout, self).__init__(None, bigdl_type,
                                              rate)


class GaussianNoise(Layer):
    '''
    Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise (GS) is a natural choice as corruption process for real valued inputs.

    As it is a regularization layer, it is only active at training time.

    :param stdev: standard deviation of the noise distribution

    >>> GaussianNoise = GaussianNoise(0.5)
    creating: createGaussianNoise
    '''

    def __init__(self,
                 stddev,
                 bigdl_type="float"):
        super(GaussianNoise, self).__init__(None, bigdl_type,
                                            stddev)


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

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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
    >>> import numpy as np
    >>> init_weight = np.random.randn(2)
    >>> init_grad_weight = np.zeros([2])
    >>> init_bias = np.zeros([2])
    >>> init_grad_bias = np.zeros([2])
    >>> batchNormalization = BatchNormalization(2, 1e-5, 1e-5, True, init_weight, init_bias,
    ... init_grad_weight, init_grad_bias)
    creating: createBatchNormalization
    '''

    def __init__(self,
                 n_output,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 bigdl_type="float"):
        super(BatchNormalization, self).__init__(None, bigdl_type,
                                                 n_output,
                                                 eps,
                                                 momentum,
                                                 affine,
                                                 JTensor.from_ndarray(init_weight),
                                                 JTensor.from_ndarray(init_bias),
                                                 JTensor.from_ndarray(init_grad_weight),
                                                 JTensor.from_ndarray(init_grad_bias))

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class BifurcateSplitTable(Layer):
    '''
    Creates a module that takes a Tensor as input and
    outputs two tables, splitting the Tensor along
    the specified dimension `dimension`.

    The input to this layer is expected to be a tensor, or a batch of tensors;

    :param dimension to be split along this dimension
    :param T Numeric type. Only support float/double now

    >>> bifurcateSplitTable = BifurcateSplitTable(1)
    creating: createBifurcateSplitTable
    '''

    def __init__(self,
                 dimension,
                 bigdl_type="float"):
        super(BifurcateSplitTable, self).__init__(None, bigdl_type,
                                                  dimension)


class Bilinear(Layer):
    '''
    a bilinear transformation with sparse inputs,
    The input tensor given in forward(input) is a table containing both inputs x_1 and x_2,
    which are tensors of size N x inputDimension1 and N x inputDimension2, respectively.

    :param input_size1 input dimension of x_1
    :param input_size2 input dimension of x_2
    :param output_size output dimension
    :param bias_res whether use bias
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.

    >>> bilinear = Bilinear(1, 1, 1, True, L1Regularizer(0.5))
    creating: createL1Regularizer
    creating: createBilinear
    '''

    def __init__(self,
                 input_size1,
                 input_size2,
                 output_size,
                 bias_res=True,
                 wRegularizer=None,
                 bRegularizer=None,
                 bigdl_type="float"):
        super(Bilinear, self).__init__(None, bigdl_type,
                                       input_size1,
                                       input_size2,
                                       output_size,
                                       bias_res,
                                       wRegularizer,
                                       bRegularizer)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.


    >>> cAdd = CAdd([1,2])
    creating: createCAdd
    '''

    def __init__(self,
                 size, bRegularizer=None,
                 bigdl_type="float"):
        super(CAdd, self).__init__(None, bigdl_type,
                                   size, bRegularizer)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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


class CAveTable(Layer):
    '''
    Merge the input tensors in the input table by element wise taking the average. The input
    table is actually an array of tensor with same size.


    :param inplace: reuse the input memory


    >>> cAveTable = CAveTable(True)
    creating: createCAveTable
    '''

    def __init__(self,
                 inplace=False,
                 bigdl_type="float"):
        super(CAveTable, self).__init__(None, bigdl_type,
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
                 wRegularizer=None,
                 bigdl_type="float"):
        super(CMul, self).__init__(None, bigdl_type,
                                   size, wRegularizer)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class CosineDistance(Layer):
    '''
    Outputs the cosine distance between inputs


    >>> cosineDistance = CosineDistance()
    creating: createCosineDistance
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(CosineDistance, self).__init__(None, bigdl_type)


class CrossProduct(Layer):
    """
    A layer which takes a table of multiple tensors(n >= 2) as input
    and calculate to dot product for `all combinations of pairs` among input tensors.

    Dot-product outputs are ordered according to orders of pairs in input Table.
    For instance, input (Table) is T(A, B, C), output (Tensor) will be [A.*B, A.*C, B.*C].

    Dimensions of input' Tensors could be one or two, if two, first dimension is `batchSize`.
    For convenience, output is 2-dim Tensor regardless of input' dims.

    Table size checking and Tensor size checking will be execute before each forward,
    when [[numTensor]] and [[embeddingSize]] are set values greater than zero.

    :param numTensor (for checking)number of Tensor input Table contains, :default 0(won't check)
    :param embeddingSize (for checking)vector length of dot product, :default 0(won't check)

    >>> crossProduct = CrossProduct()
    creating: createCrossProduct
    """

    def __init__(self,
                 numTensor=0,
                 embeddingSize=0,
                 bigdl_type="float"):
        super(CrossProduct, self).__init__(None, bigdl_type, numTensor, embeddingSize)


class UpSampling2D(Layer):
    """
    Upsampling layer for 2D inputs.
    Repeats the heights and widths of the data by size[0] and size[1] respectively.

    If input's dataformat is NCHW, then the size of output is (N, C, H * size[0], W * size[1])

    :param size tuple of 2 integers. The upsampling factors for heights and widths.
    :param format DataFormat, NCHW or NHWC

    >>> upsampled2d = UpSampling2D([2, 3])
    creating: createUpSampling2D
    """

    def __init__(self, size, data_format="nchw", bigdl_type="float"):
        super(UpSampling2D, self).__init__(None, bigdl_type, size, data_format)


class UpSampling1D(Layer):
    """
    Upsampling layer for 1D inputs.
    Repeats each temporal step length times along the time axis.

    If input's size is (batch, steps, features),
    then the output's size is (batch, steps * length, features)

    :param length integer, upsampling factor.
    >>> upsampled1d = UpSampling1D(2)
    creating: createUpSampling1D
    """

    def __init__(self, length, bigdl_type="float"):
        super(UpSampling1D, self).__init__(None, bigdl_type, length)


class Input(Node):
    '''
    Input layer do nothing to the input tensors, just passing them through. It is used as input to
    the Graph container (add a link) when the first layer of the graph container accepts multiple
    tensors as inputs.

    Each input node of the graph container should accept one tensor as input. If you want a module
    accepting multiple tensors as input, you should add some Input module before it and connect
    the outputs of the Input nodes to it.

    Please note that the return is not a layer but a Node containing input layer.

    >>> input = Input()
    creating: createInput
    '''

    def __init__(self,
                 bigdl_type="float"):
        super(Input, self).__init__(None, bigdl_type)


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

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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
    Reshape the input tensor with automatic size inference support.
    Positive numbers in the `size` argument are used to reshape the input to the
    corresponding dimension size.
    There are also two special values allowed in `size`:
       a. `0` means keep the corresponding dimension size of the input unchanged.
          i.e., if the 1st dimension size of the input is 2,
          the 1st dimension size of output will be set as 2 as well.
       b. `-1` means infer this dimension size from other dimensions.
          This dimension size is calculated by keeping the amount of output elements
          consistent with the input.
          Only one `-1` is allowable in `size`.

    For example,
       Input tensor with size: (4, 5, 6, 7)
       -> InferReshape(Array(4, 0, 3, -1))
       Output tensor with size: (4, 5, 3, 14)
    The 1st and 3rd dim are set to given sizes, keep the 2nd dim unchanged,
    and inferred the last dim as 14.

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
    :param nInputDims: specify the number of dimensions that this module will receiveIf it is more
                       than the dimension of input tensors, the first dimensionwould be considered
                       as batch size


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


class SparseJoinTable(Layer):
    '''
    :: Experimental ::

    Sparse version of JoinTable. Backward just pass the origin gradOutput back to
    the next layers without split. So this layer may just works in Wide&Deep like models.


    :param dimension: to be join in this dimension


    >>> joinTable = SparseJoinTable(1)
    creating: createSparseJoinTable
    '''

    def __init__(self,
                 dimension,
                 bigdl_type="float"):
        super(SparseJoinTable, self).__init__(None, bigdl_type,
                                              dimension)


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


class NegativeEntropyPenalty(Layer):
    '''
    Penalize the input multinomial distribution if it has low entropy.
    The input to this layer should be a batch of vector each representing a
    multinomial distribution. The input is typically the output of a softmax layer.

    For forward, the output is the same as input and a NegativeEntropy loss of
    the latent state will be calculated each time. For backward,
    gradInput = gradOutput + gradLoss

    This can be used in reinforcement learning to discourage the policy from
    collapsing to a single action for a given state, which improves exploration.
    See the A3C paper for more detail (https://arxiv.org/pdf/1602.01783.pdf).

    >>> ne = NegativeEntropyPenalty(0.01)
    creating: createNegativeEntropyPenalty

    :param beta penalty coefficient
    '''

    def __init__(self, beta=0.01, bigdl_type="float"):
        super(NegativeEntropyPenalty, self).__init__(None,
                                                     bigdl_type,
                                                     beta)


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

    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.

    >>> lookupTable = LookupTable(1, 1, 1e-5, 1e-5, 1e-5, True, L1Regularizer(0.5))
    creating: createL1Regularizer
    creating: createLookupTable
    '''

    def __init__(self,
                 n_index,
                 n_output,
                 padding_value=0.0,
                 max_norm=DOUBLEMAX,
                 norm_type=2.0,
                 should_scale_grad_by_freq=False,
                 wRegularizer=None,
                 bigdl_type="float"):
        super(LookupTable, self).__init__(None, bigdl_type,
                                          n_index,
                                          n_output,
                                          padding_value,
                                          max_norm,
                                          norm_type,
                                          should_scale_grad_by_freq,
                                          wRegularizer)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class LookupTableSparse(Layer):
    '''
    LookupTable for multi-values.
    Also called embedding_lookup_sparse in TensorFlow.

    The input of LookupTableSparse should be a 2D SparseTensor or two 2D SparseTensors.
    If the input is a SparseTensor, the values are positive integer ids,
    values in each row of this SparseTensor will be turned into a dense vector.
    If the input is two SparseTensors, the first tensor should be the integer ids, just
    like the SparseTensor input. And the second tensor is the corresponding
    weights of the integer ids.

    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.

    >>> lookupTableSparse = LookupTableSparse(20, 5, "mean", 2, L1Regularizer(0.5))
    creating: createL1Regularizer
    creating: createLookupTableSparse
    >>> indices = np.array([[0, 0, 1, 2], [0, 1, 0, 3]])
    >>> values = np.array([2, 4, 1, 2])
    >>> weightValues = np.array([2, 0.5, 1, 3])
    >>> input = JTensor.sparse(values, indices, np.array([3, 4]))
    >>> weight = JTensor.sparse(weightValues, indices, np.array([3, 4]))
    >>> layer1 = LookupTableSparse(10, 4, "mean")
    creating: createLookupTableSparse
    >>> layer1.set_weights(np.arange(1, 41, 1).reshape(10, 4)) # set weight to 1 to 40
    >>> output = layer1.forward([input, weight])
    >>> expected_output = np.array([[6.5999999 , 7.60000038, 8.60000038, 9.60000038],
    ... [ 1., 2., 3., 4.], [5., 6., 7., 8.]])
    >>> np.testing.assert_allclose(output, expected_output, rtol=1e-6, atol=1e-6)
    '''

    def __init__(self,
                 n_index,
                 n_output,
                 combiner="sum",
                 max_norm=-1.0,
                 wRegularizer=None,
                 bigdl_type="float"):
        super(LookupTableSparse, self).__init__(None, bigdl_type,
                                                n_index,
                                                n_output,
                                                combiner,
                                                max_norm + 0.0,
                                                wRegularizer)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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
                 module=None,
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
    :param n_input_dims: specify the number of dimensions that this module will receiveIf it is more
    than the dimension of input tensors, the first dimension would be consideredas batch size
    :param squeeze: default is true, which will squeeze the sum dimension; set it to false to keep
    the sum dimension

    >>> mean = Mean(1, 1, True)
    creating: createMean
    '''

    def __init__(self,
                 dimension=1,
                 n_input_dims=-1,
                 squeeze=True,
                 bigdl_type="float"):
        super(Mean, self).__init__(None, bigdl_type,
                                   dimension,
                                   n_input_dims,
                                   squeeze)


class Min(Layer):
    '''
    Applies a min operation over dimension `dim`.


    :param dim: min along this dimension
    :param num_input_dims: Optional. If in a batch model, set to the input_dim.


    >>> min = Min(1)
    creating: createMin
    '''

    def __init__(self,
                 dim=1,
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

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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
    :param n_input_dim: specify the number of dimensions that this module will receiveIf it is more
     than the dimension of input tensors, the first dimensionwould be considered as batch size
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

    In training mode negative inputs are multiplied by a factor drawn from a uniform random
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
                 lower=1.0 / 8,
                 upper=1.0 / 3,
                 inplace=False,
                 bigdl_type="float"):
        super(RReLU, self).__init__(None, bigdl_type,
                                    lower,
                                    upper,
                                    inplace)


class SpatialSeparableConvolution(Layer):
    '''
    Separable convolutions consist in first performing a depthwise spatial convolution (which acts
    on each input channel separately) followed by a pointwise convolution which mixes together the
    resulting output channels. The  depth_multiplier argument controls how many output channels are
    generated per input channel in the depthwise step.

    :param n_input_channel The number of expected input planes in the image given into forward()
    :param n_output_channel The number of output planes the convolution layer will produce.
    :param depth_multiplier how many internal channels are generated per input channel
    :param kernel_w The kernel width of the convolution
    :param kernel_h The kernel height of the convolution
    :param stride_w The step of the convolution in the width dimension.
    :param stride_h The step of the convolution in the height dimension
    :param pad_w The additional zeros added per width to the input planes.
    :param pad_h The additional zeros added per height to the input planes.
    :param with_bias: the optional initial value for if need bias
    :param data_format: a string value of "NHWC" or "NCHW" to specify the input data format of this
                       layer. In "NHWC" format
                       data is stored in the order of [batch_size, height, width, channels], in
                       "NCHW" format data is stored
                       in the order of [batch_size, channels, height, width].
    :param w_regularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                          depth weights matrices.
    :param b_regularizer: instance of [[Regularizer]]applied to the pointwise bias.
    :param p_regularizer: instance of [[Regularizer]]applied to the pointwise weights.

    >>> conv = SpatialSeparableConvolution(6, 12, 1, 5, 5)
    creating: createSpatialSeparableConvolution
    >>> conv.setWRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    >>> conv.setBRegularizer(L1Regularizer(0.5))
    creating: createL1Regularizer
    >>> conv = SpatialSeparableConvolution(6, 12, 1, 5, 5, 1, 1, 0, 0, True, "NCHW",
    ... L1Regularizer(0.5), L1Regularizer(0.5), L1Regularizer(0.5))
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createSpatialSeparableConvolution
    '''

    def __init__(self,
                 n_input_channel,
                 n_output_channel,
                 depth_multiplier,
                 kernel_w,
                 kernel_h,
                 stride_w=1,
                 stride_h=1,
                 pad_w=0,
                 pad_h=0,
                 with_bias=True,
                 data_format="NCHW",
                 w_regularizer=None,
                 b_regularizer=None,
                 p_regularizer=None,
                 bigdl_type="float"):
        super(SpatialSeparableConvolution, self).__init__(None, bigdl_type,
                                                          n_input_channel,
                                                          n_output_channel,
                                                          depth_multiplier,
                                                          kernel_w,
                                                          kernel_h,
                                                          stride_w,
                                                          stride_h,
                                                          pad_w,
                                                          pad_h,
                                                          with_bias,
                                                          data_format,
                                                          w_regularizer,
                                                          b_regularizer,
                                                          p_regularizer,
                                                          )


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


class SReLU(Layer):
    '''S-shaped Rectified Linear Unit.

    It follows:
    `f(x) = t^r + a^r(x - t^r) for x >= t^r`,
    `f(x) = x for t^r > x > t^l`,
    `f(x) = t^l + a^l(x - t^l) for x <= t^l`.

    # References
        - [Deep Learning with S-shaped Rectified Linear Activation Units](http://arxiv.org/abs/
          1512.07030)

    :param input_shape: shape for tleft, aleft, tright, aright.
            E.g. for a 4-D input, the shape is the last 3-D
    :param shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.

    >>> srelu = SReLU((2, 3))
    creating: createSReLU
    >>> srelu = SReLU((2, 2), (1, 2))
    creating: createSReLU
    >>> from bigdl.dllib.nn.initialization_method import Xavier
    >>> init = Xavier()
    creating: createXavier
    >>> srelu = srelu.set_init_method(tLeftInit=init, aLeftInit=init, tRightInit=init,
    ... aRightInit=init)
    '''

    def __init__(self,
                 input_shape,
                 share_axes=None,
                 bigdl_type="float"):
        super(SReLU, self).__init__(None, bigdl_type, input_shape,
                                    share_axes)

    def set_init_method(self, tLeftInit=None, aLeftInit=None,
                        tRightInit=None, aRightInit=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      [tLeftInit, aLeftInit, tRightInit, aRightInit])
        return self


class ActivityRegularization(Layer):
    '''
    Layer that applies an update to the cost function based input activity.

    :param l1: L1 regularization factor (positive float).
    :param l2: L2 regularization factor (positive float).


    >>> ar = ActivityRegularization(0.1, 0.02)
    creating: createActivityRegularization
    '''

    def __init__(self,
                 l1=0.0,
                 l2=0.0,
                 bigdl_type="float"):
        super(ActivityRegularization, self).__init__(None, bigdl_type, l1, l2)


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


    >>> import numpy as np
    >>> input_data = np.random.rand(2,2,6,8)
    >>> input_rois = np.array([0, 0, 0, 7, 5, 1, 6, 2, 7, 5, 1, 3, 1, 6, 4, 0, 3, 3, 3, 3],
    ... dtype='float64').reshape(4,5)
    >>> m = RoiPooling(3,2,1.0)
    creating: createRoiPooling
    >>> out = m.forward([input_data,input_rois])
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


    :param index: the index to be selected


    >>> selectTable = SelectTable(1)
    creating: createSelectTable
    '''

    def __init__(self,
                 index,
                 bigdl_type="float"):
        super(SelectTable, self).__init__(None, bigdl_type,
                                          index)


class SequenceBeamSearch(Layer):
    '''
    Find the translated sequence with the highest probability.


    :param vocab_size: size of tokens
    :param beam_size: number of beams
    :param alpha: defining the strength of length normalization
    :param decode_length: maximum length to decoded sequence
    :param eos_id: id of eos token, used to determine when a sequence has finished
    :param padding_value
    :param num_hidden_layers: number of hidden layers
    :param hidden_size: size of hidden layer


    >>> sequenceBeamSearch = SequenceBeamSearch(4, 3, 0.0, 10, 2.0, 1.0, 2, 5)
    creating: createSequenceBeamSearch
    '''

    def __init__(self,
                 vocab_size,
                 beam_size,
                 alpha,
                 decode_length,
                 eos_id,
                 padding_value,
                 num_hidden_layers,
                 hidden_size,
                 bigdl_type="float"):
        super(SequenceBeamSearch, self).__init__(None, bigdl_type,
                                                 vocab_size,
                                                 beam_size,
                                                 alpha,
                                                 decode_length,
                                                 eos_id,
                                                 padding_value,
                                                 num_hidden_layers,
                                                 hidden_size)


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
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.


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
                 wRegularizer=None,
                 bRegularizer=None,
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
                                                        wRegularizer,
                                                        bRegularizer)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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


    Other frameworks call this operation "In-network Upsampling", "Fractionally-strided
     convolution", "Backwards Convolution," "Deconvolution", or "Upconvolution."


    Reference Paper: Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic
    segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
    2015: 3431-3440.

    :param nInputPlane The number of expected input planes in the image given into forward()
    :param nOutputPlane The number of output planes the convolution layer will produce.
    :param kW The kernel width of the convolution.
    :param kH The kernel height of the convolution.
    :param dW The step of the convolution in the width dimension. Default is 1.
    :param dH The step of the convolution in the height dimension. Default is 1.
    :param padW The additional zeros added per width to the input planes. Default is 0.
    :param padH The additional zeros added per height to the input planes. Default is 0.
    :param adjW Extra width to add to the output image. Default is 0.
    :param adjH Extra height to add to the output image. Default is 0.
    :param nGroup Kernel group number.
    :param noBias If bias is needed.
    :param initMethod Init method, Default, Xavier, Bilinear.
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.


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
                 wRegularizer=None,
                 bRegularizer=None,
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
                                                     wRegularizer,
                                                     bRegularizer)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class VolumetricFullConvolution(Layer):
    '''
    Apply a 3D full convolution over an 3D input image, a sequence of images, or a video etc.
    The input tensor is expected to be a 4D or 5D(with batch) tensor. Note that instead
    of setting adjT, adjW and adjH, `VolumetricFullConvolution` also accepts a table input
    with two tensors: T(convInput, sizeTensor) where convInput is the standard input tensor,
    and the size of sizeTensor is used to set the size of the output (will ignore the adjT, adjW and
    adjH values used to construct the module). This module can be used without a bias by setting
    parameter noBias = true while constructing the module.


    If input is a 4D tensor nInputPlane x depth x height x width,
    odepth = (depth  - 1) * dT - 2*padt + kT + adjT
    owidth  = (width  - 1) * dW - 2*padW + kW + adjW
    oheight = (height - 1) * dH - 2*padH + kH + adjH


    Other frameworks call this operation "In-network Upsampling", "Fractionally-strided
    convolution", "Backwards Convolution," "Deconvolution", or "Upconvolution."


    Reference Paper: Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic
    segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
    2015: 3431-3440.

    :param nInputPlane The number of expected input planes in the image given into forward()
    :param nOutputPlane The number of output planes the convolution layer will produce.
    :param kT The kernel depth of the convolution.
    :param kW The kernel width of the convolution.
    :param kH The kernel height of the convolution.
    :param dT The step of the convolution in the depth dimension. Default is 1.
    :param dW The step of the convolution in the width dimension. Default is 1.
    :param dH The step of the convolution in the height dimension. Default is 1.
    :param padT The additional zeros added per depth to the input planes. Default is 0.
    :param padW The additional zeros added per width to the input planes. Default is 0.
    :param padH The additional zeros added per height to the input planes. Default is 0.
    :param adjT Extra depth to add to the output image. Default is 0.
    :param adjW Extra width to add to the output image. Default is 0.
    :param adjH Extra height to add to the output image. Default is 0.
    :param nGroup Kernel group number.
    :param noBias If bias is needed.
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.


    >>> volumetricFullConvolution = VolumetricFullConvolution(1, 1, 1, 1, 1, 1)
    creating: createVolumetricFullConvolution
    '''

    def __init__(self,
                 n_input_plane,
                 n_output_plane,
                 kt,
                 kw,
                 kh,
                 dt=1,
                 dw=1,
                 dh=1,
                 pad_t=0,
                 pad_w=0,
                 pad_h=0,
                 adj_t=0,
                 adj_w=0,
                 adj_h=0,
                 n_group=1,
                 no_bias=False,
                 wRegularizer=None,
                 bRegularizer=None,
                 bigdl_type="float"):
        super(VolumetricFullConvolution, self).__init__(None, bigdl_type,
                                                        n_input_plane,
                                                        n_output_plane,
                                                        kt,
                                                        kw,
                                                        kh,
                                                        dt,
                                                        dw,
                                                        dh,
                                                        pad_t,
                                                        pad_w,
                                                        pad_h,
                                                        adj_t,
                                                        adj_w,
                                                        adj_h,
                                                        n_group,
                                                        no_bias,
                                                        wRegularizer,
                                                        bRegularizer)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class SpatialShareConvolution(Layer):
    '''

    >>> spatialShareConvolution = SpatialShareConvolution(1, 1, 1, 1)
    creating: createSpatialShareConvolution
    >>> import numpy as np
    >>> init_weight = np.random.randn(1, 12, 6, 5, 5)
    >>> init_bias = np.random.randn(12)
    >>> init_grad_weight = np.zeros([1, 12, 6, 5, 5])
    >>> init_grad_bias = np.zeros([12])
    >>> conv = SpatialShareConvolution(6, 12, 5, 5, 1, 1, 0, 0, 1, True, L1Regularizer(0.5),
    ... L1Regularizer(0.5), init_weight, init_bias, init_grad_weight, init_grad_bias)
    creating: createL1Regularizer
    creating: createL1Regularizer
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
                 wRegularizer=None,
                 bRegularizer=None,
                 init_weight=None,
                 init_bias=None,
                 init_grad_weight=None,
                 init_grad_bias=None,
                 with_bias=True,
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
                                                      wRegularizer,
                                                      bRegularizer,
                                                      JTensor.from_ndarray(init_weight),
                                                      JTensor.from_ndarray(init_bias),
                                                      JTensor.from_ndarray(init_grad_weight),
                                                      JTensor.from_ndarray(init_grad_bias),
                                                      with_bias)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


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
    :param pad_t: Additional zeros added to the input plane data on both sides of time axis.Default
                 is 0. (kT-1)/2 is often used here.
    :param pad_w: The additional zeros added per width to the input planes.
    :param pad_h: The additional zeros added per height to the input planes.
    :param with_bias: whether with bias
    :param wRegularizer: instance of [[Regularizer]] (eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param bRegularizer: instance of [[Regularizer]] applied to the bias.


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
                 wRegularizer=None,
                 bRegularizer=None,
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
                                                    wRegularizer,
                                                    bRegularizer)

    def set_init_method(self, weight_init_method=None, bias_init_method=None):
        callBigDlFunc(self.bigdl_type, "setInitMethod", self.value,
                      weight_init_method, bias_init_method)
        return self


class VolumetricMaxPooling(Layer):
    '''
    Applies 3D max-pooling operation in kTxkWxkH regions by step size dTxdWxdH.
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


class VolumetricAveragePooling(Layer):
    '''
    Applies 3D average-pooling operation in kTxkWxkH regions by step size dTxdWxdH.
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
    :param count_include_pad: whether to include padding when dividing the number of elements in
                              pooling region
    :param ceil_mode: whether the output size is to be ceiled or floored


    >>> volumetricAveragePooling = VolumetricAveragePooling(5, 5, 5, 1, 1, 1)
    creating: createVolumetricAveragePooling
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
                 count_include_pad=True,
                 ceil_mode=False,
                 bigdl_type="float"):
        super(VolumetricAveragePooling, self).__init__(None, bigdl_type,
                                                       k_t,
                                                       k_w,
                                                       k_h,
                                                       d_t,
                                                       d_w,
                                                       d_h,
                                                       pad_t,
                                                       pad_w,
                                                       pad_h,
                                                       count_include_pad,
                                                       ceil_mode)


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
    the specified dimension `dimension`. Please note the dimension starts from 1.


    The input to this layer is expected to be a tensor, or a batch of tensors;
    when using mini-batch, a batch of sample tensors will be passed to the layer and
    the user needs to specify the number of dimensions of each sample tensor in a
    batch using `nInputDims`.


    :param dimension: to be split along this dimension
    :param n_input_dims: specify the number of dimensions that this module will receiveIf it is more
    than the dimension of input tensors, the first dimensionwould be considered as batch size


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
    :param n_input_dims: specify the number of dimensions that this module will receiveIf it is more
     than the dimension of input tensors, the first dimensionwould be considered as batch size
    :param size_average: default is false, if it is true, it will return the mean instead
    :param squeeze: default is true, which will squeeze the sum dimension; set it to false to keep
     the sum dimension


    >>> sum = Sum(1, 1, True, True)
    creating: createSum
    '''

    def __init__(self,
                 dimension=1,
                 n_input_dims=-1,
                 size_average=False,
                 squeeze=True,
                 bigdl_type="float"):
        super(Sum, self).__init__(None, bigdl_type,
                                  dimension,
                                  n_input_dims,
                                  squeeze,
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


class Negative(Layer):
    '''
    Create an Negative layer.  Computing negative value of each element of input tensor

    :param inplace: if output tensor reuse input tensor storage. Default value is false


    >>> negative = Negative(False)
    creating: createNegative
    '''

    def __init__(self,
                 inplace=False,
                 bigdl_type="float"):
        super(Negative, self).__init__(None, bigdl_type, inplace)


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

    def __init__(self, pos, num_input_dims=INTMIN, bigdl_type="float"):
        if isinstance(pos, int):
            posList = [pos]
            super(Unsqueeze, self).__init__(None, bigdl_type, to_list(posList), num_input_dims)
        elif isinstance(pos, list):
            super(Unsqueeze, self).__init__(None, bigdl_type, to_list(pos), num_input_dims)
        else:
            invalidInputError(False, "Error invalid input")


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
    >>> reverse = Reverse(1, False)
    creating: createReverse
    '''

    def __init__(self,
                 dimension=1,
                 is_inplace=False,
                 bigdl_type="float"):
        super(Reverse, self).__init__(None, bigdl_type,
                                      dimension,
                                      is_inplace)


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

    When padW and padH are both -1, we use a padding algorithm similar to the "SAME"
    padding of tensorflow. That is

     outHeight = Math.ceil(inHeight.toFloat/strideH.toFloat)
     outWidth = Math.ceil(inWidth.toFloat/strideW.toFloat)

     padAlongHeight = Math.max(0, (outHeight - 1) * strideH + kernelH - inHeight)
     padAlongWidth = Math.max(0, (outWidth - 1) * strideW + kernelW - inWidth)

     padTop = padAlongHeight / 2
     padLeft = padAlongWidth / 2

    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
     input weights matrices.
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.

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
                 wRegularizer=None,
                 bRegularizer=None,
                 bigdl_type="float"):
        super(SpatialConvolutionMap, self).__init__(None, bigdl_type,
                                                    JTensor.from_ndarray(conn_table),
                                                    kw,
                                                    kh,
                                                    dw,
                                                    dh,
                                                    pad_w,
                                                    pad_h,
                                                    wRegularizer,
                                                    bRegularizer)


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


class SpatialWithinChannelLRN(Layer):
    '''
    The local response normalization layer performs a kind of lateral inhibition
    by normalizing over local input regions. the local regions extend spatially,
    in separate channels (i.e., they have shape 1 x local_size x local_size).

    :param size  the side length of the square region to sum over
    :param alpha the scaling parameter
    :param beta the exponent


    >>> layer = SpatialWithinChannelLRN()
    creating: createSpatialWithinChannelLRN
    '''

    def __init__(self,
                 size=5,
                 alpha=1.0,
                 beta=0.75,
                 bigdl_type="float"):
        super(SpatialWithinChannelLRN, self).__init__(None, bigdl_type,
                                                      size,
                                                      alpha,
                                                      beta)


class Pack(Layer):
    '''
    Stacks a list of n-dimensional tensors into one (n+1)-dimensional tensor.

    >>> layer = Pack(1)
    creating: createPack
    '''

    def __init__(self, dimension, bigdl_type="float"):
        super(Pack, self).__init__(None, bigdl_type, dimension)


class ConvLSTMPeephole(Layer):
    '''

|   Convolution Long Short Term Memory architecture with peephole.
|   Ref. A.: https://arxiv.org/abs/1506.04214 (blueprint for this module)
|   B. https://github.com/viorik/ConvLSTM

    :param input_size: number of input planes in the image given into forward()
    :param output_size: number of output planes the convolution layer will produce
    :param kernel_i: Convolutional filter size to convolve input
    :param kernel_c: Convolutional filter size to convolve cell
    :param stride: The step of the convolution, default is 1
    :param padding: The additional zeros added, default is -1
    :param activation: activation function, by default to be Tanh if not specified.
                        It can also be the name of an existing activation as a string.
    :param inner_activation: activation function for the inner cells, by default to be Sigmoid if
                             not specified.
                             It can also be the name of an existing activation as a string.
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices
    :param uRegularizer: instance [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         recurrent weights matrices
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.
    :param cRegularizer: instance of [[Regularizer]]applied to peephole.
    :param with_peephole: whether use last cell status control a gate.

    >>> convlstm = ConvLSTMPeephole(4, 3, 3, 3, 1, -1, Tanh(), HardSigmoid(), L1Regularizer(0.5),
    ... L1Regularizer(0.5), L1Regularizer(0.5), L1Regularizer(0.5))
    creating: createTanh
    creating: createHardSigmoid
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createConvLSTMPeephole
    '''

    def __init__(self, input_size, output_size, kernel_i, kernel_c, stride=1, padding=-1,
                 activation=None, inner_activation=None,
                 wRegularizer=None, uRegularizer=None, bRegularizer=None, cRegularizer=None,
                 with_peephole=True, bigdl_type="float"):
        if not activation:
            activation = Tanh()
        if not inner_activation:
            inner_activation = Sigmoid()
        if isinstance(activation, six.string_types):
            activation = get_activation_by_name(activation)
        if isinstance(inner_activation, six.string_types):
            inner_activation = get_activation_by_name(inner_activation)
        super(ConvLSTMPeephole, self).__init__(None, bigdl_type, input_size, output_size, kernel_i,
                                               kernel_c,
                                               stride, padding, activation, inner_activation,
                                               wRegularizer, uRegularizer, bRegularizer,
                                               cRegularizer, with_peephole)


class Tile(Layer):
    '''
    Replicate 'copies' copy along 'dim' dimension

    >>> layer = Tile(1, 2)
    creating: createTile
    '''

    def __init__(self, dim=1, copies=2, bigdl_type="float"):
        super(Tile, self).__init__(None, bigdl_type, dim, copies)


class BinaryThreshold(Layer):
    '''
    Binary threshold, 1 if value > th, 0 otherwise
    >>> layer = BinaryThreshold(0.1, False)
    creating: createBinaryThreshold
    '''

    def __init__(self, th=1e-6, ip=False, bigdl_type="float"):
        super(BinaryThreshold, self).__init__(None, bigdl_type, th, ip)


class ConvLSTMPeephole3D(Layer):
    '''

    :param input_size: number of input planes in the image given into forward()
    :param output_size: number of output planes the convolution layer will produce
    :param kernel_i Convolutional filter size to convolve input
    :param kernel_c Convolutional filter size to convolve cell
    :param stride The step of the convolution
    :param padding The additional zeros added
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices
    :param uRegularizer: instance [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         recurrent weights matrices
    :param bRegularizer: instance of [[Regularizer]]applied to the bias.
    :param cRegularizer: instance of [[Regularizer]]applied to peephole.
    :param with_peephole: whether use last cell status control a gate.

    >>> convlstm = ConvLSTMPeephole3D(4, 3, 3, 3, 1, -1, L1Regularizer(0.5), L1Regularizer(0.5),
    ... L1Regularizer(0.5), L1Regularizer(0.5))
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createL1Regularizer
    creating: createConvLSTMPeephole3D
    '''

    def __init__(self, input_size, output_size, kernel_i, kernel_c, stride=1, padding=-1,
                 wRegularizer=None, uRegularizer=None,
                 bRegularizer=None, cRegularizer=None, with_peephole=True, bigdl_type="float"):
        super(ConvLSTMPeephole3D, self).__init__(None, bigdl_type, input_size, output_size,
                                                 kernel_i, kernel_c, stride,
                                                 padding, wRegularizer, uRegularizer, bRegularizer,
                                                 cRegularizer, with_peephole)


class MultiRNNCell(Layer):
    '''
    A cell that enables stack multiple simple rnn cells

    >>> cells = []
    >>> cells.append(ConvLSTMPeephole3D(4, 3, 3, 3, 1))
    creating: createConvLSTMPeephole3D
    >>> cells.append(ConvLSTMPeephole3D(4, 3, 3, 3, 1))
    creating: createConvLSTMPeephole3D
    >>> stacked_convlstm = MultiRNNCell(cells)
    creating: createMultiRNNCell
    '''

    def __init__(self, cells, bigdl_type="float"):
        super(MultiRNNCell, self).__init__(None, bigdl_type, cells)


class ResizeBilinear(Layer):
    """
    Resize the input image with bilinear interpolation. The input image must be a float tensor with
    NHWC or NCHW layout

    :param output_height: output height
    :param output_width: output width
    :param align_corner: align corner or not
    :param data_format: the data format of the input image, NHWC or NCHW

    >>> resizeBilinear = ResizeBilinear(10, 20, False, "NCHW")
    creating: createResizeBilinear
    """

    def __init__(self, output_height, output_width, align_corner=False, data_format="NCHW",
                 bigdl_type="float"):
        super(ResizeBilinear, self).__init__(None, bigdl_type, output_height,
                                             output_width, align_corner, data_format)


class GaussianSampler(Layer):
    """
    Takes {mean, log_variance} as input and samples from the Gaussian distribution
    >>> sampler = GaussianSampler()
    creating: createGaussianSampler
    """

    def __init__(self, bigdl_type="float"):
        super(GaussianSampler, self).__init__(None, bigdl_type)


class Masking(Layer):
    '''
    Use a mask value to skip timesteps for a sequence
    ```
   :param mask_value: mask value

    >>> masking = Masking(0.0)
    creating: createMasking
    '''

    def __init__(self,
                 mask_value,
                 bigdl_type="float"):
        super(Masking, self).__init__(None, bigdl_type,
                                      mask_value)


class Maxout(Layer):
    '''
    A linear maxout layer Maxout layer select the element-wise maximum value of
    maxoutNumber Linear(inputSize, outputSize) layers
    ```
    :param input_size: the size the each input sample
    :param output_size: the size of the module output of each sample
    :param maxout_number: number of Linear layers to use
    :param with_bias: whether use bias in Linear
    :param w_regularizer: instance of [[Regularizer]]
          (eg. L1 or L2 regularization), applied to the input weights matrices.
    :param b_regularizer: instance of [[Regularizer]]
           applied to the bias.
    :param init_weight: initial weight
    :param init_bias: initial bias

    >>> maxout = Maxout(2, 5, 3)
    creating: createMaxout
    '''

    def __init__(self,
                 input_size,
                 output_size,
                 maxout_number,
                 with_bias=True,
                 w_regularizer=None,
                 b_regularizer=None,
                 init_weight=None,
                 init_bias=None,
                 bigdl_type="float"):
        super(Maxout, self).__init__(None, bigdl_type,
                                     input_size, output_size, maxout_number, with_bias,
                                     w_regularizer, b_regularizer, init_weight, init_bias)


class HardSigmoid(Layer):
    """
    Apply Hard-sigmoid function
```
               |  0, if x < -2.5
        f(x) = |  1, if x > 2.5
               |  0.2 * x + 0.5, otherwise
```
    >>> hardSigmoid = HardSigmoid()
    creating: createHardSigmoid
    """

    def __init__(self, bigdl_type="float"):
        super(HardSigmoid, self).__init__(None, bigdl_type)


class Highway(Layer):
    """
    Densely connected highway network.
    Highway layers are a natural extension of LSTMs to feedforward networks.

    :param size input size
    :param with_bias whether to include a bias
    :param activation activation function. It can also be the name of an existing activation as a
    string.
    :param wRegularizer: instance of [[Regularizer]](eg. L1 or L2 regularization), applied to the
                         input weights matrices.
    :param bRegularizer: instance of [[Regularizer]], applied to the bias.

    >>> highway = Highway(2)
    creating: createHighway
    """

    def __init__(self, size, with_bias=True, activation=None, wRegularizer=None, bRegularizer=None,
                 bigdl_type="float"):
        if isinstance(activation, six.string_types):
            activation = get_activation_by_name(activation)
        super(Highway, self).__init__(None, bigdl_type, size, with_bias, activation, wRegularizer,
                                      bRegularizer)


class UpSampling3D(Layer):
    """
    Upsampling layer for 3D inputs.
    Repeats the 1st, 2nd and 3rd dimensions
    of the data by size[0], size[1] and size[2] respectively.
    The input data is assumed to be of the form `minibatch x channels x depth x height x width`.

    :param size Repeats the depth, height, width dimensions of the data by
    >>> upsample3d = UpSampling3D([1, 2, 3])
    creating: createUpSampling3D
    """

    def __init__(self, size, bigdl_type="float"):
        super(UpSampling3D, self).__init__(None, bigdl_type, size)


class PriorBox(Layer):
    """
    Generate the prior boxes of designated sizes and aspect ratios across
    all dimensions (H * W)
    Intended for use with MultiBox detection method to generate prior
    :param min_sizes minimum box size in pixels. can be multiple. required!
    :param max_sizes maximum box size in pixels. can be ignored or same as the # of min_size.
    :param aspect_ratios optional aspect ratios of the boxes. can be multiple
    :param is_flip optional bool, default true. if set, flip the aspect ratio.
    :param is_clip whether to clip the prior's coordidate such that it is within [0, 1]
    >>> layer = PriorBox([0.1])
    creating: createPriorBox
    """

    def __init__(self, min_sizes,
                 max_sizes=None,
                 aspect_ratios=None,
                 is_flip=True,
                 is_clip=False,
                 variances=None,
                 offset=0.5,
                 img_h=0,
                 img_w=0,
                 img_size=0,
                 step_h=0.0,
                 step_w=0.0,
                 step=0.0,
                 bigdl_type="float"):
        super(PriorBox, self).__init__(None, bigdl_type,
                                       min_sizes,
                                       max_sizes,
                                       aspect_ratios,
                                       is_flip,
                                       is_clip,
                                       variances,
                                       offset,
                                       img_h,
                                       img_w,
                                       img_size,
                                       step_h,
                                       step_w,
                                       step)


class NormalizeScale(Layer):
    """
    NormalizeScale is conposed of normalize and scale, this is equal to caffe Normalize layer
    :param p L_p norm
    :param eps smoothing parameter
    :param scale scale parameter
    :param size size of scale input
    :param w_regularizer weight regularizer
    >>> layer = NormalizeScale(2.0, scale = 20.0, size = [1, 5, 1, 1])
    creating: createNormalizeScale
    """

    def __init__(self, p, scale, size, w_regularizer=None, eps=1e-10,
                 bigdl_type="float"):
        super(NormalizeScale, self).__init__(None, bigdl_type, p, eps, scale, size, w_regularizer)


class Proposal(Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    rois: holds R regions of interest, each is a 5-tuple
    (n, x1, y1, x2, y2) specifying an image batch index n and a rectangle (x1, y1, x2, y2)
    scores: holds scores for R regions of interest
    >>> layer = Proposal(1000, 200, [0.1, 0.2], [2.0, 3.0])
    creating: createProposal
    """

    def __init__(self, pre_nms_topn, post_nms_topn, ratios, scales,
                 rpn_pre_nms_topn_train=12000, rpn_post_nms_topn_train=2000,
                 bigdl_type="float"):
        super(Proposal, self).__init__(None, bigdl_type,
                                       pre_nms_topn,
                                       post_nms_topn,
                                       ratios,
                                       scales,
                                       rpn_pre_nms_topn_train,
                                       rpn_post_nms_topn_train)


class DetectionOutputSSD(Layer):
    """
    Layer to Post-process SSD output
    :param n_classes number of classes
    :param share_location whether to share location, default is true
    :param bg_label background label
    :param nms_thresh nms threshold
    :param nms_topk nms topk
    :param keep_top_k result topk
    :param conf_thresh confidence threshold
    :param variance_encoded_in_target if variance is encoded in target,
    we simply need to retore the offset predictions,
    else if variance is encoded in bbox,
    we need to scale the offset accordingly.
    :param conf_post_process whether add some additional post process to confidence prediction
    >>> layer = DetectionOutputSSD()
    creating: createDetectionOutputSSD
    """

    def __init__(self, n_classes=21,
                 share_location=True,
                 bg_label=0,
                 nms_thresh=0.45,
                 nms_topk=400,
                 keep_top_k=200,
                 conf_thresh=0.01,
                 variance_encoded_in_target=False,
                 conf_post_process=True,
                 bigdl_type="float"):
        super(DetectionOutputSSD, self).__init__(None,
                                                 bigdl_type,
                                                 n_classes,
                                                 share_location,
                                                 bg_label,
                                                 nms_thresh,
                                                 nms_topk,
                                                 keep_top_k,
                                                 conf_thresh,
                                                 variance_encoded_in_target,
                                                 conf_post_process)


class DetectionOutputFrcnn(Layer):
    """
    Post process Faster-RCNN models
    :param nms_thresh nms threshold
    :param n_classes number of classes
    :param bbox_vote whether to vote for detections
    :param max_per_image limit max number of detections per image
    :param thresh score threshold
    >>> layer = DetectionOutputFrcnn(21, True)
    creating: createDetectionOutputFrcnn
    """

    def __init__(self, n_classes, bbox_vote, nms_thresh=0.3,
                 max_per_image=100, thresh=0.05,
                 bigdl_type="float"):
        super(DetectionOutputFrcnn, self).__init__(None, bigdl_type, nms_thresh,
                                                   n_classes,
                                                   bbox_vote,
                                                   max_per_image,
                                                   thresh)


class Cropping2D(Layer):
    """
    Cropping layer for 2D input (e.g. picture).
    It crops along spatial dimensions, i.e. width and height.

    # Input shape
        4D tensor with shape:
        `(batchSize, channels, first_axis_to_crop, second_axis_to_crop)`

    # Output shape
        4D tensor with shape:
        `(batchSize, channels, first_cropped_axis, second_cropped_axis)`

    :param heightCrop Array of length 2. How many units should be trimmed off at the beginning
                      and end of the height dimension.
    :param widthCrop Array of length 2. How many units should be trimmed off at the beginning
                      and end of the width dimension.
    :param data_format a string value (or DataFormat Object in Scala) of "NHWC" or "NCHW" to specify
                        the input data format of this layer. In "NHWC" format
                        data is stored in the order of [batch_size, height, width, channels], in
                        "NCHW" format data is stored
                        in the order of [batch_size, channels, height, width].
    >>> cropping2D = Cropping2D([1, 1], [2, 2])
    creating: createCropping2D
    """

    def __init__(self, heightCrop, widthCrop, data_format="NCHW", bigdl_type="float"):
        super(Cropping2D, self).__init__(None, bigdl_type, heightCrop, widthCrop, data_format)


class Cropping3D(Layer):
    """
    Cropping layer for 3D data (e.g. spatial or spatio-temporal).

    # Input shape
        5D tensor with shape:
        `(batchSize, channels, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`

    # Output shape
        5D tensor with shape:
        `(batchSize, channels, first_cropped_axis, second_cropped_axis, third_cropped_axis)`

    :param dim1Crop Array of length 2. How many units should be trimmed off at the beginning
                      and end of the first dimension.
    :param dim2Crop Array of length 2. How many units should be trimmed off at the beginning
                      and end of the second dimension.
    :param dim3Crop Array of length 2. How many units should be trimmed off at the beginning
                      and end of the third dimension.
    :param data_format a string value. "channel_first" or "channel_last"
    >>> cropping3D = Cropping3D([1, 1], [2, 2], [1, 1])
    creating: createCropping3D
    """

    def __init__(self, dim1Crop, dim2Crop, dim3Crop, data_format="channel_first",
                 bigdl_type="float"):
        super(Cropping3D, self).__init__(None, bigdl_type, dim1Crop, dim2Crop, dim3Crop,
                                         data_format)


class RoiAlign(Layer):
    """
    Region of interest aligning (RoIAlign) for Mask-RCNN

    The RoIAlign uses average pooling on bilinear-interpolated sub-windows to convert
    the features inside any valid region of interest into a small feature map with a
    fixed spatial extent of pooledH * pooledW (e.g., 7 * 7).

    An RoI is a rectangular window into a conv feature map.
    Each RoI is defined by a four-tuple (x1, y1, x2, y2) that specifies its
    top-left corner (x1, y1) and its bottom-right corner (x2, y2).

    RoIAlign works by dividing the h * w RoI window into an pooledH * pooledW grid of
    sub-windows of approximate size h/H * w/W. In each sub-window, compute exact values
    of input features at four regularly sampled locations, and then do average pooling on
    the values in each sub-window.

    Pooling is applied independently to each feature map channel

    :param spatial_scale:  spatial scale
    :param sampling_ratio: sampling ratio
    :param pooled_h:       spatial extent in height
    :param pooled_w:       spatial extent in width

    >>> import numpy as np
    >>> input_data = np.random.rand(1,2,6,8)
    >>> input_rois = np.array([0, 0, 7, 5, 6, 2, 7, 5, 3, 1, 6, 4, 3, 3, 3, 3],
    ... dtype='float').reshape(4,4)
    >>> m = RoiAlign(1.0,3,2,2)
    creating: createRoiAlign
    >>> out = m.forward([input_data,input_rois])
    """

    def __init__(self,
                 spatial_scale,
                 sampling_ratio,
                 pooled_h,
                 pooled_w,
                 bigdl_type="float"):
        super(RoiAlign, self).__init__(None, bigdl_type,
                                       spatial_scale,
                                       sampling_ratio,
                                       pooled_h,
                                       pooled_w)


class Pooler(Layer):
    """
    Pooler selects the feature map which matches the size of RoI for RoIAlign

    :param resolution:     the resolution of pooled feature maps. Height equals width.
    :param scales:         spatial scales of each feature map
    :param sampling_ratio: sampling ratio

    >>> import numpy as np
    >>> feature0 = np.random.rand(1,2,2,2)
    >>> feature1 = np.random.rand(1,2,4,4)
    >>> feature2 = np.random.rand(1,2,8,8)
    >>> features = [feature0, feature1, feature2]
    >>> input_rois = np.array([0, 0, 3, 3, 2, 2, 50, 50, 50, 50, 500, 500],
    ... dtype='float').reshape(3,4)
    >>> m = Pooler(2,[1.0, 0.5, 0.25],2)
    creating: createPooler
    >>> out = m.forward([features,input_rois])
    """

    def __init__(self,
                 resolution,
                 scales,
                 sampling_ratio,
                 bigdl_type="float"):
        super(Pooler, self).__init__(None, bigdl_type,
                                     resolution,
                                     scales,
                                     sampling_ratio)


class FPN(Layer):
    """
    Feature Pyramid Network (FPN) for Mask-RCNN

    :param in_channels_list:    number of channels of feature maps
    :param out_channels:        number of channels of FPN output
    :param top_blocks:          top blocks option
                                extra operation to be performed on the smallest
                                resolution FPN output, whose result is appended
                                to the result list
                                0 for null,
                                1 for using max pooling on the last level
                                2 for extra layers P6 and P7 in RetinaNet
    :param in_channels_of_p6p7     number of input channels of P6 P7
    :param out_channels_of_p6p7    number of output channels of P6 P7

    >>> import numpy as np
    >>> feature1 = np.random.rand(1,1,8,8)
    >>> feature2 = np.random.rand(1,2,4,4)
    >>> feature3 = np.random.rand(1,4,2,2)
    >>> m = FPN([1,2,4],2,2,4,2)
    creating: createFPN
    >>> out = m.forward([feature1, feature2, feature3])
    """

    def __init__(self,
                 in_channels_list,
                 out_channels,
                 top_blocks=0,
                 in_channels_of_p6p7=0,
                 out_channels_of_p6p7=0,
                 bigdl_type="float"):
        super(FPN, self).__init__(None, bigdl_type,
                                  in_channels_list,
                                  out_channels,
                                  top_blocks,
                                  in_channels_of_p6p7,
                                  out_channels_of_p6p7)


def _test():
    import doctest
    from pyspark import SparkContext
    from bigdl.dllib.nn import layer
    from bigdl.dllib.utils.common import init_engine
    from bigdl.dllib.utils.common import create_spark_conf
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
