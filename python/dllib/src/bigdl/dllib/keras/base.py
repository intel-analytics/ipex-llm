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

from bigdl.dllib.nn.layer import Layer
from bigdl.dllib.utils.common import *
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.utils.log4Error import *


if sys.version >= '3':
    long = int
    unicode = str


class ZooKerasCreator(JavaValue):
    def jvm_class_constructor(self):
        name = "createZooKeras" + self.__class__.__name__
        print("creating: " + name)
        return name


class ZooCallable(object):
    def __call__(self, x):
        """
        Some other modules point to current module
        :param x: input variables. x is either a Variable or list of Variable.
        :return: Variable containing current module
        """
        from bigdl.dllib.keras.autograd import Variable
        return Variable.from_jvalue(callZooFunc(self.bigdl_type,
                                                "connectInputs",
                                                self,
                                                to_list(x)))


class InferShape(JavaValue):
    def __init__(self, bigdl_type="float"):
        self.bigdl_type = bigdl_type

    @classmethod
    def __to_keras_shape(cls, shape):
        if shape[0] == -1:
            return tuple([None] + shape[1:])
        else:
            return tuple(shape)

    def __process_shape(self, shape):
        if len(shape) == 1:
            return self.__to_keras_shape(shape[0])
        else:
            return [self.__to_keras_shape(s) for s in shape]

    def get_input_shape(self):
        """
        Return a list of shape tuples if there are multiple inputs.
        Return one shape tuple otherwise.
        """
        input = callZooFunc(self.bigdl_type, "getInputShape",
                            self.value)
        return self.__process_shape(input)

    def get_output_shape(self):
        """
        Return a list of shape tuples if there are multiple outputs.
        Return one shape tuple otherwise.
        """
        output = callZooFunc(self.bigdl_type, "getOutputShape",
                             self.value)
        return self.__process_shape(output)


class ZooKerasLayer(ZooKerasCreator, ZooCallable, Layer, InferShape):
    def __init__(self, jvalue, *args, **kwargs):
        allowed_kwargs = {"name", "bigdl_type"}
        for kwarg in kwargs.keys():
            if kwarg not in allowed_kwargs:
                invalidInputError(False, "Wrong argument for the layer:", kwarg)
        bigdl_type = kwargs.get("bigdl_type")
        if not bigdl_type:
            bigdl_type = "float"
        super(ZooKerasCreator, self).__init__(jvalue, bigdl_type, *args)
        name = kwargs.get("name")
        if name:
            self.set_name(name)

    def get_weights_shape(self):
        """
        :return: None if without weights
        """
        jshapes = callZooFunc(self.bigdl_type, "zooGetWeightsShape",
                              self.value)
        return [tuple(jshape) for jshape in jshapes]

    def set_weights(self, weights):
        """
        Set weights for this layer

        :param weights: a list of numpy arrays which represent weight and bias
        """
        current_shapes = self.get_weights_shape()
        invalidInputError(len(current_shapes) == len(weights),
                          "The parameters number should be the same")
        for w, cws in zip(weights, current_shapes):
            invalidInputError(w.shape == cws,
                              "The shape of parameter should be the same,"
                              " but got %s, %s" % (w.shape, cws))

        tensors = [JTensor.from_ndarray(param, self.bigdl_type) for param in to_list(weights)]
        callZooFunc(self.bigdl_type, "zooSetWeights", self.value, tensors)

    @classmethod
    def of(cls, jvalue, bigdl_type="float"):
        return ZooKerasLayer(jvalue, bigdl_type)
