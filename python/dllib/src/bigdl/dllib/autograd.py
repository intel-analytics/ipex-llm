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

from zoo.pipeline.api.keras.engine.topology import ZooKerasCreator
from bigdl.util.common import callBigDlFunc
from bigdl.nn.criterion import Criterion
from bigdl.nn.layer import Layer


import sys

if sys.version >= '3':
    long = int
    unicode = str


def mean(a, axis=0, keepDims=False):
    return Variable.from_jvalue(callBigDlFunc("float", "mean", a, axis, keepDims))


def abs(a):
    return Variable.from_jvalue(callBigDlFunc("float", "abs", a))


def sum(a, axis=0, keepdims=False):
    return Variable.from_jvalue(callBigDlFunc("float", "sum", a, axis, keepdims))


def clip(a, min, max):
    return Variable.from_jvalue(callBigDlFunc("float", "clip", a, float(min), float(max)))


def square(a):
    return Variable.from_jvalue(callBigDlFunc("float", "square", a))


def sqrt(a):
    return Variable.from_jvalue(callBigDlFunc("float", "sqrt", a))


def maximum(a, b):
    return Variable.from_jvalue(callBigDlFunc("float", "maximum", a, b))


def mean(a, axis=0, keepDims=False):
    return Variable.from_jvalue(callBigDlFunc("float", "mean", a, axis, keepDims))


def log(a):
    return Variable.from_jvalue(callBigDlFunc("float", "log", a))

def epsilon():
    return Variable.from_jvalue(callBigDlFunc("float", "epsilon"))


class Variable(ZooKerasCreator):
    def __init__(self, input_shape, node=None, jvalue=None):
        if jvalue:
            self.value = jvalue
            self.bigdl_type = "float"
        else:
            if node:
                super(Variable, self).__init__(jvalue, "float", input_shape)
            else:
                super(Variable, self).__init__(jvalue, "float", input_shape)

    @staticmethod
    def from_jvalue(jvalue):
        return Variable(input_shape=None, node=None, jvalue=jvalue)

    def add(self, var):
        return Variable.from_jvalue(callBigDlFunc("float", "add", self, var)) # self.value.getClass().getSimpleName()

    def sub(self, var):
        return Variable.from_jvalue(callBigDlFunc("float", "sub", self, var))

    def __sub__(self, other):
        return self.sub(other)

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return Variable.from_jvalue(callBigDlFunc("float", "mul", self, other))


class CustomLoss(ZooKerasCreator):

    def __init__(self, loss_func, input_shape):
        """
        :param loss_func: a function which accept y_true and y_pred
        :param input_shape: a shape without batch dim.
        i.e input_shape=[3], then the feeding data would be [None, 3]
        """
        y_real = Variable(input_shape=input_shape)
        y_pred = Variable(input_shape=input_shape)
        loss_var = loss_func(y_real, y_pred)
        super(CustomLoss, self).__init__(None, "float", [y_real, y_pred], loss_var)

    def forward(self, y_true, y_pred):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        Takes an input object, and computes the corresponding loss of the criterion,
        compared with `target`

        :param input: ndarray or list of ndarray
        :param target: ndarray or list of ndarray
        :return: value of loss
        """
        input = y_pred
        target = y_true
        jinput, input_is_table = Layer.check_input(input)
        jtarget, target_is_table = Layer.check_input(target)
        output = callBigDlFunc(self.bigdl_type,
                               "criterionForward",
                               self.value,
                               jinput,
                               input_is_table,
                               jtarget,
                               target_is_table)
        return output

    def backward(self, y_true, y_pred):
        """
        NB: It's for debug only, please use optimizer.optimize() in production.
        Performs a back-propagation step through the criterion, with respect to the given input.

        :param input: ndarray or list of ndarray
        :param target: ndarray or list of ndarray
        :return: ndarray
        """
        input = y_pred
        target = y_true
        jinput, input_is_table = Layer.check_input(input)
        jtarget, target_is_table = Layer.check_input(target)
        output = callBigDlFunc(self.bigdl_type,
                               "criterionBackward",
                               self.value,
                               jinput,
                               input_is_table,
                               jtarget,
                               target_is_table)
        return Layer.convert_output(output)