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

import sys

from bigdl.nn.layer import Layer, Node
from bigdl.util.common import callBigDlFunc, to_list, JTensor

import zoo.pipeline.api.keras.base as kbase
from zoo.pipeline.api.keras.objectives import LossFunction
from zoo.pipeline.api.utils import remove_batch, toMultiShape

if sys.version >= '3':
    long = int
    unicode = str


def mean(x, axis=0, keepDims=False):
    """
    Mean of a variable, alongside the specified axis.
    :param x: A variable.
    :param axis: A list of integer. Axes to compute the mean.
    :param keepDims: A boolean, whether to keep the dimensions or not.
            If `keepDims` is `False`, the rank of the variable is reduced
            by 1 for each entry in `axis`. If `keepDims` is `True`,
            the reduced dimensions are retained with length 1.
    :return: A variable with the mean of elements of `x`.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "mean", x, axis, keepDims))


def abs(x):
    """
    Element-wise absolute value.
    :param x: A variable.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "abs", x))


def batch_dot(x, y, axes=1, normalize=False):
    """
    Operator that computes a dot product between samples in two tensors.

    E.g. if applied to two tensors `a` and `b` of shape `(batch_size, n)`,
    the output will be a tensor of shape `(batch_size, 1)`
    where each entry `i` will be the dot product between
    `a[i]` and `b[i]`.

    :param x: Shape should only be [batch, xx]
    :param y: Shape should only be [batch, xx]
    :param axes: Integer or tuple of integers,
                axis or axes along which to take the dot product.
    :param normalize: Whether to L2-normalize samples along the
                dot product axis before taking the dot product.
                If set to True, then the output of the dot product
                is the cosine proximity between the two samples.
    :return: A variable.
    """
    if not normalize:
        if isinstance(axes, int):
            axes = [axes] * 2
    return Variable.from_jvalue(callBigDlFunc("float", "batchDot", x, y, axes, normalize))


def l2_normalize(x, axis):
    """
    Normalizes a tensor wrt the L2 norm alongside the specified axis.
    :param x: A variable. Shape should only be [batch, xx]
    :param axis: axis along which to perform normalization.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "l2Normalize", x, int(axis)))


def sum(x, axis=0, keepDims=False):
    """
    Sum of the values in a a variable, alongside the specified axis.
    :param x: A variable.
    :param axis: An integer. Axes to compute the sum over.
    :param keepDims: A boolean, whether to keep the dimensions or not.
            If `keepDims` is `False`, the rank of the variable is reduced
            by 1 for each entry in `axis`. If `keepDims` is `True`,
            the reduced dimensions are retained with length 1.
    :return: A variable with sum of `x`.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "sum", x, axis, keepDims))


def stack(inputs, axis=1):
    """
    Stacks a list of rank `R` tensors into a rank `R+1` tensor.
    You should start from 1 as dim 0 is for batch.
    :param inputs: List of variables (tensors).
    :param axis: axis along which to perform stacking.
    :return:
    """
    return Variable.from_jvalue(callBigDlFunc("float", "stack", inputs, axis))


def expand_dims(x, axis):
    """
   Adds a 1-sized dimension at index "axis".
    :param x: a Variable to be expanded
    :param axis: axis Position where to add a new axis.
    The axis is 0 based and if you set the axis to 0, you would change the batch dim.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "expandDims", x, axis))


def clip(x, min, max):
    """
    Element-wise value clipping.
    :param x: A variable.
    :param min: Python float or integer.
    :param max: Python float or integer.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "clip", x, float(min), float(max)))


def contiguous(x):
    """
    Turn the output and grad to be contiguous for the input Variable
    :param x: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "contiguous", x))


def square(x):
    """
    Element-wise square.
    :param x: A variable.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "square", x))


def sqrt(x):
    """
    Element-wise square root.
    :param x: A variable.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "sqrt", x))


def exp(x):
    """
    Element-wise exponential.
    :param x: A variable.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "exp", x))


def maximum(x, y):
    """
    Element-wise maximum of two variables.
    :param x: A variable.
    :param y: A variable.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "maximum", x, y))


def log(x):
    """
    Element-wise log.
    :param x: A variable.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "log", x))


def pow(x, a):
    """
    Element-wise exponentiation.
    :param x: A variable.
    :param a: Python integer.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "pow", x, float(a)))


def epsilon():
    """
    Define the value of epsilon.
    :return: A value of type Double.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "epsilon"))


def neg(x):
    """
    Computes numerical negative value element-wise.
    :param x: A variable.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "neg", x))


def softsign(x):
    """
    Softsign of a variable.
    :param x: A variable.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "softsign", x))


def softplus(x):
    """
    Softplus of a variable.
    :param x: A variable.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "softplus", x))


def mm(x, y, axes=None):
    """
    Module to perform matrix multiplication on two mini-batch inputs,
    producing a mini-batch.
    :param x: A variable.
    :param y: A variable.
    :param axes: Axes along which to perform multiplication.
    :return: A variable.
    """
    return Variable.from_jvalue(callBigDlFunc("float", "mm", x, y, axes))


class VariableOperator(object):
    # TODO: we need to add a mapping for Shape here.
    def __to_batch_shape(cls, shape):
        return tuple([None] + shape[1:])

    def __process_shape(self, shape):
        if len(shape) == 1:
            return self.__to_batch_shape(shape[0])
        else:
            return [self.__to_batch_shape(s) for s in shape]

    def get_input_shape(self):
        return self.__process_shape(callBigDlFunc("float", "varGetInputShape", self))

    def get_output_shape(self):
        return self.__process_shape(callBigDlFunc("float", "varGetOutputShape", self))

    @property
    def shape(self):
        return self.get_output_shape()

    @staticmethod
    def from_jvalue(jvalue):
        return Variable(input_shape=None, node=None, jvalue=jvalue)

    def add(self, var):
        return Variable.from_jvalue(callBigDlFunc("float", "add", self, var))
        # self.value.getClass().getSimpleName()

    def sub(self, var):
        return Variable.from_jvalue(callBigDlFunc("float", "sub", self, var))

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return Variable.from_jvalue(callBigDlFunc("float", "sub", other, self))

    def __add__(self, other):
        return self.add(other)

    __radd__ = __add__

    def __mul__(self, other):
        return Variable.from_jvalue(callBigDlFunc("float", "mul", self, other))

    __rmul__ = __mul__

    def __div__(self, other):
        return Variable.from_jvalue(callBigDlFunc("float", "div", self, other))

    __truediv__ = __div__

    def __rdiv__(self, other):
        return Variable.from_jvalue(callBigDlFunc("float", "div", other, self))

    __rtruediv__ = __rdiv__

    def __neg__(self):
        return neg(self)

    def slice(self, dim, start_index, length):
        """
        Same as narrow in Torch.
        Slice the input with the number of dimensions not being reduced.
        The batch dimension needs to be unchanged.
        For example, if input is:
        1 2 3
        4 5 6
        slice(1, 1, 2) will give output
        2 3
        5 6
        slice(1, 2, -1) will give output
        3
        6
        :param  dim The dimension to narrow. 0-based index. Cannot narrow the batch dimension.
                -1 means the last dimension of the input.
        :param  startIndex Non-negative integer.
                The start index on the given dimension. 0-based index.
        :param length The length to be sliced. Default is 1.
        """
        return Variable.from_jvalue(
            callBigDlFunc("float", "slice", self, dim, start_index, length))

    def index_select(self, dim, index):
        """
           Select an index of the input in the given dim and return the subset part.
           The batch dimension needs to be unchanged.
           The selected dim would be remove after this operation.
           For example, if input is:
           1 2 3
           4 5 6
           Select(1, 1) will give output [2 5]
           Select(1, -1) will give output [3 6]
        :param dim: The dimension to select. 0-based index. Cannot select the batch dimension.
                -1 means the last dimension of the input.
        :param index: The index of the dimension to be selected. 0-based index.
               -1 means the last dimension of the input.
        :return:
        """
        return Variable.from_jvalue(callBigDlFunc("float", "indexSelect", self, dim, index))

    def squeeze(self, dim=None):
        """
        Delete the singleton dimension(s).
        The dim can be zero, and if so you would change the batch dim.
        For example, if input has size (2, 1, 3, 4, 1):
        Squeeze(dim = 1) will give output size (2, 3, 4, 1)
        Squeeze(dims = null) will give output size (2, 3, 4)
        """
        return Variable.from_jvalue(callBigDlFunc("float", "squeeze", self, dim))


class Variable(kbase.ZooKerasCreator, VariableOperator):
    def __init__(self, input_shape, node=None, jvalue=None, name=None):
        self.name = name
        if jvalue:
            self.value = jvalue
            self.bigdl_type = "float"
        else:
            if node:
                super(Variable, self).__init__(jvalue, "float", node, name)
            else:
                super(Variable, self).__init__(jvalue, "float", toMultiShape(input_shape), name)

    def set_name(self, name):
        self.node.element().set_name(name)

    @classmethod
    def from_node(cls, node):
        return cls(input_shape=None, node=node)

    @property
    def node(self):
        return Node.of(self.value.node())


class Lambda(kbase.ZooKerasCreator):
    """Used for evaluating an arbitrary expressions on an input.

       # Examples

       ```python
           # add a x -> x + 2 layer
           model.add(Lambda(lambda x: x + 2))
       ```
       # Arguments
           function: The function to be evaluated.
               Takes input tensor as first argument.

       # Input shape
           Arbitrary. Use the keyword argument input_shape
           (tuple of integers, does not include the samples axis)
           when using this layer as the first layer in a model.
       """

    def __init__(self, function, input_shape=None, bigdl_type="float"):
        self.function = function
        self.input_shape = input_shape
        self.bigdl_type = bigdl_type

    def __call__(self, x=None):
        """
        Some other modules point to current module
        :param x: upstream module nodes. x is either a Node or list of Node.
        :return: node containing current module
        """
        x = to_list(x if x else [])
        layer = self
        if isinstance(self, Lambda):
            input_shapes = [var.get_output_shape() for var in x]
            layer = self.create(remove_batch(input_shapes))
        return Variable.from_jvalue(callBigDlFunc(self.bigdl_type,
                                                  "connectInputs",
                                                  layer,
                                                  to_list(x)))

    # input_shapes should not contain batch dim
    def create(self, input_shapes):
        input_shapes = toMultiShape(input_shapes)
        inputs = [Variable(list(output_shape)) for output_shape in input_shapes]
        return LambdaLayer(input_vars=inputs,
                           out_var=self.function(*inputs),
                           input_shape=input_shapes)


class LambdaLayer(kbase.ZooKerasLayer):
    def __init__(self, input_vars, out_var, input_shape=None, **kwargs):
        super(LambdaLayer, self).__init__(None,
                                          input_vars,
                                          out_var,
                                          list(input_shape) if input_shape else None,
                                          **kwargs)


class Parameter(kbase.ZooKerasLayer, VariableOperator):
    """
    A trainable Variable. The default init_method is RandomUniform(-0.05, 0.05).
    You can also specify the init_weight by passing a ndarray.
    :param shape: Shape of this Parameter
    :param init_method: A method used to initialize the Parameter.
                        The default value is RandomUniform(-0.05, 0.05)
    :param init_weight: A ndarray as the init value.
    :param trainable It's true by default, meaning the value would be updated by gradient.
    """
    def __init__(self, shape, init_method=None,
                 init_weight=None, trainable=True, **kwargs):
        if not init_method:
            from bigdl.nn.initialization_method import RandomUniform
            init_method = RandomUniform(-0.05, 0.05)
        super(Parameter, self).__init__(None,
                                        list(shape),
                                        init_method,
                                        kbase.JTensor.from_ndarray(init_weight),
                                        trainable,
                                        ** kwargs)

    @property
    def shape(self):
        return self.get_weight().shape

    def get_weight(self):
        """
        :return: the ndarray for the current weight
        """
        jtensor = callBigDlFunc(self.bigdl_type,
                                "getParameterWeight",
                                self)
        return jtensor.to_ndarray()

    def set_weight(self, value):
        """
        :param value: value is a ndarray
        :return:
        """
        callBigDlFunc(self.bigdl_type,
                      "setParameterWeight",
                      self,
                      kbase.JTensor.from_ndarray(value))


class Constant(kbase.ZooKerasCreator, VariableOperator):
    """
    A constant Variable without weights.
    :param data: value of the Variable.
    :param name: Optional. Name of the Variable
    """
    def __init__(self, data, name=None, bigdl_type="float"):
        self.data = data
        super(Constant, self).__init__(None, bigdl_type, JTensor.from_ndarray(data), name)


class CustomLoss(LossFunction):
    def __init__(self, loss_func, y_pred_shape, y_true_shape=None):
        """
        :param loss_func: a function which accept y_true and y_pred
        :param y_pred_shape: The pred shape without batch dim.
        :param y_true_shape: The target shape without batch dim.
               It should be the same as y_pred_shape by default.
        i.e input_shape=[3], then the feeding data would be [None, 3]
        """

        y_real = Variable(input_shape=y_true_shape if y_true_shape else y_pred_shape)
        y_pred = Variable(input_shape=y_pred_shape)
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
