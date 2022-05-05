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

from bigdl.dllib.nn.layer import SpatialAveragePooling, SpatialBatchNormalization
from bigdl.dllib.nn.layer import SpatialConvolution, SpatialMaxPooling, JoinTable
from bigdl.dllib.nn.layer import ReLU, SoftMax, CAddTable, Unsqueeze
from bigdl.dllib.nn.onnx.layer import Constant, Gather, Gemm, Shape, Reshape
from .converter_utils import *


def average_pool(inputs, prev_modules, attrs, outputs):
    # extract attributes
    auto_pad = attrs.get('auto_pad', 'NOTSET')
    ceil_mode = True if attrs.get('ceil_mode', 0) == 1 else False
    count_include_pad = True if attrs.get('count_include_pad', 0) == 1 else False
    kernel_width, kernel_height = map(int, attrs.get('kernel_shape', (1, 1))[:2])
    stride_width, stride_height = map(int, attrs.get('strides', (1, 1))[:2])
    padding_width, padding_height = map(int, attrs.get('pads', (0, 0))[:2])
    # extract inputs
    _, data_tensor_shape = inputs[0]
    # calc output tensor shape
    input_height, input_width = data_tensor_shape[-2:]
    output_height = calc_output_shape(input_height, kernel_height,
                                      padding=padding_height, stride=stride_height,
                                      ceil_mode=ceil_mode)
    output_width = calc_output_shape(input_width, kernel_width,
                                     padding=padding_width, stride=stride_width,
                                     ceil_mode=ceil_mode)
    out_tensor_shape = list(data_tensor_shape)
    out_tensor_shape[-2] = output_height
    out_tensor_shape[-1] = output_width
    out_tensor_shape = tuple(out_tensor_shape)
    # create module node
    module = SpatialAveragePooling(kw=kernel_width, kh=kernel_height,
                                   dw=stride_width, dh=stride_height,
                                   pad_w=padding_width, pad_h=padding_height,
                                   ceil_mode=ceil_mode, count_include_pad=count_include_pad
                                   )(prev_modules)
    return module, [out_tensor_shape]


def batch_norm(inputs, prev_modules, attrs, outputs):
    # extract attributes
    epsilon = float(attrs.get('epsilon', 1e-05))
    momentum = float(attrs.get('momentum', 0.9))
    # extract inputs
    _, data_tensor_shape = inputs[0]
    scale_tensor_val, _ = inputs[1]
    bias_tensor_val, _ = inputs[2]
    mean_tensor_val, _ = inputs[3]
    var_tensor_val, _ = inputs[4]
    # calc output tensor shape
    out_tensor_shape = data_tensor_shape
    # create module node
    n_output = int(data_tensor_shape[1])

    temp_module = SpatialBatchNormalization(n_output=n_output, eps=epsilon,
                                            momentum=momentum, init_weight=scale_tensor_val,
                                            init_bias=bias_tensor_val)
    if mean_tensor_val is not None:
        temp_module.set_running_mean(mean_tensor_val)
    if var_tensor_val is not None:
        temp_module.set_running_std(var_tensor_val)
    module = temp_module(prev_modules[0])
    return module, [out_tensor_shape]


def concat(inputs, prev_modules, attrs, outputs):
    # extract attributes
    axis = int(attrs.get('axis'))
    # extract inputs
    _, data_tensor_shape = inputs[0]
    # calc output tensor shape
    dim_rank = 0
    for i in range(len(inputs)):
        _, curr_input_shape = inputs[i]
        for j in range(len(data_tensor_shape)):
            if axis != j:
                if curr_input_shape[i] != data_tensor_shape[i]:
                    invalidInputError(False, "Input shape mismatch. Expect receive input shape " +
                                      data_tensor_shape[i] + " but got " + curr_input_shape[i])
            else:
                dim_rank += curr_input_shape[axis]
    out_tensor_shape = list(data_tensor_shape)
    out_tensor_shape[axis] = dim_rank
    out_tensor_shape = tuple(out_tensor_shape)
    # create module node
    module = JoinTable(dimension=axis + 1, n_input_dims=len(data_tensor_shape))(prev_modules)
    return module, [out_tensor_shape]


def constant(inputs, prev_modules, attrs, outputs):
    # extract attributes
    value = parse_tensor_data(attrs.get('value'))

    # calc output tensor shape
    out_tensor_shape = value.shape
    # create module node
    module = Constant(value)(prev_modules[0])
    return module, [out_tensor_shape]


def conv(inputs, prev_modules, attrs, outputs):
    # extract attributes
    auto_pad = attrs.get('auto_pad', 'NOTSET')
    padW, padH = map(int, attrs.get('pads', (0, 0))[:2])
    kernelW, kernelH = map(int, attrs.get('kernel_shape', (0, 0))[:2])
    strideW, strideH = map(int, attrs.get('strides', (1, 1))[:2])
    dilationW, dilationH = map(int, attrs.get('dilations', (1, 1))[:2])
    group = int(attrs.get('group', 1))
    withBias = len(inputs) == 3 and inputs[2] is not None
    # extract inputs
    data_tensor_val, data_tensor_shape = inputs[0]
    weight_tensor_val, weight_tensor_shape = inputs[1]
    bias_tensor_val = None
    if withBias:
        bias_tensor_val, _ = inputs[2]
    # calc output tensor shape
    input_batch_size, n_input_plane = map(int, data_tensor_shape[:2])
    n_output_plane = weight_tensor_shape[0]
    input_height, input_width = data_tensor_shape[-2:]
    output_height = calc_output_shape(input_height, kernelH, padding=padH, stride=strideH)
    output_width = calc_output_shape(input_width, kernelW, padding=padW, stride=strideW)
    out_tensor_shape = (input_batch_size, n_output_plane, output_height, output_width)
    # create module node
    module = SpatialConvolution(
        n_input_plane=n_input_plane, n_output_plane=n_output_plane,
        kernel_w=kernelW, kernel_h=kernelH, stride_w=strideW, stride_h=strideH,
        pad_w=padW, pad_h=padH, n_group=group, init_weight=weight_tensor_val,
        init_bias=bias_tensor_val, with_bias=withBias
    )(prev_modules[0])
    return module, [out_tensor_shape]


def gather(inputs, prev_modules, attrs, outputs):
    # extract attributes
    axis = int(attrs.get('axis', 0))
    if axis != 0:
        invalidInputError(False, "Gather layer axis value")
    # extract inputs
    data_tensor_val, data_tensor_shape = inputs[0]
    indices_val, indices = inputs[1]
    # calc output tensor shape
    out_tensor_shape = tuple(data_tensor_shape[:axis] + indices + data_tensor_shape[axis + 1:])
    # create module node
    module = Gather()(prev_modules)
    return module, [out_tensor_shape]


def gemm(inputs, prev_modules, attrs, outputs):
    # extract attributes
    alpha = float(attrs.get("alpha", 1.0))
    beta = float(attrs.get("beta", 1.0))
    trans_a = int(attrs.get("transA", 0))
    trans_b = int(attrs.get("transB", 0))
    # extract inputs
    _, tensor_a_shape = inputs[0]
    tensor_b_val, tensor_b_shape = inputs[1]
    tensor_c_val, tensor_c_shape = inputs[2]
    # create module node
    module = Gemm(alpha=alpha, beta=beta, trans_a=trans_a, trans_b=trans_b,
                  matrix_b=tensor_b_val, matrix_c=tensor_c_val)(prev_modules)
    return module, [tensor_c_shape]


def max_pool(inputs, prev_modules, attrs, outputs):
    # extract attributes
    auto_pad = attrs.get("auto_pad", 'NOTSET')
    kernelW, kernelH = map(int, attrs.get("kernel_shape")[:2])
    strideW, strideH = map(int, attrs.get("strides", (1, 1))[:2])
    dilationW, dilationH = map(int, attrs.get('dilations', (1, 1))[:2])
    padW, padH = map(int, attrs.get("pads", (0, 0))[:2])
    ceil_mode = True if (attrs.get("ceil_mode", 0) != 0) else False
    storage_order = int(attrs.get("storage_order", 0))
    # extract inputs
    _, data_tensor_shape = inputs[0]
    input_width, input_height = data_tensor_shape[-2:]
    # calc output tensor shape
    output_width = calc_output_shape(input_width, kernelW,
                                     padding=padW, stride=strideW, dilation=dilationW,
                                     ceil_mode=ceil_mode)
    output_height = calc_output_shape(input_height, kernelH,
                                      padding=padH, stride=strideH, dilation=dilationH,
                                      ceil_mode=ceil_mode)
    out_tensor_shape_list = list(data_tensor_shape)
    out_tensor_shape_list[2] = output_height
    out_tensor_shape_list[3] = output_width
    out_tensor_shape = tuple(out_tensor_shape_list)
    # create module node
    module = SpatialMaxPooling(kw=kernelW, kh=kernelH, dw=strideW, dh=strideH,
                               pad_w=padW, pad_h=padH, to_ceil=ceil_mode)(prev_modules[0])
    return module, [out_tensor_shape]


def relu(inputs, prev_modules, attrs, outputs):
    # extract inputs
    _, data_tensor_shape = inputs[0]
    # calc output tensor shape
    output_shape = data_tensor_shape
    # create module node
    module = ReLU()(prev_modules[0])
    return module, [output_shape]


def reshape(inputs, prev_modules, attrs, outputs):
    # extract inputs
    _, data_tensor_shape = inputs[0]
    shape_tensor_val, _ = inputs[1]
    shape_arry = None
    if shape_tensor_val is not None:
        shape_arry = np.squeeze(shape_tensor_val).astype(int).tolist()
    # create module node
    module = Reshape(shape_arry)(prev_modules)
    return module, [shape_tensor_val]


def shape(inputs, prev_modules, attrs, outputs):
    # extract inputs
    _, data_tensor_shape = inputs[0]
    # create module node
    module = Shape()(prev_modules[0])
    return module, [(len(data_tensor_shape),)]


def softmax(inputs, prev_modules, attrs, outputs):
    _, data_tensor_shape = inputs[0]
    out_tensor_shape = data_tensor_shape
    axis = int(attrs.get('axis', 1))
    module = SoftMax()(prev_modules[0])
    return module, [out_tensor_shape]


def _sum(inputs, prev_modules, attrs, outputs):
    _, data_tensor_shape = inputs[0]
    out_tensor_shape = data_tensor_shape
    module = CAddTable()(prev_modules)
    return module, [data_tensor_shape]


def unsqueeze(inputs, prev_modules, attrs, outputs):
    axes = list(map(int, attrs.get('axes')))
    data_tensor_val, data_tensor_shape = inputs[0]
    out_tensor_shape = list(data_tensor_shape)
    for idx in axes:
        out_tensor_shape.insert(idx, 1)
    out_tensor_shape = tuple(out_tensor_shape)
    module = Unsqueeze(axes[0], len(data_tensor_shape))(prev_modules)
    return module, [out_tensor_shape]
