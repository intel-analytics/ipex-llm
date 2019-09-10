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

from bigdl.nn.onnx.layer import *
from .translation_utils import *


# Method definitions for the callable objects mapped in the import_helper module
def average_pool(inputs, prev_modules, attrs, outputs):
	auto_pad = attrs.get('auto_pad', 'NOTSET')
	ceil_mode = attrs.get('ceil_mode', 0)
	ceil_mode_flag = True if attrs.get('ceil_mode', 0) == 1 else False
	count_include_pad = attrs.get('count_include_pad', 0)
	kernel_shape = attrs.get('kernel_shape', (0, 0))
	strides = attrs.get('strides', (0, 0))
	pads = attrs.get('pads', (0, 0, 0, 0))

	kernel_width, kernel_height = kernel_shape[:2]
	stride_width, stride_height = strides[:2]
	padding_width, padding_height = pads[:2]

	_, data_tensor_shape = inputs[0]
	input_height, input_width = data_tensor_shape[-2:]

	output_height = calc_output_shape(input_height, kernel_height, 
		padding = padding_height, stride = stride_height, ceil_mode = ceil_mode_flag)
	output_width = calc_output_shape(input_width, kernel_width, 
		padding = padding_width, stride = stride_width, ceil_mode = ceil_mode_flag)

	out_tensor_shape = list(data_tensor_shape)
	out_tensor_shape[-2] = output_height
	out_tensor_shape[-1] = output_width
	out_tensor_shape = tuple(out_tensor_shape)

	module = AveragePool(kernel_shape = kernel_shape, auto_pad = auto_pad,
		ceil_mode = ceil_mode, count_include_pad = count_include_pad,
		pads = pads, strides = strides)(prev_modules)

	return module, [out_tensor_shape]


def batch_norm(inputs, prev_modules, attrs, outputs):
	epsilon = float(attrs.get('epsilon', 1e-05))
	momentum = float(attrs.get('momentum', 0.9))
	_, data_tensor_shape = inputs[0]
	scale_tensor_val, _ = inputs[1]
	bias_tensor_val, _ = inputs[2]
	n_output = data_tensor_shape[1]
	out_tensor_shape = data_tensor_shape
	module = BatchNormalization(n_output, epsilon, momentum)(prev_modules[0])
	return module, [out_tensor_shape]


def concat(inputs, prev_modules, attrs, outputs):
	axis = attrs.get('axis')
	_, data_tensor_shape = inputs[0]
	dim_rank = 0
	for i in range(len(inputs)):
		_, curr_input_shape = inputs[i]
		for j in range(len(data_tensor_shape)):
			if axis != j:
				if curr_input_shape[i] != data_tensor_shape[i]:
					raise ValueError("Input shape mismatch. Expect receive input shape " +
						data_tensor_shape[i] + " but got " + curr_input_shape[i])
			else:
				dim_rank += curr_input_shape[axis]
	out_tensor_shape = list(data_tensor_shape)
	out_tensor_shape[axis] = dim_rank
	out_tensor_shape = tuple(out_tensor_shape)
	module = Concat(len(data_tensor_shape), axis)(prev_modules)

	return module, [out_tensor_shape]


def constant(inputs, prev_modules, attrs, outputs):
	value = parse_tensor_data(attrs.get('value'))
	module = Constant(value)(prev_modules[0])
	out_tensor_shape = value.shape
	return module, [out_tensor_shape]


def conv(inputs, prev_modules, attrs, outputs):
	auto_pad = attrs.get('auto_pad', 'NOTSET')
	dilations = attrs.get('dilations', (1, 1))
	group = attrs.get('group', 1)
	pads = attrs.get('pads', (0, 0, 0, 0))
	kernel_shape = attrs.get('kernel_shape', (0, 0))
	strides = attrs.get('strides', (1, 1))

	data_tensor_val, data_tensor_shape = inputs[0]
	weight_tensor_val, weight_tensor_shape = inputs[1]
	bias_tensor_val = None

	if len(inputs) == 3:
		bias_tensor_val, _ = inputs[2]

	input_batch_size, n_input_plane = data_tensor_shape[:2]
	n_output_plane = weight_tensor_shape[0]
	input_height, input_width = data_tensor_shape[-2:]

	kernelW, kernelH = kernel_shape
	strideW, strideH = strides
	padH, padW = pads[-2:]
	group = group
	withBias = len(inputs) == 3

	module = Conv(
		n_input_plane = n_input_plane,
		n_output_plane = n_output_plane,
		kernel_shape = kernel_shape,
		weight = weight_tensor_val,
		bias = bias_tensor_val,
		auto_pad = auto_pad,
		dilations = dilations,
		group = group,
		pads = pads[-2:],
		strides = strides
		)(prev_modules[0])
	
	output_height = calc_output_shape(input_height, kernelH, padding = padH, stride=strideH)
	output_width = calc_output_shape(input_width, kernelW, padding = padW, stride=strideW)
	out_tensor_shape = (input_batch_size, n_output_plane, output_height, output_width)
	return module, [out_tensor_shape]


def gather(inputs, prev_modules, attrs, outputs):
	axis = attrs.get('axis', 0)
	if axis != 0:
		raise ValueError("Gather layer axis value")
	data_tensor_val, data_tensor_shape = inputs[0]
	indices_val, indices = inputs[1]
	out_tensor_shape = tuple(data_tensor_shape[:axis] + indices + data_tensor_shape[axis + 1:])
	module = Gather(axis = axis)(prev_modules)
	return module, [out_tensor_shape]


def gemm(inputs, prev_modules, attrs, outputs):
	_, tensor_a_shape = inputs[0]
	tensor_b_val, tensor_b_shape = inputs[1]
	bias_tensor_val, bias_tensor_shape = inputs[2]
	alpha = float(attrs.get("alpha", 1.0))
	beta = float(attrs.get("beta", 1.0))
	trans_a = attrs.get("transA", 0)
	trans_b = attrs.get("transB", 0)
	module = Gemm(alpha=alpha, beta=beta, trans_a=trans_a, trans_b=trans_b)(prev_modules)
	return module, [bias_tensor_shape]


def max_pool(inputs, prev_modules, attrs, outputs):
	auto_pad = attrs.get("auto_pad", 'NOTSET')
	kernelW, kernelH = attrs.get("kernel_shape")
	kernel_shape = attrs.get("kernel_shape")
	strides = attrs.get("strides", (1, 1))
	strideW, strideH = attrs.get("strides", (1, 1))
	dilations = attrs.get('dilations', (1, 1))
	pads = attrs.get("pads", (0, 0, 0, 0))[:2]
	padW, padH = attrs.get("pads", (0, 0, 0, 0))[:2]
	dilationW, dilationH = (1, 1)
	ceil_mode = attrs.get("ceil_mode", 0)
	ceil_mode_flag = True if (attrs.get("ceil_mode", 0) != 0) else False
	storage_order = attrs.get("storage_order", 0)

	_, data_tensor_shape = inputs[0]
	input_width, input_height = data_tensor_shape[-2:]

	output_width = calc_output_shape(input_width, kernelW,
		padding = padW, stride = strideW, dilation = dilationW, ceil_mode = ceil_mode_flag)

	output_height = calc_output_shape(input_height, kernelH,
		padding = padH, stride = strideH, dilation = dilationH, ceil_mode = ceil_mode_flag)

	module = MaxPool(kernel_shape = kernel_shape, auto_pad = auto_pad,
					 ceil_mode = ceil_mode, dilations = dilations,
					 pads = pads, storage_order = storage_order,
					 strides = strides)(prev_modules[0])

	out_tensor_shape_list = list(data_tensor_shape)
	out_tensor_shape_list[2] = output_height
	out_tensor_shape_list[3] = output_width
	out_tensor_shape = tuple(out_tensor_shape_list)
	return module, [out_tensor_shape]


def relu(inputs, prev_modules, attrs, outputs):
	_, data_tensor_shape = inputs[0]
	output_shape = data_tensor_shape
	module = Relu()(prev_modules[0])
	return module, [output_shape]


def reshape(inputs, prev_modules, attrs, outputs):
	data_tensor_val, data_tensor_shape = inputs[0]
	out_tensor_val, out_tensor_shape = inputs[1]
	module = Reshape()(prev_modules)
	return module, [data_tensor_shape]


def shape(inputs, prev_modules, attrs, outputs):
	_, data_tensor_shape = inputs[0]
	module = Shape()(prev_modules[0])
	return module, [(len(data_tensor_shape),)]


def softmax(inputs, prev_modules, attrs, outputs):
	_, data_tensor_shape = inputs[0]
	out_tensor_shape = data_tensor_shape
	axis = attrs.get('axis', 1)
	module = Softmax(axis = axis)(prev_modules[0])
	return module, [out_tensor_shape]


def _sum(inputs, prev_modules, attrs, outputs):
	_, data_tensor_shape = inputs[0]
	out_tensor_shape = data_tensor_shape
	module = OnnxSum()(prev_modules)
	return module, [data_tensor_shape]


def unsqueeze(inputs, prev_modules, attrs, outputs):
	axes = attrs.get('axes')
	data_tensor_val, data_tensor_shape = inputs[0]
	out_tensor_shape = list(data_tensor_shape)
	for idx in axes:
		out_tensor_shape.insert(idx, 1)
	out_tensor_shape = tuple(out_tensor_shape)
	module = Unsqueeze(axes, len(data_tensor_shape))(prev_modules)
	return module, [out_tensor_shape]
