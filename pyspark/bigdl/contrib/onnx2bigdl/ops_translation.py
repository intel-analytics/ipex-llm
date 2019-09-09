
# coding: utf-8
""" Module for translating ONNX operators into Mxnet operatoes"""
# pylint: disable=unused-argument,protected-access
import numpy as np
from bigdl.nn.onnx.layer import *
from translation_utils import *

from onnx.numpy_helper import to_array


# Method definitions for the callable objects mapped in the import_helper module
def avg_pooling(inputs, prev_modules, attrs, outputs):
	auto_pad = attrs.get('auto_pad', 'NOTSET')
	ceil_mode = True if attrs.get('ceil_mode', 0) == 1 else False
	count_include_pad = True if attrs.get('count_include_pad', 0) == 1 else False
	kernel_shape = attrs.get('kernel_shape', (0, 0))
	pads = attrs.get('pads', [0, 0, 0, 0])
	strides = attrs.get('strides', [0, 0])

	kernel_width, kernel_height = kernel_shape[:2]
	stride_width, stride_height = strides[:2]
	padding_width, padding_height = pads[:2]

	_, data_tensor_shape = inputs[0]
	input_height, input_width = data_tensor_shape[-2:]

	output_height = calc_output_shape(input_height, kernel_height, 
		padding = padding_height, stride = stride_height, ceil_mode = ceil_mode)
	output_width = calc_output_shape(input_width, kernel_width, 
		padding = padding_width, stride = stride_width, ceil_mode = ceil_mode)

	out_tensor_shape = list(data_tensor_shape)
	out_tensor_shape[-2] = output_height
	out_tensor_shape[-1] = output_width
	out_tensor_shape = tuple(out_tensor_shape)

	module = SpatialAveragePooling(kernel_width, kernel_height,
		dw = stride_width, dh = stride_height, pad_w = padding_width, pad_h = padding_height,
		ceil_mode = ceil_mode, count_include_pad = count_include_pad)(prev_modules)

	return module, [out_tensor_shape]


def batch_norm(inputs, prev_modules, attrs, outputs):

	epsilon = attrs.get('epsilon', 1e-05)
	momentum = attrs.get('momentum', 0.9)

	_, data_tensor_shape = inputs[0]
	scale_tensor_val, _ = inputs[1]
	bias_tensor_val, _ = inputs[2]

	n_output = data_tensor_shape[1]
	out_tensor_shape = data_tensor_shape

	module = SpatialBatchNormalization(n_output, eps = epsilon)(prev_modules)

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
					raise ValueError()
			else:
				dim_rank += curr_input_shape[axis]

	out_tensor_shape = list(data_tensor_shape)
	out_tensor_shape[axis] = dim_rank
	out_tensor_shape = tuple(out_tensor_shape)

	module = Identity()(prev_modules)

	print('Concat', 'axis', axis, 'dim_rank', dim_rank, data_tensor_shape, out_tensor_shape)

	return module, [out_tensor_shape]


def constant(inputs, prev_modules, attrs, outputs):

	value = parse_tensor_data(attrs.get('value'))
	module = Identity()(prev_modules)

	print(value)
	print(prev_modules)

	out_tensor_shape = value.shape
	print(out_tensor_shape)
	return module, [out_tensor_shape]

def conv(inputs, prev_modules, attrs, outputs):

	auto_pad = attrs.get('auto_pad', 'NOTSET')
	dilations = attrs.get('dilations', (1, 1))
	group = attrs.get('group', 1)
	pads = attrs.get('pads', (0, 0, 0, 0))[:2]
	kernel_shape = attrs.get('kernel_shape', (0, 0))
	strides = attrs.get('strides', (1, 1))

	data_tensor_val, data_tensor_shape = inputs[0]
	weight_tensor_val, weight_tensor_shape = inputs[1]


	bias_tensor_val = None

	if len(inputs) == 3:
		bias_tensor_val, _ = inputs[2]

	n_input_plane = int(data_tensor_shape[1])
	n_output_plane = int(weight_tensor_shape[0])
	input_height, input_width = data_tensor_shape[-2:]

	kernelW, kernelH = kernel_shape
	strideW, strideH = strides
	padH, padW = pads[-2:]
	nGroup = group
	withBias = len(inputs) == 3 

	module = Conv(
		n_input_plane = n_input_plane,
		n_output_plane = n_output_plane,
		kernel_shape = kernel_shape,
		weight = weight_tensor_val,
		bias = bias_tensor_val,
		auto_pad=auto_pad,
		dilations=dilations,
		group = group,
		pads = pads[-2:],
		strides = strides
		)(prev_modules)
	
	output_height = calc_output_shape(input_height, kernelH)
	output_width  = calc_output_shape(input_width, kernelW)

	out_tensor_shape = (n_input_plane, n_output_plane, output_height, output_width)

	return module, [out_tensor_shape]


def gather(inputs, prev_modules, attrs, outputs):

	axis = attrs.get('axis', 0)

	if axis != 0:
		raise ValueError("Gather layer axis value")

	_, data_tensor_shape = inputs[0]
	_, indices = inputs[1]

	out_tensor_shape = tuple(data_tensor_shape[:axis] + indices + data_tensor_shape[axis + 1:])

	module = Identity()(prev_modules)
	# Todo
	return module, [out_tensor_shape]

def linalg_gemm(inputs, prev_modules, attrs, outputs):

	_, tensor_a_shape = inputs[0]
	_, tensor_b_shape = inputs[1]
	_, bias_tensor_shape = inputs[2]

	alpha = attrs.get("alpha", 1.0)
	beta = attrs.get("beta", 1.0)
	transA = attrs.get("transA", 0)
	transB = attrs.get("transB", 0)

	print(tensor_a_shape)
	print(tensor_b_shape)
	print(bias_tensor_shape)

	module = SpatialConvolution(1, 6, 2, 2)(prev_modules)

	# Todo:
	return module, [bias_tensor_shape]


def max_pooling(inputs, prev_modules, attrs, outputs):

	kernelW, kernelH = attrs.get("kernel_shape")
	strideW, strideH = attrs.get("strides", (1, 1))
	padW, padH = attrs.get("pads", (0, 0, 0, 0))[:2]
	dilationW, dilationH = (0, 0)
	ceil_mode = attrs.get("ceil_mode", False)

	_, data_tensor_shape = inputs[0]
	input_width, input_height = data_tensor_shape[-2:]

	output_width = calc_output_shape(input_width, kernelW,
	 	padding = padW, stride = strideW, dilation = dilationW, ceil_mode = ceil_mode)

	output_height = calc_output_shape(input_height, kernelH,
		padding = padH, stride = strideH, dilation = dilationH, ceil_mode = ceil_mode)

	module = SpatialMaxPooling(kernelW, kernelH,
		dw = strideW, dh = strideH, pad_w = padW, pad_h = padH,
		to_ceil = ceil_mode)(prev_modules)

	out_tensor_shape_list = list(data_tensor_shape)
	out_tensor_shape_list[2] = output_height
	out_tensor_shape_list[3] = output_width

	out_tensor_shape = tuple(out_tensor_shape_list)

	return module, [out_tensor_shape]


def relu(inputs, prev_modules, attrs, outputs):

	_, data_tensor_shape = inputs[0]
	output_shape = data_tensor_shape

	module = ReLU()(prev_modules)

	return module, [output_shape]


def reshape(inputs, prev_modules, attrs, outputs):

	_, data_tensor_shape = inputs[0]
	_, out_tensor_shape = inputs[1]

	source_elem_size = np.prod(data_tensor_shape)
	target_elem_size = np.prod(out_tensor_shape)

	print('reshape', data_tensor_shape, out_tensor_shape, prev_modules)

	if source_elem_size != target_elem_size:
		raise ValueError('Invalid shape')

	module = Reshape(list(out_tensor_shape))(prev_modules)

	# Todo: 
	return module, [data_tensor_shape]


def shape(inputs, prev_modules, attrs, outputs):
	_, data_tensor_shape = inputs[0]

	module = Identity()(prev_modules)

	return module, [data_tensor_shape]


def softmax(inputs, prev_modules, attrs, outputs):

	_, data_tensor_shape = inputs[0]
	out_tensor_shape = data_tensor_shape
	axis = attrs.get('axis', 1)

	module = SoftMax()(prev_modules)

	return module, [out_tensor_shape]


def _sum(inputs, prev_modules, attrs, outputs):

	_, data_tensor_shape = inputs[0]
	out_tensor_shape = data_tensor_shape

	module = CAddTable()(prev_modules)

	# Todo:
	return module, [data_tensor_shape]


def unsqueeze(inputs, prev_modules, attrs, outputs):
	axes = attrs.get('axes')

	data_tensor_val, data_tensor_shape = inputs[0]

	print(axes)
	print(data_tensor_shape)
	out_tensor_shape = list(data_tensor_shape)
	for idx in axes:
		out_tensor_shape.insert(idx, 1)
	out_tensor_shape = tuple(out_tensor_shape)

	print(data_tensor_shape, out_tensor_shape)

	module = Identity()(prev_modules)

	return module, [out_tensor_shape]
