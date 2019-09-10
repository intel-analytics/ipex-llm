
import math
import numpy as np

def calc_output_shape(input, kernel, padding = 0, stride = 1, dilation = 1, ceil_mode = False):
	def dilated_kernel_size(kernel, dilation):
		return kernel + (kernel - 1) * (dilation - 1)
	rounding = math.ceil if ceil_mode else math.floor
	out = (input + 2 * padding - dilated_kernel_size(kernel, dilation)) / stride + 1
	out = int(rounding(out))
	print(input, kernel, padding, stride, dilation, ceil_mode, out)
	return out

def parse_tensor_data(tensor_proto):
	try:
		from onnx.numpy_helper import to_array
	except ImportError:
		raise ImportError("Onnx and protobuf need to be installed.")
	if len(tuple(tensor_proto.dims)) > 0:
		np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
	else:
		# If it is a scalar tensor
		np_array = np.array([to_array(tensor_proto)])
	return np_array

if __name__ == '__main__':
	print()
