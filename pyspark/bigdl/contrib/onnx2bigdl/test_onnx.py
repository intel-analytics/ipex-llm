import onnx
import pytest
import numpy as np

from onnx import AttributeProto, TensorProto, GraphProto


class TestOnnxOps:

	def __init__(self):
		return

	def hello(self):
		print('hello')

	def test_avg_pooling(self):
		# Create one input (ValueInfoProto)
		X = onnx.helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])

		# Create one output (ValueInfoProto)
		Y = onnx.helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
		x = np.random.randn(1, 3, 32, 32).astype(np.float32)
		x_shape = np.shape(x)
		kernel_shape = (2, 2)
		strides = (1, 1)

		node = onnx.helper.make_node(
			'AveragePool',
			inputs=['x'],
			outputs=['y'],
			kernel_shape=[2, 2],
		)

		# Create a node (NodeProto)
		node_def = onnx.helper.make_node(
			'Pad', # node name
			['X'], # inputs
			['Y'], # outputs
			mode='constant', # attributes
			value=1.5,
			pads=[0, 1, 0, 1],
		)

		graph_def = onnx.helper.make_graph(
			[node_def],
			'test-model',
			[X],
			[Y],
		)

		model_def = onnx.helper.make_model(graph_def, producer_name='onnx-example')

		print('The model is:\n{}'.format(model_def))
		onnx.checker.check_model(model_def)
		print('The model is checked!')

		print(dir(model_def))
		print(model_def.SerializeToString())

		with open('test.onnx', 'wb') as f:
			f.write(model_def.SerializeToString())

		return



def main():
	tester = TestOnnxOps()
	tester.hello()
	tester.test_avg_pooling()
	assert True



if __name__ == '__main__':
	main()