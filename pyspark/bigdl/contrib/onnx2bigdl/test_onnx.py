import sys, os
import onnx
import pytest
import numpy as np

from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from .import_model import OnnxModel

class TestExample(object):

    def test_example(self):
        assert('Hello, World!' == 'Hello, World!')


class TestAveragePool(object):

    def test_average_pool(self):
        # node = onnx.helper.make_node(
        #     'AveragePool',
        #     inputs=['x'],
        #     outputs=['y'],
        #     kernel_shape=[2, 2],
        # )
        # x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        # x_shape = np.shape(x)
        # kernel_shape = (2, 2)
        # strides = (1, 1)
        # padded = x

        # Create one input (ValueInfoProto)
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])

        # Create one output (ValueInfoProto)
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])

        # Create a node (NodeProto)
        node_def = helper.make_node(
            'AveragePool', # node name
            inputs=['X'], # inputs
            outputs=['Y'], # outputs
            kernel_shape=[2, 2]
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test-averagePool',
            [X],
            [Y],
        )

        # Create the model (ModelProto)
        model_def = helper.make_model(graph_def, producer_name='test-averagePool')
        assert('Hello, World!' == 'Hello, World!')
        onnx.checker.check_model(model_def)


def main():
    avg_pool = TestAveragePool()
    avg_pool.test_average_pool()



if __name__ == "__main__":
    main()