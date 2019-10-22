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

import onnx
import pytest
import numpy as np

from bigdl.contrib.onnx.onnx_loader import load_model_proto
from bigdl.nn.layer import CAddTable, JoinTable, ReLU
from bigdl.nn.layer import SoftMax, SpatialAveragePooling, SpatialBatchNormalization
from bigdl.nn.layer import SpatialConvolution, SpatialMaxPooling
from bigdl.nn.layer import Unsqueeze
from bigdl.nn.onnx.layer import Gemm, Reshape, Shape


class TestAveragePool(object):

    def test_average_pool(self):
        ceil_mode = 0
        kernel_width, kernel_height = 3, 3
        pad_width, pad_height = 0, 0
        stride_width, stride_height = 1, 1
        input_shape = [1, 3, 224, 224]
        output_shape = [1, 3, 222, 222]
        input_x = np.random.random(input_shape)

        # Create one input (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        # Create one output (ValueInfoProto)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        # Create a node (NodeProto)
        avgpool_node = onnx.helper.make_node(
            op_type='AveragePool',  # node name
            inputs=['X'],  # inputs
            outputs=['Y'],  # outputs
            auto_pad='NOTSET',
            ceil_mode=ceil_mode,
            kernel_shape=(kernel_width, kernel_height),
            pads=(pad_width, pad_height),
            strides=(stride_width, stride_height)
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[avgpool_node],
            name='test-averagePool',
            inputs=[X],
            outputs=[Y],
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        loaded_model = load_model_proto(onnx_model)
        bigdl_model = SpatialAveragePooling(
            kw=kernel_width,
            kh=kernel_height,
            dw=stride_width,
            dh=stride_height,
            pad_w=pad_width,
            pad_h=pad_height,
            ceil_mode=False if ceil_mode == 0 else True
        )

        loaded_out = loaded_model.forward(input_x)
        expected_out = bigdl_model.forward(input_x)

        assert(np.array_equal(expected_out, loaded_out))


class TestBatchNormalization(object):

    def test_batch_normalization(self):
        input_shape = [1, 3, 224, 224]
        output_shape = [1, 3, 224, 224]
        # Create inputs (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        scale = onnx.helper.make_tensor_value_info('scale', onnx.TensorProto.FLOAT, input_shape[:2])
        bias = onnx.helper.make_tensor_value_info('bias', onnx.TensorProto.FLOAT, input_shape[:2])
        mean = onnx.helper.make_tensor_value_info('mean', onnx.TensorProto.FLOAT, input_shape[:2])
        var = onnx.helper.make_tensor_value_info('var', onnx.TensorProto.FLOAT, input_shape[:2])
        # Create one output (ValueInfoProto)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        scale_vals = np.random.random(input_shape[1]) * 10
        bias_vals = np.random.random(input_shape[1]) * 10
        mean_vals = np.random.random(input_shape[1]) * 10
        var_vals = np.random.random(input_shape[1]) * 10
        input_x = np.random.random(input_shape) * 10
        epsilon = float(1e-05)
        momentum = float(0.9)

        init_scale = onnx.helper.make_tensor(
            name='scale',
            data_type=onnx.TensorProto.FLOAT,
            dims=input_shape[:2],
            vals=scale_vals.tolist(),
        )

        init_bias = onnx.helper.make_tensor(
            name='bias',
            data_type=onnx.TensorProto.FLOAT,
            dims=input_shape[:2],
            vals=bias_vals.tolist(),
        )

        init_mean = onnx.helper.make_tensor(
            name='mean',
            data_type=onnx.TensorProto.FLOAT,
            dims=input_shape[:2],
            vals=mean_vals.tolist(),
        )

        init_var = onnx.helper.make_tensor(
            name='var',
            data_type=onnx.TensorProto.FLOAT,
            dims=input_shape[:2],
            vals=var_vals.tolist(),
        )

        # Create a node (NodeProto)
        batch_norm_node = onnx.helper.make_node(
            op_type='BatchNormalization',  # node name
            inputs=['X', 'scale', 'bias', 'mean', 'var'],  # inputs
            outputs=['Y'],  # outputs
            epsilon=epsilon,
            momentum=momentum
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[batch_norm_node],
            name='test-batch_norm',
            inputs=[X],
            outputs=[Y],
            initializer=[init_scale, init_bias, init_mean, init_var]
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        loaded_model = load_model_proto(onnx_model)
        bigdl_model = SpatialBatchNormalization(
            n_output=input_shape[1],
            eps=epsilon,
            momentum=momentum,
            init_weight=scale_vals,
            init_bias=bias_vals,
            init_grad_weight=None,
            init_grad_bias=None,
        )
        bigdl_model.set_running_mean(mean_vals)
        bigdl_model.set_running_std(var_vals)

        loaded_out = loaded_model.forward(input_x)
        expected_out = bigdl_model.forward(input_x)

        assert(np.array_equal(loaded_out, expected_out))


class TestConcat(object):

    def test_concat(self):
        axis = 0
        input_shape = [2, 3]
        output_shape = [4, 3]
        x1_val = np.random.random(input_shape)
        x2_val = np.random.random(input_shape)

        # Create input (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        X1 = onnx.helper.make_tensor_value_info('X1', onnx.TensorProto.FLOAT, input_shape)
        X2 = onnx.helper.make_tensor_value_info('X2', onnx.TensorProto.FLOAT, input_shape)

        # Create one output (ValueInfoProto)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        # Create a node (NodeProto)
        const_X1 = onnx.helper.make_node(
            op_type='Constant',
            inputs=[],
            outputs=['X1'],
            value=onnx.helper.make_tensor(
                name='X1',
                data_type=onnx.TensorProto.FLOAT,
                dims=input_shape,
                vals=x1_val.flatten().tolist(),
            )
        )

        const_X2 = onnx.helper.make_node(
            op_type='Constant',
            inputs=[],
            outputs=['X2'],
            value=onnx.helper.make_tensor(
                name='X2',
                data_type=onnx.TensorProto.FLOAT,
                dims=input_shape,
                vals=x2_val.flatten().tolist(),
            )
        )

        concat_node = onnx.helper.make_node(
            op_type='Concat',  # node name
            inputs=['X1', 'X2'],  # inputs
            outputs=['Y'],  # outputs
            axis=axis
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[const_X1, const_X2, concat_node],
            name='test-concat',
            inputs=[X],
            outputs=[Y],
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        loaded_model = load_model_proto(onnx_model)
        bigdl_model = JoinTable(dimension=axis + 1, n_input_dims=len(input_shape))

        loaded_out = loaded_model.forward([x1_val, x2_val])
        expected_out = bigdl_model.forward([x1_val, x2_val])

        assert(np.array_equal(loaded_out, expected_out))


class TestConstant(object):

    def test_constant(self):

        shape = [5, 5]
        values = np.float32(np.round(np.random.random(shape), 6))
        dummy_input = np.random.random([1])

        # Create one output (ValueInfoProto)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, values.shape)

        constant_node = onnx.helper.make_node(
            op_type='Constant',
            inputs=[],
            outputs=['Y'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().tolist(),
            ),
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[constant_node],
            name='test-constant',
            inputs=[],
            outputs=[Y],
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)
        loaded_model = load_model_proto(onnx_model)

        loaded_out = loaded_model.forward(dummy_input)
        expected_out = values

        assert(np.array_equal(loaded_out, expected_out))


class TestConv(object):

    def test_conv(self):
        kernel_width, kernel_height = (3, 3)
        stride_width, stride_height = (1, 1)
        pad_width, pad_height = (0, 0)
        input_shape = [1, 3, 224, 224]
        output_shape = [1, 8, 222, 222]
        weight_shape = [8, 3, 3, 3]
        input_x = np.random.random(input_shape)
        weight_values = np.random.random(weight_shape)

        # Create input (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        W = onnx.helper.make_tensor_value_info('W', onnx.TensorProto.FLOAT, weight_shape)
        # Create one output (ValueInfoProto)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        init_weight = onnx.helper.make_tensor(
            name='W',
            data_type=onnx.TensorProto.FLOAT,
            dims=weight_shape,
            vals=weight_values.flatten().astype(float),
        )

        conv_node = onnx.helper.make_node(
            op_type='Conv',
            inputs=['X', 'W'],
            outputs=['Y'],
            kernel_shape=(kernel_width, kernel_height),
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[conv_node],
            name='test-conv',
            inputs=[X],
            outputs=[Y],
            initializer=[init_weight]
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        loaded_model = load_model_proto(onnx_model)
        bigdl_model = SpatialConvolution(
            n_input_plane=3,
            n_output_plane=8,
            kernel_w=kernel_width,
            kernel_h=kernel_height,
            stride_w=stride_width,
            stride_h=stride_height,
            pad_w=pad_width,
            pad_h=pad_height,
            init_weight=weight_values,
            with_bias=False
        )

        loaded_out = loaded_model.forward(input_x)
        expected_out = bigdl_model.forward(input_x)

        assert(np.array_equal(loaded_out, expected_out))


class TestGather(object):

    def test_gather(self):
        axis = 0
        input_x = np.array([
            [1.0, 1.2],
            [2.3, 3.4],
            [4.5, 5.7],
        ], dtype=float)
        indices_val = np.array([[0, 1], [1, 2]], dtype=float)
        expected_out = np.array([[[1, 1.2], [2.3, 3.4]],
                                 [[2.3, 3.4], [4.5, 5.7]]], dtype=float)
        input_shape = input_x.shape
        indices_shape = indices_val.shape
        output_shape = [2, 2, 2]

        # Create one output (ValueInfoProto)
        data = onnx.helper.make_tensor_value_info('data', onnx.TensorProto.FLOAT, input_shape)
        indices = onnx.helper.make_tensor_value_info('indices',
                                                     onnx.TensorProto.FLOAT, indices_shape)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        init_indices = onnx.helper.make_tensor(
            name='indices',
            data_type=onnx.TensorProto.FLOAT,
            dims=indices_shape,
            vals=indices_val.flatten().tolist(),
        )

        gather_node = onnx.helper.make_node(
            op_type='Gather',
            inputs=['data', 'indices'],
            outputs=['Y'],
            axis=axis
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[gather_node],
            name='test-gather',
            inputs=[data, indices],
            outputs=[Y],
            initializer=[init_indices]
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        loaded_model = load_model_proto(onnx_model)
        loaded_out = loaded_model.forward([input_x, indices_val])

        assert(np.allclose(loaded_out, expected_out))


class TestGemm(object):

    def test_gemm(self):
        mata_shape = [2, 7]
        matb_shape = [7, 4]
        matc_shape = [2, 4]
        output_shape = [2, 4]
        alpha = np.round(np.random.rand(), 2)
        beta = np.round(np.random.rand(), 2)
        trans_a, trans_b = 0, 0
        input_x = np.random.random(mata_shape)
        b_val = np.random.random(matb_shape)
        c_val = np.random.random(matc_shape)

        # Create one output (ValueInfoProto)
        a = onnx.helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT, mata_shape)
        b = onnx.helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT, matb_shape)
        c = onnx.helper.make_tensor_value_info('c', onnx.TensorProto.FLOAT, matc_shape)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        init_b = onnx.helper.make_tensor(
            name='b',
            data_type=onnx.TensorProto.FLOAT,
            dims=matb_shape,
            vals=b_val.flatten().tolist(),
        )

        init_c = onnx.helper.make_tensor(
            name='c',
            data_type=onnx.TensorProto.FLOAT,
            dims=matc_shape,
            vals=c_val.flatten().tolist(),
        )

        gemm_node = onnx.helper.make_node(
            op_type='Gemm',
            inputs=['a', 'b', 'c'],
            outputs=['Y'],
            alpha=alpha,
            beta=beta,
            transA=trans_a,
            transB=trans_b
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[gemm_node],
            name='test-gather',
            inputs=[a, b, c],
            outputs=[Y],
            initializer=[init_b, init_c]
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        bigdl_model = Gemm(b_val, c_val,
                           alpha=alpha, beta=beta, trans_a=trans_a, trans_b=trans_b)
        loaded_model = load_model_proto(onnx_model)

        expected_out = bigdl_model.forward(input_x)
        loaded_out = loaded_model.forward(input_x)

        assert(np.array_equal(expected_out, loaded_out))


class TestMaxPool(object):

    def test_max_poll(self):
        kernel_width, kernel_height = 2, 2
        stride_width, stride_height = 1, 1
        pad_width, pad_height = 0, 0
        ceil_mode = 0
        input_shape = [1, 3, 224, 224]
        output_shape = [1, 3, 223, 223]
        input_x = np.random.random(input_shape)

        # Create one output (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        maxpool_node = onnx.helper.make_node(
            op_type='MaxPool',
            inputs=['X'],
            outputs=['Y'],
            kernel_shape=(kernel_width, kernel_height),
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[maxpool_node],
            name='test-maxpool',
            inputs=[X],
            outputs=[Y],
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        loaded_model = load_model_proto(onnx_model)
        bigdl_model = SpatialMaxPooling(
            kw=kernel_width,
            kh=kernel_height,
            dw=stride_width,
            dh=stride_height,
            pad_w=pad_width,
            pad_h=pad_height,
            to_ceil=False if ceil_mode == 0 else True
        )

        loaded_out = loaded_model.forward(input_x)
        expected_out = bigdl_model.forward(input_x)

        assert(np.array_equal(expected_out, loaded_out))


class TestRelu(object):

    def test_relu(self):
        input_shape = [1, 3, 224, 224]
        output_shape = [1, 3, 224, 224]
        input_x = np.random.random(input_shape)

        # Create one output (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        relu_node = onnx.helper.make_node(
            op_type='Relu',
            inputs=['X'],
            outputs=['Y']
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[relu_node],
            name='test-relu',
            inputs=[X],
            outputs=[Y],
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        bigdl_model = ReLU()
        loaded_model = load_model_proto(onnx_model)

        expected_out = bigdl_model.forward(input_x)
        loaded_out = loaded_model.forward(input_x)

        assert(np.array_equal(expected_out, loaded_out))


class TestReshape(object):

    def test_reshape(self):

        input_x = np.random.random([1, 3, 4, 4])
        # Create one output (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 4, 4])
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [2, 3, 8])
        shape = onnx.helper.make_tensor_value_info('shape', onnx.TensorProto.FLOAT, [1, 3])

        init_shape = onnx.helper.make_tensor(
            name='shape',  # type: Text
            data_type=onnx.TensorProto.FLOAT,  # type: int
            dims=[1, 3],  # type: Sequence[int]
            vals=[2, 3, 8],  # type: Any
        )

        reshape_node = onnx.helper.make_node(
            op_type='Reshape',
            inputs=['X', 'shape'],
            outputs=['Y']
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[reshape_node],
            name='test-reshape',
            inputs=[X, shape],
            outputs=[Y],
            initializer=[init_shape]
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        loaded_model = load_model_proto(onnx_model)
        bigdl_model = Reshape([2, 3, 8])

        loaded_out = loaded_model.forward(input_x)
        expected_out = bigdl_model.forward(input_x)

        assert(np.array_equal(expected_out, loaded_out))


class TestShape(object):

    def test_shape(self):
        input_shape = [3, 4, 5]
        input_x = np.random.random(input_shape)
        # Create one output (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1])

        shape_node = onnx.helper.make_node(
            op_type='Shape',
            inputs=['X'],
            outputs=['Y'],
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[shape_node],
            name='test-shape',
            inputs=[X],
            outputs=[Y],
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        bigdl_model = Shape()
        loaded_model = load_model_proto(onnx_model)

        expected_out = bigdl_model.forward(input_x)
        loaded_out = loaded_model.forward(input_x)

        assert(np.array_equal(expected_out, loaded_out))


class TestSoftmax(object):

    def test_softmax(self):
        input_shape = [1, 3, 224, 224]
        output_shape = [1, 3, 224, 224]
        input_x = np.random.random(input_shape)
        # Create one output (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        softmax_node = onnx.helper.make_node(
            op_type='Softmax',
            inputs=['X'],
            outputs=['Y']
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[softmax_node],
            name='test-softmax',
            inputs=[X],
            outputs=[Y],
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        loaded_model = load_model_proto(onnx_model)
        bigdl_model = SoftMax()

        loaded_out = loaded_model.forward(input_x)
        expected_out = bigdl_model.forward(input_x)

        assert(np.array_equal(expected_out, loaded_out))


class TestSum(object):

    def test_sum(self):
        input_shape = [2, 3]
        input_x1 = np.random.random(input_shape)
        input_x2 = np.random.random(input_shape)
        # Create one output (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [4, 3])
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 3])

        sum_node = onnx.helper.make_node(
            op_type='Sum',
            inputs=['X'],
            outputs=['Y']
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[sum_node],
            name='test-sum',
            inputs=[X],
            outputs=[Y],
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        loaded_model = load_model_proto(onnx_model)
        bigdl_model = CAddTable()

        expected_out = bigdl_model.forward([input_x1, input_x2])
        loaded_out = loaded_model.forward([input_x1, input_x2])

        assert(np.array_equal(expected_out, loaded_out))


class TestUnsqueeze(object):

    def test_unsqueeze(self):
        axis = 0
        input_shape = [3, 4, 5]
        output_shape = [1, 3, 4, 5]
        input_x = np.random.random([3, 4, 5])

        # Create one output (ValueInfoProto)
        X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, input_shape)
        Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, output_shape)

        unsqueeze_node = onnx.helper.make_node(
            op_type='Unsqueeze',
            inputs=['X'],
            outputs=['Y'],
            axes=[axis],
        )

        # Create the graph (GraphProto)
        onnx_graph = onnx.helper.make_graph(
            nodes=[unsqueeze_node],
            name='test-unsqueeze',
            inputs=[X],
            outputs=[Y],
        )

        # Create the model (ModelProto)
        onnx_model = onnx.helper.make_model(onnx_graph, producer_name='ONNX')
        onnx.checker.check_model(onnx_model)

        bigdl_model = Unsqueeze(pos=axis, num_input_dims=len(input_shape))
        expected_out = bigdl_model.forward(input_x)

        loaded_model = load_model_proto(onnx_model)
        loaded_out = loaded_model.forward(input_x)

        assert(np.array_equal(expected_out, loaded_out))


def main():
    pytest.main([__file__])


if __name__ == "__main__":
    pytest.main([__file__])
