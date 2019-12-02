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
import numpy as np


def make_avgpool_onnx_model():
    # avgpool
    avgpool_ceil_mode = 0
    avgpool_kernel_width, avgpool_kernel_height = 3, 3
    avgpool_pad_width, avgpool_pad_height = 0, 0
    avgpool_stride_width, avgpool_stride_height = 1, 1
    avgpool_input_shape = [1, 3, 224, 224]
    avgpool_output_shape = [1, 3, 222, 222]
    avgpool_input_x = np.random.random(avgpool_input_shape).astype('float32')
    avgpool_model_path = "./onnx_model/avgpool.onnx"

    # Create one input (ValueInfoProto)
    avgpool_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, avgpool_input_shape)
    # Create one output (ValueInfoProto)
    avgpool_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, avgpool_output_shape)

    # Create a node (NodeProto)
    avgpool_node = onnx.helper.make_node(
        op_type='AveragePool',  # node name
        inputs=['X'],  # inputs
        outputs=['Y'],  # outputs
        auto_pad='NOTSET',
        ceil_mode=avgpool_ceil_mode,
        kernel_shape=(avgpool_kernel_width, avgpool_kernel_height),
        pads=(avgpool_pad_width, avgpool_pad_height,
              avgpool_pad_width, avgpool_pad_height),
        strides=(avgpool_stride_width, avgpool_stride_height)
    )

    # Create the graph (GraphProto)
    avgpool_onnx_graph = onnx.helper.make_graph(
        nodes=[avgpool_node],
        name='test-averagePool',
        inputs=[avgpool_X],
        outputs=[avgpool_Y],
    )

    # Create the model (ModelProto)
    avgpool_onnx_model = onnx.helper.make_model(
        avgpool_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(avgpool_onnx_model)
    onnx.save(avgpool_onnx_model, avgpool_model_path)

    return avgpool_model_path

# def make_batchnorm_onnx_model():
#     # BatchNormalization
#     batchnorm_input_shape = [1, 3, 224, 224]
#     batchnorm_output_shape = [1, 3, 224, 224]
#     batchnorm_scale_vals = np.random.random(batchnorm_input_shape[1]) * 10
#     batchnorm_bias_vals = np.random.random(batchnorm_input_shape[1]) * 10
#     batchnorm_mean_vals = np.random.random(batchnorm_input_shape[1]) * 10
#     batchnorm_var_vals = np.random.random(batchnorm_input_shape[1]) * 10
#     batchnorm_input_x = np.random.random(input_shape) * 10
#     batchnorm_epsilon = float(1e-05)
#     batchnorm_momentum = float(0.9)
#
#     batchnorm_input_x = np.random.random(input_shape).astype('float32')
#     batchnorm_model_path = "./onnx_model/batchnorm.onnx"
#
#     return

def make_concat_onnx_model():
    # Concat
    concat_axis = 0
    concat_input_shape = [2, 3]
    concat_output_shape = [4, 3]
    concat_x1_val = np.random.random(concat_input_shape).astype("float32")
    concat_x2_val = np.random.random(concat_input_shape).astype("float32")
    concat_model_path = "./onnx_model/concat.onnx"

    # Create input (ValueInfoProto)
    concat_X1 = onnx.helper.make_tensor_value_info(
        'X1', onnx.TensorProto.FLOAT, concat_input_shape)
    concat_X2 = onnx.helper.make_tensor_value_info(
        'X2', onnx.TensorProto.FLOAT, concat_input_shape)
    concat_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, [2] + concat_input_shape)
    concat_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, concat_output_shape)

    # Create a node (NodeProto)
    concat_const_X1 = onnx.helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['X1'],
        value=onnx.helper.make_tensor(
            name='X1',
            data_type=onnx.TensorProto.FLOAT,
            dims=concat_input_shape,
            vals=concat_x1_val.flatten().tolist(),
        )
    )

    concat_const_X2 = onnx.helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['X2'],
        value=onnx.helper.make_tensor(
            name='X2',
            data_type=onnx.TensorProto.FLOAT,
            dims=concat_input_shape,
            vals=concat_x2_val.flatten().tolist(),
        )
    )

    concat_node = onnx.helper.make_node(
        op_type='Concat',  # node name
        inputs=['X1', 'X2'],  # inputs
        outputs=['Y'],  # outputs
        axis=concat_axis
    )

    # Create the graph (GraphProto)
    concat_onnx_graph = onnx.helper.make_graph(
        nodes=[concat_const_X1, concat_const_X2, concat_node],
        name='test-concat',
        inputs=[concat_X],
        outputs=[concat_Y],
    )

    # Create the model (ModelProto)
    concat_onnx_model = onnx.helper.make_model(
        concat_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(concat_onnx_model)
    onnx.save(concat_onnx_model, concat_model_path)

    return concat_model_path


def make_constant_onnx_model():
    # Constant
    constant_shape = [5, 5]
    constant_values = np.float32(np.round(np.random.random(constant_shape), 6))
    constant_dummy_input = np.random.random([1]).astype('float32')
    constant_model_path = "./onnx_model/constant.onnx"

    # Create one output (ValueInfoProto)
    constant_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, [1])
    constant_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, constant_values.shape)

    constant_node = onnx.helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['Y'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=onnx.TensorProto.FLOAT,
            dims=constant_values.shape,
            vals=constant_values.flatten().tolist(),
        ),
    )

    # Create the graph (GraphProto)
    constant_onnx_graph = onnx.helper.make_graph(
        nodes=[constant_node],
        name='test-constant',
        inputs=[constant_X],
        outputs=[constant_Y],
    )

    # Create the model (ModelProto)
    constant_onnx_model = onnx.helper.make_model(
        constant_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(constant_onnx_model)
    onnx.save(constant_onnx_model, constant_model_path)

    return constant_model_path


def make_conv_onnx_model():
    # Convolution
    conv_kernel_width, conv_kernel_height = (3, 3)
    conv_stride_width, conv_stride_height = (1, 1)
    conv_pad_width, conv_pad_height = (0, 0)
    conv_input_shape = [1, 3, 224, 224]
    conv_output_shape = [1, 8, 222, 222]
    conv_weight_shape = [8, 3, 3, 3]
    conv_input_x = np.random.random(conv_input_shape).astype('float32')
    conv_weight_values = np.random.random(conv_weight_shape).astype('float32')
    conv_model_path = "./onnx_model/conv.onnx"

    # Create input (ValueInfoProto)
    conv_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, conv_input_shape)
    conv_W = onnx.helper.make_tensor_value_info(
        'W', onnx.TensorProto.FLOAT, conv_weight_shape)
    conv_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, conv_output_shape)

    conv_init_weight = onnx.helper.make_tensor(
        name='W',
        data_type=onnx.TensorProto.FLOAT,
        dims=conv_weight_shape,
        vals=conv_weight_values.flatten().astype(float),
    )

    conv_node = onnx.helper.make_node(
        op_type='Conv',
        inputs=['X', 'W'],
        outputs=['Y'],
        kernel_shape=(conv_kernel_width, conv_kernel_height),
    )

    # Create the graph (GraphProto)
    conv_onnx_graph = onnx.helper.make_graph(
        nodes=[conv_node],
        name='test-conv',
        inputs=[conv_X],
        outputs=[conv_Y],
        initializer=[conv_init_weight]
    )

    # Create the model (ModelProto)
    conv_onnx_model = onnx.helper.make_model(
        conv_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(conv_onnx_model)
    onnx.save(conv_onnx_model, conv_model_path)

    return conv_model_path


def make_gather_onnx_model():
    # Gather
    gather_axis = 0
    gather_input_x = np.array([
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
    ], dtype=float)
    gather_indices_val = np.array([[0, 1], [1, 2]], dtype=int)
    gather_expected_out = np.array([[[1, 1.2], [2.3, 3.4]],
                                    [[2.3, 3.4], [4.5, 5.7]]],
                                   dtype=float)
    gather_input_shape = gather_input_x.shape
    gather_indices_shape = gather_indices_val.shape
    gather_output_shape = [2, 2, 2]
    gather_model_path = "./onnx_model/gather.onnx"

    # Create one output (ValueInfoProto)
    gather_data_tvi = onnx.helper.make_tensor_value_info(
        'data', onnx.TensorProto.FLOAT, gather_input_shape)

    gather_const_data_tvi = onnx.helper.make_tensor_value_info(
        'const_data', onnx.TensorProto.FLOAT, gather_input_shape)

    gather_indices_tvi = onnx.helper.make_tensor_value_info(
        'indices', onnx.TensorProto.INT64, gather_indices_shape)

    gather_const_indices_tvi = onnx.helper.make_tensor_value_info(
        'const_indices', onnx.TensorProto.INT64, gather_indices_shape)

    gather_Y_tvi = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, gather_output_shape)

    # Create a node (NodeProto)
    gather_const_data = onnx.helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['const_data'],
        value=onnx.helper.make_tensor(
            name='const_data',
            data_type=onnx.TensorProto.FLOAT,
            dims=gather_input_shape,
            vals=gather_input_x.flatten().tolist(),
        )
    )

    gather_const_indices = onnx.helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['const_indices'],
        value=onnx.helper.make_tensor(
            name='const_indices',
            data_type=onnx.TensorProto.INT64,
            dims=gather_indices_shape,
            vals=gather_indices_val.flatten().tolist(),
        )
    )

    gather_node = onnx.helper.make_node(
        op_type='Gather',
        inputs=['const_data', 'const_indices'],
        outputs=['Y'],
        axis=gather_axis
    )

    # Create the graph (GraphProto)
    gather_onnx_graph = onnx.helper.make_graph(
        nodes=[gather_const_data, gather_const_indices, gather_node],
        name='test-gather',
        inputs=[gather_data_tvi, gather_indices_tvi],
        outputs=[gather_Y_tvi]
    )

    # Create the model (ModelProto)
    gather_onnx_model = onnx.helper.make_model(
        gather_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(gather_onnx_model)
    onnx.save(gather_onnx_model, gather_model_path)

    return gather_model_path


def make_gemm_onnx_model():
    # Gemm
    gemm_mata_shape = [2, 7]
    gemm_matb_shape = [7, 4]
    gemm_matc_shape = [2, 4]
    gemm_output_shape = [2, 4]
    gemm_alpha = np.round(np.random.rand(), 2)
    gemm_beta = np.round(np.random.rand(), 2)
    gemm_trans_a, gemm_trans_b = 0, 0
    gemm_input_x = np.random.random(gemm_mata_shape).astype('float32')
    gemm_b_val = np.random.random(gemm_matb_shape).astype('float32')
    gemm_c_val = np.random.random(gemm_matc_shape).astype('float32')
    gemm_model_path = "./onnx_model/gemm.onnx"

    # Create one output (ValueInfoProto)
    gemm_a = onnx.helper.make_tensor_value_info(
        'a', onnx.TensorProto.FLOAT, gemm_mata_shape)
    gemm_b = onnx.helper.make_tensor_value_info(
        'b', onnx.TensorProto.FLOAT, gemm_matb_shape)
    gemm_c = onnx.helper.make_tensor_value_info(
        'c', onnx.TensorProto.FLOAT, gemm_matc_shape)
    gemm_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, gemm_output_shape)

    gemm_init_b = onnx.helper.make_tensor(
        name='b',
        data_type=onnx.TensorProto.FLOAT,
        dims=gemm_matb_shape,
        vals=gemm_b_val.flatten().tolist(),
    )

    gemm_init_c = onnx.helper.make_tensor(
        name='c',
        data_type=onnx.TensorProto.FLOAT,
        dims=gemm_matc_shape,
        vals=gemm_c_val.flatten().tolist(),
    )

    gemm_node = onnx.helper.make_node(
        op_type='Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['Y'],
        alpha=gemm_alpha,
        beta=gemm_beta,
        transA=gemm_trans_a,
        transB=gemm_trans_b
    )

    # Create the graph (GraphProto)
    gemm_onnx_graph = onnx.helper.make_graph(
        nodes=[gemm_node],
        name='test-gather',
        inputs=[gemm_a, gemm_b, gemm_c],
        outputs=[gemm_Y],
        initializer=[gemm_init_b, gemm_init_c]
    )

    # Create the model (ModelProto)
    gemm_onnx_model = onnx.helper.make_model(
        gemm_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(gemm_onnx_model)
    onnx.save(gemm_onnx_model, gemm_model_path)

    return gemm_model_path


def make_maxpool_onnx_model():
    # MaxPool
    maxpool_kernel_width, maxpool_kernel_height = 2, 2
    maxpool_stride_width, maxpool_stride_height = 1, 1
    maxpool_pad_width, maxpool_pad_height = 0, 0
    maxpool_ceil_mode = 0
    maxpool_input_shape = [1, 3, 224, 224]
    maxpool_output_shape = [1, 3, 223, 223]
    maxpool_input = np.random.random(maxpool_input_shape).astype('float32')
    maxpool_model_path = "./onnx_model/maxpool.onnx"

    # Create one output (ValueInfoProto)
    maxpool_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, maxpool_input_shape)
    maxpool_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, maxpool_output_shape)

    maxpool_node = onnx.helper.make_node(
        op_type='MaxPool',
        inputs=['X'],
        outputs=['Y'],
        kernel_shape=(maxpool_kernel_width, maxpool_kernel_height),
    )

    # Create the graph (GraphProto)
    maxpool_onnx_graph = onnx.helper.make_graph(
        nodes=[maxpool_node],
        name='test-maxpool',
        inputs=[maxpool_X],
        outputs=[maxpool_Y],
    )

    # Create the model (ModelProto)
    maxpool_onnx_model = onnx.helper.make_model(
        maxpool_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(maxpool_onnx_model)
    onnx.save(maxpool_onnx_model, maxpool_model_path)

    return maxpool_model_path


def make_relu_onnx_model():
    # Relu
    relu_input_shape = [1, 3, 224, 224]
    relu_output_shape = [1, 3, 224, 224]
    relu_input = np.random.random(relu_input_shape).astype('float32')
    relu_model_path = "./onnx_model/relu.onnx"

    # Create one output (ValueInfoProto)
    relu_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, relu_input_shape)
    relu_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, relu_output_shape)

    relu_node = onnx.helper.make_node(
        op_type='Relu',
        inputs=['X'],
        outputs=['Y']
    )

    # Create the graph (GraphProto)
    relu_onnx_graph = onnx.helper.make_graph(
        nodes=[relu_node],
        name='test-relu',
        inputs=[relu_X],
        outputs=[relu_Y],
    )

    # Create the model (ModelProto)
    relu_onnx_model = onnx.helper.make_model(
        relu_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(relu_onnx_model)
    onnx.save(relu_onnx_model, relu_model_path)

    return relu_model_path


def make_reshape_onnx_model():
    # Reshape
    reshape_data_shape = [1, 3, 4, 4]
    reshape_data_val = np.random.random(reshape_data_shape).astype('float32')
    reshape_shape_val = np.array([2, 3, 8])
    reshape_model_path = "./onnx_model/reshape.onnx"

    # Create one output (ValueInfoProto)
    reshape_data_tvi = onnx.helper.make_tensor_value_info(
        'data', onnx.TensorProto.FLOAT, reshape_data_shape)
    reshape_const_data_tvi = onnx.helper.make_tensor_value_info(
        'const_data', onnx.TensorProto.FLOAT, reshape_data_shape)

    reshape_shape_tvi = onnx.helper.make_tensor_value_info(
        'shape', onnx.TensorProto.INT64, [3,])
    reshape_const_shape_tvi = onnx.helper.make_tensor_value_info(
        'const_shape', onnx.TensorProto.INT64, [3,])

    reshape_Y_tvi = onnx.helper.make_tensor_value_info('Y',
               onnx.TensorProto.FLOAT, [2, 3, 8])

    # Create a node (NodeProto)
    reshape_const_data = onnx.helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['const_data'],
        value=onnx.helper.make_tensor(
            name='const_data',
            data_type=onnx.TensorProto.FLOAT,
            dims=reshape_data_shape,
            vals=reshape_data_val.flatten().tolist(),
        )
    )

    reshape_const_shape = onnx.helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['const_shape'],
        value=onnx.helper.make_tensor(
            name='const_shape',
            data_type=onnx.TensorProto.INT64,
            dims=[3,],
            vals=reshape_shape_val.flatten().tolist(),
        )
    )

    reshape_node = onnx.helper.make_node(
        op_type='Reshape',
        inputs=['const_data', 'const_shape'],
        outputs=['Y']
    )

    # Create the graph (GraphProto)
    reshape_onnx_graph = onnx.helper.make_graph(
        nodes=[reshape_const_data, reshape_const_shape, reshape_node],
        name='test-reshape',
        inputs=[reshape_data_tvi, reshape_shape_tvi],
        outputs=[reshape_Y_tvi]
    )

    # Create the model (ModelProto)
    reshape_onnx_model = onnx.helper.make_model(
        reshape_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(reshape_onnx_model)
    onnx.save(reshape_onnx_model, reshape_model_path)

    return reshape_model_path


def make_shape_onnx_model():
    # Shape
    shape_input_shape = [3, 4, 5]
    shape_input_x = np.random.random(shape_input_shape).astype('float32')
    shape_model_path = "./onnx_model/shape.onnx"

    # Create one output (ValueInfoProto)
    shape_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, shape_input_shape)
    shape_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.INT64, [1])

    shape_node = onnx.helper.make_node(
        op_type='Shape',
        inputs=['X'],
        outputs=['Y'],
    )

    # Create the graph (GraphProto)
    shape_onnx_graph = onnx.helper.make_graph(
        nodes=[shape_node],
        name='test-shape',
        inputs=[shape_X],
        outputs=[shape_Y],
    )

    # Create the model (ModelProto)
    shape_onnx_model = onnx.helper.make_model(
        shape_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(shape_onnx_model)
    onnx.save(shape_onnx_model, shape_model_path)

    return shape_model_path


def make_softmax_onnx_model():
    # Softmax
    softmax_input_shape = [np.random.randint(1, 100)]
    softmax_output_shape = softmax_input_shape
    softmax_input_x = np.random.random(softmax_input_shape).astype('float32')
    # softmax_input_x = np.array([-1, 0, 1]).astype('float32')
    softmax_model_path = "./onnx_model/softmax.onnx"

    # Create one output (ValueInfoProto)
    softmax_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, softmax_input_shape)
    softmax_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, softmax_output_shape)

    softmax_node = onnx.helper.make_node(
        op_type='Softmax',
        inputs=['X'],
        outputs=['Y'],
        axis=0
    )

    # Create the graph (GraphProto)
    softmax_onnx_graph = onnx.helper.make_graph(
        nodes=[softmax_node],
        name='test-softmax',
        inputs=[softmax_X],
        outputs=[softmax_Y],
    )

    # Create the model (ModelProto)
    softmax_onnx_model = onnx.helper.make_model(
        softmax_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(softmax_onnx_model)
    onnx.save(softmax_onnx_model, softmax_model_path)

    return softmax_model_path


def make_sum_onnx_model():
    # Sum
    sum_input_shape = [2, 3]
    sum_input0_val = np.random.random(sum_input_shape).astype('float32')
    sum_input1_val = np.random.random(sum_input_shape).astype('float32')
    sum_model_path = "./onnx_model/sum.onnx"

    # Create one output (ValueInfoProto)
    sum_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, [4, 3])
    sum_X1 = onnx.helper.make_tensor_value_info(
        'X0', onnx.TensorProto.FLOAT, sum_input_shape)
    sum_X2 = onnx.helper.make_tensor_value_info(
        'X1', onnx.TensorProto.FLOAT, sum_input_shape)
    sum_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, sum_input_shape)

    # Create a node (NodeProto)
    sum_const_X0 = onnx.helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['X0'],
        value=onnx.helper.make_tensor(
            name='X0',
            data_type=onnx.TensorProto.FLOAT,
            dims=sum_input_shape,
            vals=sum_input0_val.flatten().tolist(),
        )
    )

    sum_const_X1 = onnx.helper.make_node(
        op_type='Constant',
        inputs=[],
        outputs=['X1'],
        value=onnx.helper.make_tensor(
            name='X1',
            data_type=onnx.TensorProto.FLOAT,
            dims=sum_input_shape,
            vals=sum_input1_val.flatten().tolist(),
        )
    )

    sum_node = onnx.helper.make_node(
        op_type='Sum',
        inputs=['X0', 'X1'],
        outputs=['Y']
    )

    # Create the graph (GraphProto)
    sum_onnx_graph = onnx.helper.make_graph(
        nodes=[sum_const_X0, sum_const_X1, sum_node],
        name='test-sum',
        inputs=[sum_X],
        outputs=[sum_Y],
    )

    # Create the model (ModelProto)
    sum_onnx_model = onnx.helper.make_model(
        sum_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(sum_onnx_model)
    onnx.save(sum_onnx_model, sum_model_path)

    return sum_model_path


def make_unsqueeze_onnx_model():
    # Unsqueeze
    unsqueeze_axis = 0 # fix axis
    unsqueeze_X_shape = [3, 4, 5]
    unsqueeze_Y_shape = [1, 3, 4, 5]
    unsqueeze_X_val = np.random.random(unsqueeze_X_shape).astype('float32')
    unsqueeze_model_path = "./onnx_model/unsqueeze.onnx"

    # Create one output (ValueInfoProto)
    unsqueeze_X = onnx.helper.make_tensor_value_info(
        'X', onnx.TensorProto.FLOAT, unsqueeze_X_shape)
    unsqueeze_Y = onnx.helper.make_tensor_value_info(
        'Y', onnx.TensorProto.FLOAT, unsqueeze_Y_shape)

    unsqueeze_node = onnx.helper.make_node(
        op_type='Unsqueeze',
        inputs=['X'],
        outputs=['Y'],
        axes=[unsqueeze_axis],
    )

    # Create the graph (GraphProto)
    unsqueeze_onnx_graph = onnx.helper.make_graph(
        nodes=[unsqueeze_node],
        name='test-unsqueeze',
        inputs=[unsqueeze_X],
        outputs=[unsqueeze_Y],
    )

    # Create the model (ModelProto)
    unsqueeze_onnx_model = onnx.helper.make_model(
        unsqueeze_onnx_graph, producer_name='ONNX')
    onnx.checker.check_model(unsqueeze_onnx_model)
    onnx.save(unsqueeze_onnx_model, unsqueeze_model_path)

    return unsqueeze_model_path


if __name__ == '__main__':
    make_avgpool_onnx_model()
