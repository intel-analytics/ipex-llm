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
import onnxruntime as rt

from bigdl.contrib.onnx import load
from .test_model_generator import *


class TestAveragePool(object):

    def test_average_pool(self):
        avgpool_model_path = make_avgpool_onnx_model()
        bigdl_avgpool = load(avgpool_model_path)
        avgpool_sess = rt.InferenceSession(avgpool_model_path)

        avgpool_input_shape = [1, 3, 224, 224]
        avgpool_input_x = np.random.random(avgpool_input_shape).astype('float32')
        avgpool_input_name = avgpool_sess.get_inputs()[0].name
        avgpool_output_name = avgpool_sess.get_outputs()[0].name

        bigdl_avgpool_out = bigdl_avgpool.forward(avgpool_input_x)

        try:
            rt_avgpool_out = avgpool_sess.run(
                [avgpool_output_name],
                {avgpool_input_name: avgpool_input_x})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_avgpool_out, rt_avgpool_out, decimal=5))


# class TestBatchNormalization(object):
#
#     def test_batch_normalization(self):
#         assert(np.array_equal(loaded_out, expected_out))


class TestConcat(object):

    def test_concat(self):
        concat_model_path = make_concat_onnx_model()
        bigdl_concat = load(concat_model_path)

        concat_sess = rt.InferenceSession(concat_model_path)
        concat_input_name = concat_sess.get_inputs()[0].name
        concat_output_name = concat_sess.get_outputs()[0].name

        concat_input_shape = [2, 3]
        concat_x1_val = np.random.random(concat_input_shape).astype("float32")
        concat_x2_val = np.random.random(concat_input_shape).astype("float32")

        bigdl_concat_out = bigdl_concat.forward(np.array([1], dtype='float32'))

        try:
            rt_concat_out = concat_sess.run(
                [concat_output_name],
                {concat_input_name: np.array([concat_x1_val, concat_x2_val])})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_concat_out, rt_concat_out, decimal=5))


class TestConstant(object):

    def test_constant(self):
        constant_model_path = make_constant_onnx_model()
        bigdl_constant = load(constant_model_path)
        constant_sess = rt.InferenceSession(constant_model_path)
        constant_input_name = constant_sess.get_inputs()[0].name
        constant_output_name = constant_sess.get_outputs()[0].name

        bigdl_constant_out = bigdl_constant.forward(np.array([1], dtype='float32'))

        try:
            rt_constant_out = constant_sess.run(
                [constant_output_name],
                {constant_input_name: np.array([1]).astype('float32')})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_constant_out, rt_constant_out, decimal=5))


class TestConv(object):

    def test_conv(self):
        conv_model_path = make_conv_onnx_model()
        bigdl_conv = load(conv_model_path)
        conv_sess = rt.InferenceSession(conv_model_path)
        conv_input_name = conv_sess.get_inputs()[0].name
        conv_output_name = conv_sess.get_outputs()[0].name

        conv_input_shape = [1, 3, 224, 224]
        conv_input_x = np.random.random(conv_input_shape).astype('float32')

        bigdl_conv_out = bigdl_conv.forward(conv_input_x)

        try:
            rt_conv_out = conv_sess.run(
                [conv_output_name],
                {conv_input_name: conv_input_x})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_conv_out, rt_conv_out, decimal=5))


class TestGather(object):

    def test_gather(self):
        gather_model_path = make_gather_onnx_model()
        bigdl_gather = load(gather_model_path)
        gather_sess = rt.InferenceSession(gather_model_path)
        gather_input_x_name = gather_sess.get_inputs()[0].name
        gather_input_indices_name = gather_sess.get_inputs()[1].name
        gather_output_name = gather_sess.get_outputs()[0].name

        gather_input_x = np.array([
            [1.0, 1.2],
            [2.3, 3.4],
            [4.5, 5.7],
        ])
        gather_indices_val = np.array([[0, 1], [1, 2]])

        bigdl_gather_out = bigdl_gather.forward(np.array([gather_input_x]))

        try:
            rt_gather_out = gather_sess.run(
                [gather_output_name],
                {gather_input_x_name: gather_input_x.astype('float32'),
                 gather_input_indices_name: gather_indices_val.astype('int64')})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_gather_out, rt_gather_out, decimal=5))


class TestGemm(object):

    def test_gemm(self):
        gemm_model_path = make_gemm_onnx_model()
        bigdl_gemm = load(gemm_model_path)
        gemm_sess = rt.InferenceSession(gemm_model_path)
        gemm_input_name = gemm_sess.get_inputs()[0].name
        gemm_output_name = gemm_sess.get_outputs()[0].name
        gemm_mata_shape = [2, 7]
        gemm_input_x = np.random.random(gemm_mata_shape).astype('float32')

        bigdl_gemm_out = bigdl_gemm.forward(gemm_input_x)

        try:
            rt_gemm_out = gemm_sess.run(
                [gemm_output_name],
                {gemm_input_name: gemm_input_x})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_gemm_out, rt_gemm_out, decimal=5))


class TestMaxPool(object):

    def test_max_poll(self):
        maxpool_model_path = make_maxpool_onnx_model()
        bigdl_maxpool = load(maxpool_model_path)
        maxpool_sess = rt.InferenceSession(maxpool_model_path)
        maxpool_input_name = maxpool_sess.get_inputs()[0].name
        maxpool_output_name = maxpool_sess.get_outputs()[0].name
        maxpool_input_shape = [1, 3, 224, 224]
        maxpool_input = np.random.random(maxpool_input_shape).astype('float32')

        bigdl_maxpool_out = bigdl_maxpool.forward(maxpool_input)

        try:
            rt_maxpool_out = maxpool_sess.run(
                [maxpool_output_name],
                {maxpool_input_name: maxpool_input})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_maxpool_out, rt_maxpool_out, decimal=5))


class TestRelu(object):

    def test_relu(self):
        relu_model_path = make_relu_onnx_model()
        bigdl_relu = load(relu_model_path)
        relu_sess = rt.InferenceSession(relu_model_path)
        relu_input_name = relu_sess.get_inputs()[0].name
        relu_output_name = relu_sess.get_outputs()[0].name

        relu_input_shape = [1, 3, 224, 224]
        relu_input = np.random.random(relu_input_shape).astype('float32')

        bigdl_relu_out = bigdl_relu.forward(relu_input)

        try:
            rt_relu_out = relu_sess.run(
                [relu_output_name],
                {relu_input_name: relu_input})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_relu_out, rt_relu_out, decimal=5))


class TestReshape(object):

    def test_reshape(self):
        reshape_model_path = make_shape_onnx_model()
        bigdl_reshape = load(reshape_model_path)
        reshape_sess = rt.InferenceSession(reshape_model_path)
        reshape_input_data_name = reshape_sess.get_inputs()[0].name
        reshape_input_shape_name = reshape_sess.get_inputs()[1].name
        reshape_output_name = reshape_sess.get_outputs()[0].name

        reshape_data_shape = [1, 3, 4, 4]
        reshape_data_val = np.random.random(reshape_data_shape).astype('float32')
        reshape_shape_val = np.array([2, 3, 8])

        bigdl_reshape_out = bigdl_reshape.forward(reshape_data_val)

        try:
            rt_reshape_out = reshape_sess.run(
                [reshape_output_name],
                {reshape_input_data_name: reshape_data_val,
                 reshape_input_shape_name: reshape_shape_val})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_reshape_out, rt_reshape_out, decimal=5))


class TestShape(object):

    def test_shape(self):
        shape_model_path = make_shape_onnx_model()
        bigdl_shape = load(shape_model_path)
        shape_sess = rt.InferenceSession(shape_model_path)
        shape_input_name = shape_sess.get_inputs()[0].name
        shape_output_name = shape_sess.get_outputs()[0].name

        shape_input_shape = [3, 4, 5]
        shape_input_x = np.random.random(shape_input_shape).astype('float32')
        bigdl_shape_out = bigdl_shape.forward(shape_input_x)

        try:
            rt_shape_out = shape_sess.run(
                [shape_output_name],
                {shape_input_name: shape_input_x})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_shape_out, rt_shape_out, decimal=5))


# class TestSoftmax(object):
#
#     def test_softmax(self):
#         softmax_model_path = make_softmax_onnx_model()
#         bigdl_softmax = load(softmax_model_path)
#         softmax_sess = rt.InferenceSession(softmax_model_path)
#         softmax_input_name = softmax_sess.get_inputs()[0].name
#         softmax_output_name = softmax_sess.get_outputs()[0].name
#
#         softmax_input_shape = [np.random.randint(1, 100)]
#         softmax_input_x = np.random.random(softmax_input_shape).astype('float32')
#         bigdl_softmax_out = bigdl_softmax.forward(softmax_input_x)
#
#         try:
#             rt_softmax_out = softmax_sess.run(
#             [softmax_output_name], {softmax_input_name: softmax_input_x})
#         except Exception as e:
#             print("Unexpected type")
#             print("{0}: {1}".format(type(e), e))
#
#         assert(np.testing.assert_array_almost_equal(
#         bigdl_softmax_out, rt_softmax_out, decimal=5))


class TestSum(object):

    def test_sum(self):
        sum_model_path = make_sum_onnx_model()
        bigdl_sum = load(sum_model_path)
        sum_sess = rt.InferenceSession(sum_model_path)
        sum_input_name = sum_sess.get_inputs()[0].name
        sum_output_name = sum_sess.get_outputs()[0].name

        sum_input_shape = [2, 3]
        sum_input0_val = np.random.random(sum_input_shape).astype('float32')
        sum_input1_val = np.random.random(sum_input_shape).astype('float32')
        bigdl_sum_out = bigdl_sum.forward([sum_input0_val, sum_input1_val])

        try:
            rt_sum_out = sum_sess.run(
                [sum_output_name],
                {sum_input_name:
                    np.concatenate([sum_input0_val,
                                    sum_input1_val]).astype('float32')})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_sum_out, rt_sum_out, decimal=5))


class TestUnsqueeze(object):

    def test_unsqueeze(self):
        unsqueeze_model_path = make_unsqueeze_onnx_model()
        bigdl_unsqueeze = load(unsqueeze_model_path)
        unsqueeze_sess = rt.InferenceSession(unsqueeze_model_path)
        unsqueeze_input_name = unsqueeze_sess.get_inputs()[0].name
        unsqueeze_output_name = unsqueeze_sess.get_outputs()[0].name

        unsqueeze_X_shape = [3, 4, 5]
        unsqueeze_X_val = np.random.random(unsqueeze_X_shape).astype('float32')

        bigdl_unsqueeze_out = bigdl_unsqueeze.forward(unsqueeze_X_val)

        try:
            rt_unsqueeze_out = unsqueeze_sess.run(
                [unsqueeze_output_name],
                {unsqueeze_input_name: unsqueeze_X_val})[0]
        except Exception as e:
            print("Unexpected type")
            print("{0}: {1}".format(type(e), e))

        assert(np.testing.assert_array_almost_equal(
            bigdl_unsqueeze_out, rt_unsqueeze_out, decimal=5))


if __name__ == "__main__":
    pytest.main([__file__])
