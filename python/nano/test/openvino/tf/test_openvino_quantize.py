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
from unittest import TestCase
import tensorflow as tf
import pytest
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from bigdl.nano.tf.keras import Model, InferenceOptimizer
import tempfile
from bigdl.nano.deps.openvino.tf.model import KerasOpenVINOModel


class TestOpenVINO(TestCase):
    def test_model_quantize_openvino(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

        # Case1: Trace and quantize
        openvino_model = InferenceOptimizer.trace(model, accelerator='openvino')
        openvino_quantized_model = InferenceOptimizer.quantize(openvino_model,
                                                               accelerator='openvino',
                                                               x=train_dataset,
                                                               thread_num=8)

        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

        y_hat = openvino_quantized_model.predict(train_examples, batch_size=5)
        assert y_hat.shape == (100, 10)

        # Case2: Quantize directly from tensorflow
        openvino_quantized_model = InferenceOptimizer.quantize(model,
                                                               accelerator='openvino',
                                                               x=train_dataset)
        
        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

        preds = model.predict(train_examples)
        openvino_preds = openvino_quantized_model.predict(train_examples)
        np.testing.assert_allclose(preds, openvino_preds, rtol=1e-2)

        # Case 3: with config
        openvino_quantized_model = InferenceOptimizer.quantize(openvino_model,
                                                               accelerator='openvino',
                                                               x=train_dataset,
                                                               openvino_config={"PERFORMANCE_HINT": "LATENCY"})
        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

        # Case 4: with kargs
        openvino_quantized_model = InferenceOptimizer.quantize(openvino_model,
                                                               accelerator='openvino',
                                                               x=train_dataset,
                                                               thread_num=8,
                                                               mean_value=[123.68,116.78,103.94]) # mo param
        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

    def test_model_quantize_openvino_without_dataset(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))

        openvino_quantized_model = InferenceOptimizer.quantize(model,
                                                               accelerator='openvino',
                                                               x=train_examples,
                                                               y=train_labels)

        preds = model.predict(train_examples)
        openvino_preds = openvino_quantized_model.predict(train_examples)
        np.testing.assert_allclose(preds, openvino_preds, rtol=1e-2)

    def test_model_quantize_openvino_with_only_x(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        # test numpy array
        train_examples = np.random.random((100, 40, 40, 3))
        openvino_quantized_model = InferenceOptimizer.quantize(model,
                                                               accelerator='openvino',
                                                               x=train_examples,
                                                               y=None)

        preds = model.predict(train_examples)
        openvino_preds = openvino_quantized_model.predict(train_examples)
        np.testing.assert_allclose(preds, openvino_preds, rtol=1e-2)
        
        # test tf tensor
        train_examples = tf.convert_to_tensor(train_examples)
        openvino_quantized_model = InferenceOptimizer.quantize(model,
                                                               accelerator='openvino',
                                                               x=train_examples,
                                                               y=None)

        preds = model.predict(train_examples)
        openvino_preds = openvino_quantized_model.predict(train_examples)
        np.testing.assert_allclose(preds, openvino_preds, rtol=1e-2)
        
        # test dataset
        train_examples = np.random.random((100, 40, 40, 3))
        train_dataset = tf.data.Dataset.from_tensor_slices(train_examples)
        openvino_quantized_model = InferenceOptimizer.quantize(model,
                                                               accelerator='openvino',
                                                               x=train_dataset,
                                                               y=None)

        preds = model.predict(train_examples)
        openvino_preds = openvino_quantized_model.predict(train_examples)
        np.testing.assert_allclose(preds, openvino_preds, rtol=1e-2)

    def test_model_quantize_openvino_bf16(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))

        try:
            openvino_quantized_model = InferenceOptimizer.quantize(model,
                                                                accelerator='openvino',
                                                                precision='bf16')
        except RuntimeError as e:
            assert e.__str__() == "Platform doesn't support BF16 format"
            return

        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

        y_hat = openvino_quantized_model.predict(train_examples, batch_size=5)
        assert y_hat.shape == (100, 10)

        preds = model.predict(train_examples)
        openvino_preds = openvino_quantized_model.predict(train_examples)
        np.testing.assert_allclose(preds, openvino_preds, rtol=1e-2)

    def test_model_quantize_openvino_save_load(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

        openvino_quantized_model = InferenceOptimizer.quantize(model,
                                                               accelerator='openvino',
                                                               x=train_dataset,
                                                               thread_num=8)
        
        # test original save / load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            openvino_quantized_model._save(tmp_dir_name)
            new_ov_model = KerasOpenVINOModel._load(tmp_dir_name)

        preds1 = openvino_quantized_model(train_examples).numpy()
        preds2 = new_ov_model(train_examples).numpy()
        np.testing.assert_almost_equal(preds1, preds2, decimal=5)

        # test InferencOptimizer save / load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(openvino_quantized_model, tmp_dir_name)
            new_ov_model = InferenceOptimizer.load(tmp_dir_name, model)

        preds1 = openvino_quantized_model(train_examples).numpy()
        preds2 = new_ov_model(train_examples).numpy()
        np.testing.assert_almost_equal(preds1, preds2, decimal=5)
    
    def test_model_quantize_openvino_vpu(self):
        # test whether contains VPU
        from openvino.runtime import Core
        core = Core()
        devices = core.available_devices
        vpu_avaliable = any('VPUX' in x for x in devices)
        
        if vpu_avaliable is False:
            return
        
        # test VPUX int8
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

        openvino_model = InferenceOptimizer.trace(model, accelerator='openvino')
        openvino_quantized_model = InferenceOptimizer.quantize(openvino_model,
                                                               accelerator='openvino',
                                                               x=train_dataset,
                                                               thread_num=8,
                                                               device='VPUX',
                                                               precision='int8')
        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)
        
        # test VPUX fp16
        with pytest.raises(RuntimeError):
            openvino_quantized_model = InferenceOptimizer.quantize(openvino_model,
                                                                   accelerator='openvino',
                                                                   x=train_dataset,
                                                                   thread_num=8,
                                                                   device='VPUX',
                                                                   precision='fp16')
        
        openvino_quantized_model = InferenceOptimizer.quantize(openvino_model,
                                                               accelerator='openvino',
                                                               x=train_dataset,
                                                               thread_num=8,
                                                               device='VPUX',
                                                               precision='fp16',
                                                               mean_value=[127.5, 127.5, 127.5])
        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

    def test_model_quantize_openvino_fp16(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

        openvino_quantized_model = InferenceOptimizer.quantize(model,
                                                               accelerator='openvino',
                                                               x=train_dataset,
                                                               thread_num=8,
                                                               precision='fp16')

        # test original save / load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(openvino_quantized_model, tmp_dir_name)
            new_ov_model = InferenceOptimizer.load(tmp_dir_name, model)

        preds1 = openvino_quantized_model(train_examples).numpy()
        preds2 = new_ov_model(train_examples).numpy()
        np.testing.assert_almost_equal(preds1, preds2, decimal=5)
