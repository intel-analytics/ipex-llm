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
import tempfile
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
from bigdl.nano.tf.keras import Model, InferenceOptimizer
from bigdl.nano.deps.onnxruntime.tensorflow.model \
    import KerasONNXRuntimeModel


class TestONNX(TestCase):
    def test_model_quantize_onnx(self):
        model = EfficientNetB0(weights=None, input_shape=[224, 224, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        input_examples = np.random.random((100, 224, 224, 3)).astype(np.float32)
        input_features = np.random.randint(0, 10, size=100)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_examples,
                                                            input_features))

        # quantize a Keras model
        onnx_quantized_model = InferenceOptimizer.quantize(model,
                                                           accelerator='onnxruntime',
                                                           x=train_dataset,
                                                           thread_num=8,
                                                           accuracy_criterion={'relative': 0.99,
                                                                               'higher_is_better': True})

        y_hat = onnx_quantized_model(input_examples[:10])
        assert y_hat.shape == (10, 10)

        y_hat = onnx_quantized_model.predict(input_examples, batch_size=5)
        assert y_hat.shape == (100, 10)

        preds = model.predict(input_examples, batch_size=5)
        onnx_preds = onnx_quantized_model.predict(input_examples, batch_size=5)
        np.testing.assert_allclose(preds, onnx_preds, rtol=5e-2)

    def test_model_quantize_onnx_without_dataset(self):
        model = EfficientNetB0(weights=None, input_shape=[224, 224, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        input_examples = np.random.random((100, 224, 224, 3)).astype(np.float32)
        input_features = np.random.randint(0, 10, size=100)

        # quantize a Keras model
        onnx_quantized_model = InferenceOptimizer.quantize(model,
                                                           accelerator='onnxruntime',
                                                           x=input_examples,
                                                           y=input_features,
                                                           thread_num=8,
                                                           accuracy_criterion={'relative': 0.99,
                                                                               'higher_is_better': True})

        preds = model.predict(input_examples, batch_size=5)
        onnx_preds = onnx_quantized_model.predict(input_examples, batch_size=5)
        np.testing.assert_allclose(preds, onnx_preds, rtol=5e-2)

    def test_model_quantize_onnx_with_only_x(self):
        model = EfficientNetB0(weights=None, input_shape=[224, 224, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        input_examples = np.random.random((100, 224, 224, 3)).astype(np.float32)
        preds = model.predict(input_examples, batch_size=5)

        # quantize a Keras model based on numpy array
        onnx_quantized_model = InferenceOptimizer.quantize(model,
                                                           accelerator='onnxruntime',
                                                           x=input_examples,
                                                           y=None,
                                                           thread_num=8,
                                                           accuracy_criterion={'relative': 0.99,
                                                                               'higher_is_better': True})
        onnx_preds = onnx_quantized_model.predict(input_examples, batch_size=5)
        np.testing.assert_allclose(preds, onnx_preds, rtol=5e-2)

        # quantize a Keras model based on dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(input_examples)
        onnx_quantized_model = InferenceOptimizer.quantize(model,
                                                           accelerator='onnxruntime',
                                                           x=train_dataset,
                                                           y=None,
                                                           thread_num=8,
                                                           accuracy_criterion={'relative': 0.99,
                                                                               'higher_is_better': True})
        onnx_preds = onnx_quantized_model.predict(input_examples, batch_size=5)
        np.testing.assert_allclose(preds, onnx_preds, rtol=5e-2)

        # quantize a Keras model based on tf tensor
        input_tensor = tf.convert_to_tensor(input_examples)
        onnx_quantized_model = InferenceOptimizer.quantize(model,
                                                           accelerator='onnxruntime',
                                                           x=input_tensor,
                                                           y=None,
                                                           thread_num=8,
                                                           accuracy_criterion={'relative': 0.99,
                                                                               'higher_is_better': True})
        onnx_preds = onnx_quantized_model.predict(input_examples, batch_size=5)
        np.testing.assert_allclose(preds, onnx_preds, rtol=5e-2)

    def test_model_quantize_onnx_save_load(self):
        model = EfficientNetB0(weights=None, input_shape=[224, 224, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        input_examples = np.random.random((100, 224, 224, 3)).astype(np.float32)
        input_features = np.random.randint(0, 10, size=100)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_examples,
                                                            input_features))

        # quantize a Keras model
        onnx_model = InferenceOptimizer.quantize(model,
                                                 accelerator='onnxruntime',
                                                 x=train_dataset,
                                                 thread_num=1,
                                                 accuracy_criterion={'relative': 0.99,
                                                                               'higher_is_better': True})
        preds1 = onnx_model.predict(input_examples, batch_size=5)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            onnx_model._save(tmp_dir_name)
            new_onnx_model = KerasONNXRuntimeModel._load(tmp_dir_name)

        assert new_onnx_model.session_options.intra_op_num_threads == 1
        assert new_onnx_model.session_options.inter_op_num_threads == 1

        preds2 = new_onnx_model.predict(input_examples, batch_size=5)
        np.testing.assert_almost_equal(preds1, preds2, decimal=5)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(onnx_model, tmp_dir_name)
            new_onnx_model = InferenceOptimizer.load(tmp_dir_name, model)

        assert new_onnx_model.session_options.intra_op_num_threads == 1
        assert new_onnx_model.session_options.inter_op_num_threads == 1

        preds2 = new_onnx_model.predict(input_examples, batch_size=5)
        np.testing.assert_almost_equal(preds1, preds2, decimal=5)
