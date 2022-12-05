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
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from bigdl.nano.tf.keras import Model


class TestONNX(TestCase):
    def test_model_quantize_onnx(self):
        model = ResNet50(weights=None, input_shape=[224, 224, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        input_examples = np.random.random((100, 224, 224, 3))
        input_features = np.random.randint(0, 10, size=100)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((input_examples,
                                                            input_features))

        # quantize a Keras model
        onnx_quantized_model = model.quantize(accelerator='onnxruntime',
                                              x=train_dataset,
                                              thread_num=8)

        y_hat = onnx_quantized_model(input_examples[:10])
        assert y_hat.shape == (10, 10)

        y_hat = onnx_quantized_model.predict(input_examples, batch_size=5)
        assert y_hat.shape == (100, 10)

        preds = model.predict(input_examples)
        onnx_preds = onnx_quantized_model.predict(input_examples)
        np.testing.assert_allclose(preds, onnx_preds, rtol=1e-2)

    def test_model_quantize_onnx_without_dataset(self):
        model = ResNet50(weights=None, input_shape=[224, 224, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        input_examples = np.random.random((100, 224, 224, 3))
        input_features = np.random.randint(0, 10, size=100)

        # quantize a Keras model
        onnx_quantized_model = model.quantize(accelerator='onnxruntime',
                                              x=input_examples,
                                              y=input_features,
                                              thread_num=8)

        preds = model.predict(input_examples)
        onnx_preds = onnx_quantized_model.predict(input_examples)
        np.testing.assert_allclose(preds, onnx_preds, rtol=1e-2)
