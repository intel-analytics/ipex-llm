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
    def test_model_trace_onnx(self):
        model = ResNet50(weights=None, input_shape=[224, 224, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 224, 224, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).batch(2)

        # trace a Keras model
        spec = tf.TensorSpec((None, 224, 224, 3), tf.float32)
        onnx_model = model.trace(accelerator='onnxruntime', input_sample=spec)

        y_hat = onnx_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

        y_hat = onnx_model.predict(train_examples, batch_size=5)
        assert y_hat.shape == (100, 10)

        onnx_model.compile(metrics=[tf.keras.metrics.CategoricalAccuracy()])
        acc = onnx_model.evaluate(train_dataset, return_dict=True)['categorical_accuracy']
