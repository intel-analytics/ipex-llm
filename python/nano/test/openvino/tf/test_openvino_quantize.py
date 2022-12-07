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
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from bigdl.nano.tf.keras import Model


class TestOpenVINO(TestCase):
    def test_model_quantize_openvino(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

        # Case1: Trace and quantize
        openvino_model = model.trace(accelerator='openvino')
        openvino_quantized_model = openvino_model.quantize(accelerator='openvino',
                                                           x=train_dataset,
                                                           thread_num=8)

        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

        y_hat = openvino_quantized_model.predict(train_examples, batch_size=5)
        assert y_hat.shape == (100, 10)

        # Case2: Quantize directly from tensorflow
        openvino_quantized_model = model.quantize(accelerator='openvino',
                                                  x=train_dataset)
        
        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

        y_hat = openvino_quantized_model.predict(train_examples, batch_size=5)
        assert y_hat.shape == (100, 10)

        preds = model.predict(train_examples)
        openvino_preds = openvino_quantized_model.predict(train_examples)
        np.testing.assert_allclose(preds, openvino_preds, rtol=1e-2)

        # Case 3: with config
        openvino_quantized_model = model.quantize(accelerator='openvino',
                                                  x=train_dataset,
                                                  openvino_config={"PERFORMANCE_HINT": "LATENCY"})

        y_hat = openvino_quantized_model(train_examples[:10])
        assert y_hat.shape == (10, 10)

    def test_model_quantize_openvino_without_dataset(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))

        openvino_quantized_model = model.quantize(accelerator='openvino',
                                                  x=train_examples,
                                                  y=train_labels)

        preds = model.predict(train_examples)
        openvino_preds = openvino_quantized_model.predict(train_examples)
        np.testing.assert_allclose(preds, openvino_preds, rtol=1e-2)
