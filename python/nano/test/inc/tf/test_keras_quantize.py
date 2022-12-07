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


import pytest
from unittest import TestCase
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from keras.utils.np_utils import to_categorical
import numpy as np
from bigdl.nano.tf.keras import Model


class TestModelQuantize(TestCase):
    def test_model_quantize_ptq(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()],)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_labels = to_categorical(train_labels, num_classes=10)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        # Case 1: Default
        q_model = model.quantize(x=train_dataset)
        assert q_model
        output = q_model(train_examples[0:10])
        assert output.shape == (10, 10)

        # Case 2: Override by arguments
        q_model = model.quantize(x=train_dataset,
                                 metric=tf.keras.metrics.CategoricalAccuracy(),
                                 tuning_strategy='basic',
                                 accuracy_criterion={'relative': 0.99,
                                                     'higher_is_better': True})
        assert q_model
        output = q_model(train_examples[0:10])
        assert output.shape == (10, 10)

        # Case 3: Invalid approach, dynamic or qat is not supported
        invalid_approach = 'dynamic'
        with pytest.raises(RuntimeError, match="Only 'static' approach is supported now."):
            model.quantize(x=None, approach=invalid_approach)

    def test_model_quantize_without_dataset(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        model = Model(inputs=model.inputs, outputs=model.outputs)
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_labels = to_categorical(train_labels, num_classes=10)

        q_model = model.quantize(x=train_examples, y=train_labels)
        assert q_model(train_examples[0:10]).shape == (10, 10)
