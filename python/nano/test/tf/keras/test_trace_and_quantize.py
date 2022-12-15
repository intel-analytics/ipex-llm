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
# ==============================================================================

from unittest import TestCase

import numpy as np
import tensorflow as tf

from bigdl.nano.tf.keras import InferenceOptimizer


# used to test attributes access
class MyModel(tf.keras.Model):
    def __init__(self, x):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.x = x

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
        
    def get_x(self):
        return self.x

    @staticmethod
    def do_nothing():
        pass


class TestTraceAndQuantize(TestCase):
    def test_attribute_access_after_trace(self):
        x = 100
        model = MyModel(x)
        traced_model = InferenceOptimizer.trace(model, accelerator="onnxruntime",
                                                input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float64))
        # try to access some custom attributes
        traced_model.do_nothing()
        assert traced_model.get_x() == traced_model.x == x

    def test_attribute_access_after_quantize(self):
        x = 100
        model = MyModel(x)
        quantized_model = InferenceOptimizer.quantize(model,
                                                      accelerator="onnxruntime",
                                                      input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                      x=np.random.random((100, 4)),
                                                      y=np.random.random((100, 5)),
                                                      accuracy_criterion = {'relative': 0,
                                                                            'higher_is_better': True})
        # try to access some custom attributes
        quantized_model.do_nothing()
        assert quantized_model.get_x() == quantized_model.x == x
