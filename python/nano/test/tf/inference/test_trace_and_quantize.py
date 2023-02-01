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
import tempfile
import operator
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError

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


class MyModelReturnList(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)

    def call(self, inputs: tf.Tensor):
        return [self.dense1(inputs)]


class TestTraceAndQuantize(TestCase):
    def test_attribute_access_after_trace(self):
        x = 100
        # for onnxruntime
        model = MyModel(x)
        traced_model = InferenceOptimizer.trace(model, accelerator="onnxruntime",
                                                input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
        # try to access some custom attributes
        traced_model.do_nothing()
        assert traced_model.get_x() == traced_model.x == x
        traced_model(np.random.random((1, 4)).astype(np.float32))
        traced_model(inputs=np.random.random((1, 4)).astype(np.float32))

        # test save/load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(traced_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == traced_model.x == x
        
        # for openvino
        model = MyModel(x)
        traced_model = InferenceOptimizer.trace(model, accelerator="openvino",
                                                input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
        # try to access some custom attributes
        traced_model.do_nothing()
        assert traced_model.get_x() == traced_model.x == x
        traced_model(np.random.random((1, 4)).astype(np.float32))

        # test save/load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(traced_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == traced_model.x == x

    def test_attribute_access_after_quantize(self):
        x = 100
        # for onnxruntime
        model = MyModel(x)
        quantized_model = InferenceOptimizer.quantize(model,
                                                      accelerator="onnxruntime",
                                                      input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                      x=np.random.random((100, 4)),
                                                      y=np.random.random((100, 5)))
        # try to access some custom attributes
        quantized_model.do_nothing()
        assert quantized_model.get_x() == quantized_model.x == x
        quantized_model(np.random.random((1, 4)).astype(np.float32))
        quantized_model(inputs=np.random.random((1, 4)).astype(np.float32))
        
        # test save/load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(quantized_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == quantized_model.x == x

        # for openvino
        model = MyModel(x)
        quantized_model = InferenceOptimizer.quantize(model,
                                                      accelerator="openvino",
                                                      input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                      x=np.random.random((100, 4)),
                                                      y=np.random.random((100, 5)))
        # try to access some custom attributes
        quantized_model.do_nothing()
        assert quantized_model.get_x() == quantized_model.x == x
        quantized_model(np.random.random((1, 4)).astype(np.float32))

        # test save/load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(quantized_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == quantized_model.x == x
        
        # for inc
        from bigdl.nano.utils.util import compare_version
        INC_LESS_14 = compare_version("neural_compressor", operator.lt, "1.14")
        if INC_LESS_14:
            return
        model = MyModel(x)
        quantized_model = InferenceOptimizer.quantize(model,
                                                      accelerator=None,
                                                      input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                                      x=np.random.random((100, 4)),
                                                      y=np.random.random((100, 5)))
        # try to access some custom attributes
        quantized_model.do_nothing()
        assert quantized_model.get_x() == quantized_model.x == x
        quantized_model(np.random.random((1, 4)).astype(np.float32))

        # test save/load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(quantized_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.do_nothing()
        assert new_model.get_x() == quantized_model.x == x

    def test_evaluate_after_trace(self):
        # test onnxxruntime
        model = MyModel(100)
        model.compile(loss='mse', metrics=MeanSquaredError())
        x = np.random.random((100, 4))
        y = np.random.random((100, 5))

        traced_model = InferenceOptimizer.trace(model, accelerator="onnxruntime",
                                                input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
        traced_model.evaluate(x=x, y=y)

        # test save/load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(traced_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.evaluate(x=x, y=y)

        # test openvino
        model = MyModel(100)
        model.compile(loss='mse', metrics=MeanSquaredError())
        x = np.random.random((100, 4))
        y = np.random.random((100, 5))

        traced_model = InferenceOptimizer.trace(model, accelerator="openvino",
                                                input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
        traced_model.evaluate(x=x, y=y)

        # test save/load
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(traced_model, tmp_dir_name)
            new_model = InferenceOptimizer.load(tmp_dir_name, model)
        new_model.evaluate(x=x, y=y)

    def test_inference_output_shape(self):
        model = MyModelReturnList()
        x = np.random.random((100, 4))
        traced_model = InferenceOptimizer.trace(model, accelerator="onnxruntime",
                                                input_spec=tf.TensorSpec(shape=(None, 4), dtype=tf.float32))
        outputs = traced_model(x)
        assert isinstance(outputs, list) and isinstance(outputs[0], tf.Tensor)

        quantized_model = InferenceOptimizer.quantize(model, accelerator="onnxruntime", 
                                                      input_spec=tf.TensorSpec(shape=(None, 4)), x=x)
        outputs = quantized_model(x)
        assert isinstance(outputs, list) and isinstance(outputs[0], tf.Tensor)

    def test_quantize_bf16(self):
        model = MyModel(100)
        model.compile(loss='mse', metrics=MeanSquaredError())
        x = np.random.random((100, 4))
        model(x)

        traced_model = InferenceOptimizer.quantize(model, precision="bf16")

        from bigdl.nano.utils import CPUInfo
        cpuinfo = CPUInfo()
        if cpuinfo.has_bf16:
            model(x)

        InferenceOptimizer.save(model, "save_bf16")
        model = InferenceOptimizer.load("save_bf16", model)