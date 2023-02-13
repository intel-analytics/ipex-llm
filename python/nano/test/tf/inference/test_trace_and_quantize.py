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
from tensorflow.keras.metrics import MeanSquaredError, CategoricalAccuracy
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

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
        from bigdl.nano.utils.common import compare_version
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

    def test_evaluate(self):
        inputs = tf.keras.Input(shape=(28*28,), name='digits')
        x = layers.Dense(10, name='dense_logits')(inputs)
        outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.RMSprop(),
                      metrics=CategoricalAccuracy())

        x = np.random.random((100, 28*28))
        y = np.random.randint(0, 10, 100)

        inc_q_model = InferenceOptimizer.quantize(model, x=x, y=y)
        inc_q_model.evaluate(x=x, y=y)

        ov_t_model = InferenceOptimizer.trace(model, accelerator="openvino")
        ov_t_model.evaluate(x=x, y=y)

        ov_q_model = InferenceOptimizer.quantize(model, accelerator="openvino", x=x, y=y)
        ov_q_model.evaluate(x=x, y=y)

        ort_t_model = InferenceOptimizer.trace(model, accelerator="onnxruntime")
        ort_t_model.evaluate(x=x, y=y)

        ort_q_model = InferenceOptimizer.quantize(model, accelerator="onnxruntime", x=x, y=y)
        ort_q_model.evaluate(x=x, y=y)

        # test save/load
        for m in [inc_q_model, ov_t_model, ov_q_model, ort_t_model, ort_q_model]:
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                InferenceOptimizer.save(m, tmp_dir_name)
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
        # for custom model, quantized model still return fp32 output
        # test at the same time that the quantization does not 
        # change the original's model dtype policy 
        model = MyModel(100)
        model.compile(loss='mse', metrics=MeanSquaredError())
        model_policys = []
        for layer in model.layers:
            model_policys.append(layer._dtype_policy)
        x = np.random.random((100, 4))
        model(x)

        bf16_model = InferenceOptimizer.quantize(model, precision="bf16")
        for idx, layer in enumerate(model.layers):
            assert layer._dtype_policy == model_policys[idx]

        from bigdl.nano.utils.common import _avx512_checker
        if _avx512_checker():
            output = bf16_model(x)
            assert output.dtype == tf.float32

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)

        if _avx512_checker():
            output = load_model(x)
            assert output.dtype == tf.float32

        # test standard model, quantized model return bf16 output
        # test at the same time that the quantization does not 
        # change the original's model dtype policy 
        model = MobileNetV2(weights="imagenet")
        ori_model_config = model.get_config()
        x = np.random.rand(32, 224, 224, 3)
        model(x)

        bf16_model = InferenceOptimizer.quantize(model, precision="bf16")
        assert ori_model_config == model.get_config()

        from bigdl.nano.utils.common import _avx512_checker
        if _avx512_checker():
            output = bf16_model(x)
            assert output.dtype == tf.bfloat16

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(bf16_model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)

        if _avx512_checker():
            output = load_model(x)
            assert output.dtype == tf.bfloat16
