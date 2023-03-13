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
import tensorflow as tf
import tempfile
import os
from bigdl.nano.tf.keras import InferenceOptimizer
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model


class TestInferencePipeline(TestCase):
    def test_optimize_nano_model_without_accuracy(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        # prepare dataset
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        # prepare optimizer
        opt = InferenceOptimizer()
        opt.optimize(model=model,
                     x=train_dataset,
                     latency_sample_num=10,
                     thread_num=8)
        model = opt.get_best_model()
        assert isinstance(opt.optimized_model_dict["original"]["latency"], float)
        assert isinstance(opt.optimized_model_dict["static_int8"]["latency"], float)

    def test_optimize_model_without_accuracy(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        # prepare dataset
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        # prepare optimizer
        opt = InferenceOptimizer()
        opt.optimize(model=model,
                     x=train_dataset,
                     latency_sample_num=10)
        model = opt.get_best_model()

    def test_optimize_nano_model_without_accuracy_large_batch(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        # prepare dataset
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        # prepare optimizer
        opt = InferenceOptimizer()
        opt.optimize(model=model,
                     x=train_dataset,
                     batch_size=32,
                     latency_sample_num=10)
        model = opt.get_best_model()

    def test_optimize_model_with_accuracy(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        # prepare dataset
        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        # prepare optimizer
        opt = InferenceOptimizer()
        from tensorflow.keras.metrics import CategoricalAccuracy
        opt.optimize(model=model,
                     x=train_dataset,
                     validation_data=train_dataset, # for test
                     batch_size=32,
                     metric=CategoricalAccuracy(),
                     latency_sample_num=10)
        opt.summary()
        model = opt.get_best_model()

    def test_optimize_model_without_dataset(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)

        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))

        opt = InferenceOptimizer()
        opt.optimize(model=model,
                     x=train_examples,
                     y=train_labels,
                     latency_sample_num=10,
                     thread_num=8)
        model = opt.get_best_model()

    def test_optimize_model_with_only_x(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        # test numpy array
        train_examples = np.random.random((100, 40, 40, 3))
        opt = InferenceOptimizer()
        opt.optimize(model=model,
                     x=train_examples,
                     y=None,
                     latency_sample_num=10,
                     thread_num=8)
        model = opt.get_best_model()

        # test tf tensor
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        train_examples = tf.convert_to_tensor(train_examples)
        opt = InferenceOptimizer()
        opt.optimize(model=model,
                     x=train_examples,
                     y=None,
                     latency_sample_num=10,
                     thread_num=8)
        model = opt.get_best_model()

        # test dataset with only x
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        train_examples = np.random.random((100, 40, 40, 3))
        train_dataset = tf.data.Dataset.from_tensor_slices(train_examples)
        opt = InferenceOptimizer()
        opt.optimize(model=model,
                     x=train_examples,
                     y=None,
                     latency_sample_num=10,
                     thread_num=8)
        model = opt.get_best_model()

    def test_optimizer_save_load(self):
        model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
        # prepare dataset
        train_examples = np.random.random((100, 40, 40, 3))
        # save load for original model
        output1 = model(train_examples)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            InferenceOptimizer.save(model, tmp_dir_name)
            load_model = InferenceOptimizer.load(tmp_dir_name, model)
            output2 = load_model(train_examples)
            np.testing.assert_almost_equal(output1.numpy(), output2.numpy(), decimal=5)

    def test_compile_model(self):
        from tensorflow.keras.metrics import CategoricalAccuracy
        inputs = tf.keras.Input(shape=(28*28,), name='digits')
        x = layers.Dense(10, name='dense_logits')(inputs)
        outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
        # test model with metrics in subprocess
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.RMSprop(),
                      metrics=CategoricalAccuracy())

        x = np.random.random((100, 28*28))
        y = np.random.randint(0, 10, 100)

        infer_opt = InferenceOptimizer()
        infer_opt.optimize(model, x=x, y=y,
                           batch_size=1,
                           metric=CategoricalAccuracy(),
                           direction="max",
                           includes=["static_int8"])
        assert infer_opt.optimized_model_dict["static_int8"]["status"] == "successful"

        # test model without metrics in subprocess
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.RMSprop())

        infer_opt = InferenceOptimizer()
        infer_opt.optimize(model, x=x, y=y,
                           batch_size=1,
                           metric=CategoricalAccuracy(),
                           direction="max",
                           includes=["static_int8"])
        assert infer_opt.optimized_model_dict["static_int8"]["status"] == "successful"
