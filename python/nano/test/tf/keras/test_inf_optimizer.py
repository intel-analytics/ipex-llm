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
from tensorflow.keras import Model
from bigdl.nano.tf.keras import Sequential
from bigdl.nano.tf.keras import InferenceOptimizer
import numpy as np
from bigdl.nano.tf.keras import Model as NanoModel
from tensorflow.keras.applications.resnet import ResNet50


class TestInferencePipeline(TestCase):
    def test_optimize_nano_model_without_accuracy(self):
        model = ResNet50(weights=None, input_shape=[40, 40, 3], classes=10)
        model = NanoModel(inputs=model.inputs, outputs=model.outputs)
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

    def test_optimize_model_without_accuracy(self):
        model = ResNet50(weights=None, input_shape=[40, 40, 3], classes=10)
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
        model = ResNet50(weights=None, input_shape=[40, 40, 3], classes=10)
        model = NanoModel(inputs=model.inputs, outputs=model.outputs)
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
        model = ResNet50(weights=None, input_shape=[40, 40, 3], classes=10)
        model = NanoModel(inputs=model.inputs, outputs=model.outputs)
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
        model = ResNet50(weights=None, input_shape=[40, 40, 3], classes=10)
        model = NanoModel(inputs=model.inputs, outputs=model.outputs)

        train_examples = np.random.random((100, 40, 40, 3))
        train_labels = np.random.randint(0, 10, size=(100,))

        opt = InferenceOptimizer()
        opt.optimize(model=model,
                     x=train_examples,
                     y=train_labels,
                     latency_sample_num=10,
                     thread_num=8)
        model = opt.get_best_model()
