#
# Copyright 2018 Analytics Zoo Authors.
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

from zoo.feature.common import ChainedPreprocessing, FeatureSet
from zoo.feature.image import *
from zoo.pipeline.api.net import TFOptimizer
from test.zoo.pipeline.utils.test_utils import ZooTestCase
import tensorflow as tf
import numpy as np
import os

from zoo.tfpark import KerasModel, TFDataset

resource_path = os.path.join(os.path.split(__file__)[0], "../resources")


class TestTFParkModel(ZooTestCase):

    def setup_method(self, method):
        tf.keras.backend.clear_session()
        super(TestTFParkModel, self).setup_method(method)

    def create_multi_input_output_model(self):
        data1 = tf.keras.layers.Input(shape=[10])
        data2 = tf.keras.layers.Input(shape=[10])

        x1 = tf.keras.layers.Flatten()(data1)
        x1 = tf.keras.layers.Dense(10, activation='relu')(x1)
        pred1 = tf.keras.layers.Dense(2, activation='softmax')(x1)

        x2 = tf.keras.layers.Flatten()(data2)
        x2 = tf.keras.layers.Dense(10, activation='relu')(x2)
        pred2 = tf.keras.layers.Dense(2)(x2)

        model = tf.keras.models.Model(inputs=[data1, data2], outputs=[pred1, pred2])
        model.compile(optimizer='rmsprop',
                      loss=['sparse_categorical_crossentropy', 'mse'])
        return model

    def create_training_data(self):
        np.random.seed(20)
        x = np.random.rand(20, 10)
        y = np.random.randint(0, 2, (20))
        return x, y

    def test_training_with_validation_data_distributed_multi_heads(self):

        keras_model = self.create_multi_input_output_model()
        model = KerasModel(keras_model)

        x, y = self.create_training_data()

        val_x, val_y = self.create_training_data()

        model.fit([x, x], [y, y], validation_data=([val_x, val_x], [val_y, val_y]),
                  batch_size=4, distributed=True)


if __name__ == "__main__":
    pytest.main([__file__])
