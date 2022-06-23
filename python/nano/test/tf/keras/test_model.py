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

import tensorflow as tf
from bigdl.nano.tf.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def test_functional_usage():
    model = MobileNetV2(weights=None, input_shape=[40, 40, 3], classes=10)
    model = Model(inputs=model.inputs, outputs=model.outputs)
    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
            self.dropout = tf.keras.layers.Dropout(0.5)

        def call(self, inputs, training=False):
            x = self.dense1(inputs)
            if training:
                x = self.dropout(x, training=training)
            return self.dense2(x)
    mymodel = MyModel()
