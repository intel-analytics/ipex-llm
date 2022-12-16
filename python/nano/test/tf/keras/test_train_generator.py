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
from tensorflow import keras
from tensorflow.keras import layers
from bigdl.nano.tf.keras import Sequential, mixedprecision_train_step_generator


def test_train_step_generator():

    from bigdl.nano.tf import patch_tensorflow
    patch_tensorflow(precision='mixed_bfloat16')

    class Model():
        model = Sequential([layers.Dense(units=1, input_shape=[1])])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        def loss(self, y, pred):
            return tf.losses.mean_squared_error(y, pred)

        @tf.function
        def train(x, y):
            with tf.GradientTape() as tape:
                pred = self.model(x, training=True)
                loss_value = self.loss(y, pred)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss_value.numpy()

    model = Model()
    fit = mixedprecision_train_step_generator(model.model, model.loss, model.optimizer)

