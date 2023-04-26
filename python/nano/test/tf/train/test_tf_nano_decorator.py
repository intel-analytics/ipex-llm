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
from tensorflow.keras import backend
from bigdl.nano.tf.keras import Sequential, nano_bf16


def test_tf_nano_bf16_decorator():

    from bigdl.nano.tf import patch_tensorflow, unpatch_tensorflow
    patch_tensorflow(precision='mixed_bfloat16')

    class Model():
        model = Sequential([layers.Dense(units=1, input_shape=[1])])
        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        def loss(self, y, pred):
            return tf.losses.mean_squared_error(y, pred)

        @nano_bf16
        @tf.function
        def train(self, x, y):
            with tf.GradientTape() as tape:
                pred = self.model(x, training=True)
                loss_value = self.loss(y, pred)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss_value

    model = Model()
    # If call model.train() on devices without BF16 instruction set, core dumped
    # x = np.random.random(1000)
    # y = np.random.random(1000)
    # model.train(x, y)
    unpatch_tensorflow()


def test_tf_nano_multiprocessing_customized_loop():
    from bigdl.nano.tf.keras import nano_multiprocessing, nano

    global_batch_size = 32
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    optimizer = tf.keras.optimizers.SGD()

    dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(128).batch(
        global_batch_size)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @nano_multiprocessing
    @tf.function
    def train_step(inputs, model, loss_object, optimizer):
        features, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @nano(num_processes=2)
    def train_whole_data(model, dataset, loss_object, optimizer, train_step):
        for inputs in dataset:
            print(train_step(inputs, model, loss_object, optimizer))

    train_whole_data(model, dataset, loss_object, optimizer, train_step)


def test_tf_nano_multiprocessing_customized_loss_datagenerator():
    # Test use cases of nano_multiprocessing_loss and dataset created by `from_generator`
    from bigdl.nano.tf.keras import nano_multiprocessing, nano, nano_multiprocessing_loss

    global_batch_size = 32
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    optimizer = tf.keras.optimizers.SGD()

    def dummy_data_generator():
        for i in range(128):
            yield tf.constant([i]), tf.constant([i])

    dataset = tf.data.Dataset.from_generator(dummy_data_generator,
                                             output_signature=(tf.TensorSpec(shape=(1,), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(1,), dtype=tf.float32)))
    # necessary to initiate dataset._GeneratorState
    dataset._GeneratorState = dataset._GeneratorState(dummy_data_generator)

    @nano_multiprocessing_loss()
    def loss_object(x, pred):
        res = backend.mean(tf.math.squared_difference(x, pred), axis=-1)
        return res

    @nano_multiprocessing
    @tf.function
    def train_step(inputs, model, loss_object, optimizer):
        features, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @nano(num_processes=2)
    def train_whole_data(model, dataset, loss_object, optimizer, train_step):
        for inputs in dataset:
            print(train_step(inputs, model, loss_object, optimizer))

    train_whole_data(model, dataset, loss_object, optimizer, train_step)
